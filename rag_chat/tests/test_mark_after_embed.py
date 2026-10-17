"""
test_mark_after_embed.py — 「标记 ⇒ 已入库」不变量回归测试
====================================================
爬取入库有两个阶段，落「已爬」标记的位置必须在**后**：

    爬取（逐篇）→ 落盘 .txt → 分块 → 嵌入 ChromaDB → 落「已爬」标记

这条顺序就是「标记 ⇒ 已入库」不变量。它一旦被反过来（标记写在爬取阶段），
就会出现一个横跨**整个分类**的危险窗口：

    标记了 ✓ —— 但正文还只在内存里 ✗

此时进程被杀（日常中断最常见的方式），标记留在库里、正文没了；而下一轮爬取的
is_scraped 会**静默跳过**这篇，于是它永远进不了知识库，光看库也发现不了。
2026-10-11 当天两次中途停机共丢了 20 篇（14 学校新闻 + 6 马克思主义学院新闻），
是靠「今天标记 137 篇 vs 库内实际 117 篇」的差额才查出来的。

放这些测试是为了钉住**不变量本身**，而不只是钉住某次修法：
  1. 标记必须晚于入库调用——钉的是**顺序**，不是条数
  2. 入库失败就不许标记——宁可下轮重爬，不可标记一个不在库里的东西
  3. 爬取阶段不许碰标记（防止有人把它加回去，那正是这次丢 20 篇的成因）
"""

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper import scraper as scraper_mod
from campus_scraper.models import ArticleMetadata
from campus_scraper.pipeline import _split_and_embed

_CATEGORY = {
    "name": "通知公告",
    "icon": "📢",
    "base_url": "https://www.hait.edu.cn",
    "list_path": "/tzgg.htm",
    "site_label": "河南工学院",
}


def _hash8(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def _article(url: str, text_len: int = 700) -> ArticleMetadata:
    """造一篇正文够长、必然切出块的文章。"""
    return ArticleMetadata(
        title=f"标题-{url}",
        publish_date="2026-10-11",
        category="通知公告",
        source_site="河南工学院",
        url=url,
        clean_text="河南工学院关于测试工作的通知内容。" * (text_len // 17 + 1),
        content_hash=hashlib.md5(url.encode()).hexdigest(),
    )


class _FakeStore:
    """假向量库：只记「谁被存了」，并可按序号注入失败。

    关键点——把 save 和 mark 写进**同一条日志**，这样测试才能断言两者的先后，
    而不只是各自的条数。
    """

    def __init__(self, log: list, fail_on_save: int | None = None):
        self.log = log
        self.fail_on_save = fail_on_save
        self.save_count = 0

    def save_documents(self, documents=None, metadatas=None, ids=None):
        if self.fail_on_save is not None and self.save_count == self.fail_on_save:
            raise RuntimeError("Chroma 挂了")
        self.save_count += 1
        # id 形如 scrape_{日期}_{hash8}_{i}（日期带横杠，不会把 split 拆歪）
        self.log.append(("save", ids[0].split("_")[2]))


def _marker(log: list):
    def _mark(**kw):
        log.append(("mark", kw["content_hash"][:8]))
    return _mark


def test_mark_happens_after_each_successful_save():
    """每篇都是「先入库、后标记」，逐篇交替——不是攒到最后一起标记。"""
    log = []
    urls = [f"https://www.hait.edu.cn/info/1/{i}.htm" for i in range(3)]
    articles = [_article(u) for u in urls]

    _split_and_embed(articles, _FakeStore(log), mark_fn=_marker(log))

    assert [kind for kind, _ in log] == ["save", "mark"] * 3, (
        f"入库与标记必须逐篇交替且入库在前，实际: {log}"
    )
    assert [h for k, h in log if k == "mark"] == [_hash8(u) for u in urls]


def test_embed_failure_leaves_no_mark():
    """第 2 篇入库时炸 → 只有第 1 篇有标记，第 2、3 篇绝不能标记。

    这是整个不变量的要害：标记 = 「这篇已进知识库」的承诺。入库没成功就不能承诺，
    宁可下一轮重爬一遍——重爬只是白干，标记了却没有内容才是**永久丢失**。
    """
    log = []
    urls = [f"https://www.hait.edu.cn/info/2/{i}.htm" for i in range(3)]
    articles = [_article(u) for u in urls]

    with pytest.raises(RuntimeError):
        _split_and_embed(articles, _FakeStore(log, fail_on_save=1), mark_fn=_marker(log))

    marks = [h for k, h in log if k == "mark"]
    assert marks == [_hash8(urls[0])], (
        f"入库失败的篇目被标记了 → 它以后永远进不了库，实际标记: {marks}"
    )


def test_marks_once_per_article_not_per_chunk():
    """一篇切成多块也只标记一次（标记的单位是文章，不是块）。"""
    log = []
    article = _article("https://www.hait.edu.cn/info/3/long.htm", text_len=2000)

    n_chunks = _split_and_embed([article], _FakeStore(log), mark_fn=_marker(log))

    assert n_chunks >= 3, f"这篇应切出多块，否则本测试失去意义（实际 {n_chunks} 块）"
    assert [k for k, _ in log] == ["save", "mark"]


def test_scrape_category_does_not_mark(monkeypatch):
    """爬取阶段一律不落标记——标记只能由入库阶段落。

    这条防的是「把标记搬回爬取阶段」的回归，那正是 2026-10-11 丢 20 篇的成因：
    标记跨着整个分类提前落下，分类跑到一半被杀就永久丢正文。
    """
    marked: list[dict] = []

    async def fake_list_page(url, base_url, session):
        return (
            [{"url": f"https://www.hait.edu.cn/p1/{i}.htm",
              "title": f"t{i}", "date": "2026-10-01"} for i in range(3)],
            {"total": 0, "current": 1, "total_pages": 1, "next_url": None},
        )

    async def fake_scrape_article(url, category, source_site, session, fallback_date=""):
        return ArticleMetadata(
            title=url, publish_date="2026-10-01", category=category,
            source_site=source_site, url=url, clean_text="正文", content_hash="h",
        )

    monkeypatch.setattr(scraper_mod, "scrape_list_page", fake_list_page)
    monkeypatch.setattr(scraper_mod, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(scraper_mod, "is_scraped", lambda u: False)
    monkeypatch.setattr(scraper_mod, "mark_scraped", lambda **kw: marked.append(kw))
    monkeypatch.setattr(scraper_mod, "init_db", lambda: None)
    monkeypatch.setattr(scraper_mod, "_random_delay", lambda: 0.0)

    articles = asyncio.run(
        scraper_mod.scrape_category(_CATEGORY, session=None, max_pages=1)
    )

    assert len(articles) == 3, "抓取本身要正常"
    assert marked == [], (
        f"爬取阶段落了「已爬」标记 → 又会出现「标记了但没入库」的窗口，实际: {marked}"
    )
