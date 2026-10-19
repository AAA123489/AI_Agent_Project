"""
test_content_recheck.py — 「URL 不变、正文被改」的复检回归测试
============================================================
增量爬取按 URL 去重，学校改了正文却沿用原 URL 时，库里会一直是旧版本，
而且**没有任何报错**。pipeline._recheck_recent_changes 每天抽检最近入库的
文章：重抓 → 比 content_hash → 变了就重嵌入。

重点钉住三件事：
  1. 哈希变了才动库，没变不重复嵌入；
  2. **先嵌入新块、后删旧块**——反过来嵌入失败就会重演「标记在、正文没了」；
  3. 抓不到 / 死链时保留旧内容，绝不能拿空结果覆盖。
"""

import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper import pipeline
from campus_scraper.models import ArticleMetadata
from campus_scraper.storage import get_recent_scraped, init_db, mark_scraped

_URL = "https://www.hait.edu.cn/info/1/100.htm"


def _article(content_hash="newhash001", title="关于测试的通知"):
    return ArticleMetadata(
        title=title,
        publish_date="2026-10-18",
        category="通知公告",
        source_site="河南工学院",
        url=_URL,
        clean_text="河南工学院关于测试工作的通知内容。" * 20,
        content_hash=content_hash,
    )


class _Store:
    """假向量库：记录 get / save / delete 的调用顺序。"""

    def __init__(self, old_ids=("old1", "old2"), fail_save=False):
        self.log = []
        self.old_ids = list(old_ids)
        self.fail_save = fail_save

    def get_ids_by_url(self, url):
        self.log.append(("get", url))
        return list(self.old_ids)

    def save_documents(self, documents=None, metadatas=None, ids=None):
        if self.fail_save:
            raise RuntimeError("Chroma 挂了")
        self.log.append(("save", tuple(ids)))

    def delete_ids(self, ids):
        self.log.append(("delete", tuple(ids)))


def _env(monkeypatch, days=7, limit=5):
    monkeypatch.setenv("SCRAPE_RECHECK_DAYS", str(days))
    monkeypatch.setenv("SCRAPE_RECHECK_MAX", str(limit))


@pytest.fixture()
def env(monkeypatch, tmp_path):
    _env(monkeypatch)
    # 正文落盘改到临时目录，别往仓库的 scraped_docs 里写
    monkeypatch.setattr(pipeline, "_SCRAPED_DOCS_DIR", tmp_path / "scraped_docs")
    monkeypatch.setattr(pipeline, "_random_delay", lambda: 0.0)
    monkeypatch.setattr(
        pipeline, "get_recent_scraped",
        lambda *a, **kw: [{
            "url": _URL, "title": "关于测试的通知", "publish_date": "2026-10-18",
            "category": "通知公告", "content_hash": "oldhash001",
            "scraped_at": datetime.now().isoformat(),
        }],
    )
    return tmp_path


def _fake_scrape(article):
    async def _fake(**kw):
        return article

    return _fake


def test_哈希变更加入新块后再删旧块(env, monkeypatch):
    """顺序必须是 save → delete；反了就是「先拆旧房，再发现新房盖不起来」。"""
    store = _Store()
    marked = []
    monkeypatch.setattr(pipeline, "scrape_article", _fake_scrape(_article()))
    monkeypatch.setattr(pipeline, "mark_scraped", lambda **kw: marked.append(kw))

    checked, updated, chunks = asyncio.run(
        pipeline._recheck_recent_changes("通知公告", "河南工学院", None, store)
    )

    kinds = [k for k, _ in store.log]
    assert checked == 1 and updated == 1
    assert kinds == ["get", "save", "delete"], f"实际调用顺序: {store.log}"
    new_ids = store.log[1][1]
    assert all("newhash0" in i for i in new_ids), "新块 id 应带新哈希"
    assert store.log[2][1] == ("old1", "old2"), "删的必须是最初取到的旧 id"
    assert marked[0]["content_hash"] == "newhash001", "标记要更新成新哈希，否则下轮又判为变更"
    assert chunks == len(new_ids)


def test_哈希未变时不重复入库(env, monkeypatch):
    store = _Store()
    monkeypatch.setattr(pipeline, "scrape_article", _fake_scrape(_article("oldhash001")))
    monkeypatch.setattr(pipeline, "mark_scraped", lambda **kw: pytest.fail("不该落标记"))

    checked, updated, chunks = asyncio.run(
        pipeline._recheck_recent_changes("通知公告", "河南工学院", None, store)
    )

    assert (checked, updated, chunks) == (1, 0, 0)
    assert store.log == [], "内容没变就不该读写向量库"


def test_抓不到时保留旧内容(env, monkeypatch):
    """死链 / 404 / 超窗返回 None——旧内容比空内容有价值，绝不能覆盖。"""
    store = _Store()
    called = []

    async def fake(**kw):
        called.append(kw)
        return None

    monkeypatch.setattr(pipeline, "scrape_article", fake)
    monkeypatch.setattr(pipeline, "mark_scraped", lambda **kw: pytest.fail("不该落标记"))

    checked, updated, chunks = asyncio.run(
        pipeline._recheck_recent_changes("通知公告", "河南工学院", None, store)
    )

    assert len(called) == 1, "复检确实发了请求"
    assert (checked, updated, chunks) == (0, 0, 0)
    assert store.log == []


def test_重嵌入失败时不删旧块(env, monkeypatch):
    store = _Store(fail_save=True)
    monkeypatch.setattr(pipeline, "scrape_article", _fake_scrape(_article()))
    monkeypatch.setattr(pipeline, "mark_scraped", lambda **kw: pytest.fail("入库失败不能标记"))

    checked, updated, chunks = asyncio.run(
        pipeline._recheck_recent_changes("通知公告", "河南工学院", None, store)
    )

    assert (updated, chunks) == (0, 0)
    assert [k for k, _ in store.log] == ["get"], f"旧块必须保留，实际: {store.log}"


def test_天数为0时关闭复检(env, monkeypatch):
    _env(monkeypatch, days=0)
    store = _Store()
    monkeypatch.setattr(
        pipeline, "get_recent_scraped",
        lambda *a, **kw: pytest.fail("关闭时不该查库"),
    )
    monkeypatch.setattr(
        pipeline, "scrape_article",
        lambda **kw: pytest.fail("关闭时不该发请求"),
    )

    assert asyncio.run(
        pipeline._recheck_recent_changes("通知公告", "河南工学院", None, store)
    ) == (0, 0, 0)


# ── storage.get_recent_scraped ──────────────────────────────

def _insert(db, url, category, scraped_at, content_hash="h"):
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO scraped_articles
           (url, title, publish_date, category, source_site, content_hash, scraped_at)
           VALUES (?, 't', '2026-10-01', ?, '河南工学院', ?, ?)""",
        (url, category, content_hash, scraped_at),
    )
    conn.commit()
    conn.close()


def test_只取窗口内同分类且有哈希的文章(tmp_path):
    db = str(tmp_path / "cache.db")
    init_db(db)
    now = datetime.now()
    _insert(db, "u1", "通知公告", (now - timedelta(days=1)).isoformat())
    _insert(db, "u2", "通知公告", (now - timedelta(days=30)).isoformat())   # 出窗口
    _insert(db, "u3", "学校新闻", (now - timedelta(days=1)).isoformat())   # 别的分类
    _insert(db, "u4", "通知公告", (now - timedelta(days=1)).isoformat(), content_hash="")

    rows = get_recent_scraped("通知公告", days=7, limit=5, db_path=db)

    assert [r["url"] for r in rows] == ["u1"]


def test_按入库时间升序取前limit篇(tmp_path):
    """升序 = 先查快滑出窗口的，保证每篇在窗口内至少轮到一次。"""
    db = str(tmp_path / "cache.db")
    init_db(db)
    now = datetime.now()
    for i in range(4):
        _insert(db, f"u{i}", "通知公告", (now - timedelta(days=3 - i)).isoformat())

    rows = get_recent_scraped("通知公告", days=7, limit=2, db_path=db)

    assert [r["url"] for r in rows] == ["u0", "u1"], "应取最旧的 2 篇"


def test_mark_scraped重新入库不破坏查询(tmp_path):
    db = str(tmp_path / "cache.db")
    init_db(db)
    mark_scraped(url="u1", category="通知公告", content_hash="abc", db_path=db)

    rows = get_recent_scraped("通知公告", days=1, limit=5, db_path=db)

    assert [r["content_hash"] for r in rows] == ["abc"]
