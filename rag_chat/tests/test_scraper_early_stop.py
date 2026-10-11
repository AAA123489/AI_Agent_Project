"""
test_scraper_early_stop.py — 「遇到已爬内容即停止翻页」回归测试
====================================================
背景（2026-10-11）：列表页是**跨页严格新→旧**的连续序列（实测通知公告
第 1/2/3 页分别是 10-10~09-23 / 09-22~09-03 / 09-02~08-05，无缝衔接）。
因此一旦某页出现「已爬过」的 URL，就说明时间线已接上缓存，后面的页只会更旧。

这条判据让 max_pages 从「每天固定要付出的成本」变成「真积压时的上限」：
改动前每页之间无条件休息 3~8 分钟，日常增量空跑要 1.3~2 小时；
改动后日常跑完第 1 页就停，约 3~5 分钟。

放这些测试是为了钉住两件事：
  1. 该停的时候要停（否则白等几小时）
  2. 不该停的时候绝不能停——早停会**静默漏抓**内容，比慢更糟
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper import scraper as scraper_mod
from campus_scraper.models import ArticleMetadata

_CATEGORY = {
    "name": "通知公告",
    "icon": "📢",
    "base_url": "https://www.hait.edu.cn",
    "list_path": "/tzgg.htm",
    "site_label": "河南工学院",
}


def _page(n: int, prefix: str) -> list[dict]:
    """造一页 n 条文章。"""
    return [
        {"url": f"https://www.hait.edu.cn/{prefix}/{i}.htm",
         "title": f"{prefix}-{i}", "date": "2026-10-01"}
        for i in range(n)
    ]


async def _run(monkeypatch, pages: list[list[dict]], scraped: set[str], max_pages: int):
    """跑一次 scrape_category，返回 (抓到的文章, 实际请求的列表页数)。

    列表页按调用顺序从 pages 里取；全部依赖打桩，不发任何真实请求、不碰真实库。
    """
    fetched_pages: list[int] = []

    async def fake_list_page(url, base_url, session):
        idx = len(fetched_pages)
        fetched_pages.append(idx + 1)
        arts = pages[idx] if idx < len(pages) else []
        nxt = "next.htm" if idx + 1 < len(pages) else None
        return arts, {"total": 0, "current": idx + 1, "total_pages": len(pages),
                      "next_url": nxt}

    async def fake_scrape_article(url, category, source_site, session):
        return ArticleMetadata(
            title=url, publish_date="2026-10-01", category=category,
            source_site=source_site, url=url, clean_text="正文", content_hash="h",
        )

    monkeypatch.setattr(scraper_mod, "scrape_list_page", fake_list_page)
    monkeypatch.setattr(scraper_mod, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(scraper_mod, "is_scraped", lambda u: u in scraped)
    monkeypatch.setattr(scraper_mod, "mark_scraped", lambda **kw: None)
    monkeypatch.setattr(scraper_mod, "init_db", lambda: None)
    monkeypatch.setattr(scraper_mod, "_random_delay", lambda: 0.0)
    # 页间长休息改成 0 秒，否则测试要真睡 3~8 分钟
    monkeypatch.setattr(scraper_mod, "BATCH_REST_MIN", 0)
    monkeypatch.setattr(scraper_mod, "BATCH_REST_MAX", 0)

    articles = await scraper_mod.scrape_category(
        _CATEGORY, session=None, max_pages=max_pages,
    )
    return articles, len(fetched_pages)


def test_stops_at_first_page_when_already_caught_up(monkeypatch):
    """日常增量：第 1 页有 2 篇新的 + 3 篇已爬 → 抓完第 1 页就停，不翻页。"""
    page1 = _page(5, "p1")
    scraped = {a["url"] for a in page1[2:]}          # 后 3 篇已爬
    articles, n_pages = asyncio.run(_run(
        monkeypatch, [page1, _page(5, "p2"), _page(5, "p3")], scraped, max_pages=3,
    ))
    assert n_pages == 1, "已追上进度时不应继续翻页"
    assert len(articles) == 2, "第 1 页的新内容必须全部抓到"


def test_stops_immediately_when_nothing_new(monkeypatch):
    """完全追上：第 1 页全是已爬 → 只请求 1 页、0 篇新增。"""
    page1 = _page(5, "p1")
    scraped = {a["url"] for a in page1}
    articles, n_pages = asyncio.run(_run(
        monkeypatch, [page1, _page(5, "p2")], scraped, max_pages=3,
    ))
    assert n_pages == 1
    assert articles == []


def test_keeps_paging_while_pages_are_all_new(monkeypatch):
    """积压追赶：连续两页全是新内容 → 必须继续翻，不能提前停（否则静默漏抓）。"""
    articles, n_pages = asyncio.run(_run(
        monkeypatch,
        [_page(4, "p1"), _page(4, "p2"), _page(4, "p3")],
        scraped=set(),
        max_pages=3,
    ))
    assert n_pages == 3, "全新增时必须翻满上限"
    assert len(articles) == 12


def test_stops_on_older_page_after_overflow(monkeypatch):
    """积压但未到底：第 1、2 页全新，第 3 页出现已爬 → 停在第 3 页。"""
    p1, p2, p3 = _page(4, "p1"), _page(4, "p2"), _page(4, "p3")
    scraped = {a["url"] for a in p3[:2]}             # 第 3 页混入 2 篇旧的
    articles, n_pages = asyncio.run(_run(
        monkeypatch, [p1, p2, p3, _page(4, "p4")], scraped, max_pages=5,
    ))
    assert n_pages == 3, "应在接上缓存的那一页停止，而不是跑满 max_pages"
    assert len(articles) == 10                       # 4 + 4 + 2


def test_max_pages_is_a_ceiling(monkeypatch):
    """max_pages=1（日常默认）时，即使全是新内容也只跑 1 页。"""
    articles, n_pages = asyncio.run(_run(
        monkeypatch, [_page(4, "p1"), _page(4, "p2")], scraped=set(), max_pages=1,
    ))
    assert n_pages == 1
    assert len(articles) == 4
