"""
test_real_page_snapshots.py — 用官网真实页面快照守住解析器
==========================================================
背景（2026-10-19）：其余爬虫测试的 HTML 都是手写的，只能验证「我以为
官网长什么样」。官网改版、换模板、加一栏公告时，手写用例照样全绿，
线上却可能整类文章日期全空、翻页停在第一页。

这里保存了 6 份**真实页面**（2026-10-19 抓取）：

| 文件 | 来源 | 覆盖的模板 |
|---|---|---|
| main_notice.html | www.hait.edu.cn/tzgg.htm | 主站通知列表（日期在行内） |
| jwc_notice.html | jwc.hait.edu.cn/index/tzgg.htm | 教务处列表（详情页无日期，靠列表兜底） |
| zn_notice.html | znxy.hait.edu.cn/index/tzgg.htm | 学院列表 + 置顶公告 |
| clx_index.html | clxnew.hait.edu.cn/index.htm | 首页多频道混排（需 url_contains 过滤） |
| detail_notice.html | www.hait.edu.cn/info/1278/18600.htm | 动易详情页 |
| dead_link.html | jwc.hait.edu.cn/info/1887/11128.htm | 死链返回 200 + 导航壳 |

**刷新方式**（官网改版后手动重抓，别改断言去迁就解析错误）：

    ../.venv/Scripts/python.exe - <<'PY'
    import asyncio, pathlib
    from campus_scraper.scraper import fetch_page
    ...
    PY

注意必须用 `campus_scraper.scraper.fetch_page`（OpenSSL SECLEVEL=1）：
Windows 自带的 curl/Schannel 协商不了 hait.edu.cn 的 SHA-1 套件。
"""

import asyncio
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper import scraper as scraper_mod
from campus_scraper.html_parser import (
    extract_article_urls, extract_pagination, parse_html,
)
from campus_scraper.scraper import filter_list_articles

_PAGES = pathlib.Path(__file__).resolve().parent / "fixtures" / "pages"


def _html(name: str) -> str:
    return (_PAGES / f"{name}.html").read_text(encoding="utf-8")


def _list(name: str, base_url: str) -> tuple[list[dict], dict]:
    html = _html(name)
    return extract_article_urls(html, base_url), extract_pagination(html)


def test_main_notice_list_page():
    """主站通知：18 条全部带日期，翻页指向真实的下一页。"""
    articles, pagination = _list("main_notice", "https://www.hait.edu.cn")

    assert len(articles) == 18
    assert all(a.get("date") for a in articles), "列表页日期全部丢失"
    assert all("/info/" in a["url"] for a in articles)
    assert pagination.get("next_url", "").endswith("tzgg/314.htm")


def test_jwc_notice_list_page():
    """教务处列表：15 条全部带日期（详情页没日期，这一页是唯一来源）。"""
    articles, pagination = _list("jwc_notice", "https://jwc.hait.edu.cn")

    assert len(articles) == 15
    assert all(a.get("date") for a in articles)
    assert pagination.get("next_url", "").endswith("tzgg/82.htm")


def test_zn_notice_list_page_keeps_pinned_item():
    """智能工程学院列表：置顶的旧公告与 15 条正文都在，且都有日期。"""
    articles, _ = _list("zn_notice", "https://znxy.hait.edu.cn")

    assert len(articles) == 15
    assert all(a.get("date") for a in articles)
    assert any("巡 察 公 告" in a.get("title", "") for a in articles)


def test_clx_homepage_filter_and_pagination():
    """材料学院首页：多频道混排 → 过滤后只剩本栏目；「更多」箭头不是下一页。"""
    articles, pagination = _list("clx_index", "https://clxnew.hait.edu.cn")

    assert len(articles) == 50          # 首页确实混着多个频道
    filtered = filter_list_articles(articles, "info/1147/")
    assert len(filtered) == 4           # 本栏目文章
    assert all(a.get("date") for a in filtered)
    assert pagination.get("next_url") is None, "文章链接被误判成下一页"


def test_detail_page_parses_title_date_body():
    """动易详情页：标题、发布日期、正文都要认出来。"""
    text, meta = parse_html(
        _html("detail_notice"),
        "https://www.hait.edu.cn/info/1278/18600.htm",
    )

    assert meta["title"] == "关于2026年端午节放假安排的通知"
    assert meta["publish_date"] == "2026-06-16"
    assert "6月19日" in text and "端午节" in text


def test_dead_link_snapshot_is_rejected(monkeypatch):
    """真实死链快照（HTTP 200 + 「没有找到文章ID」）不能被当成文章。"""

    async def fake_fetch(url, session=None):
        return _html("dead_link")

    monkeypatch.setattr(scraper_mod, "fetch_page", fake_fetch)
    article = asyncio.run(scraper_mod.scrape_article(
        url="https://jwc.hait.edu.cn/info/1887/11128.htm",
        category="教务处通知",
        source_site="教务处",
        session=None,
    ))

    assert article is None
