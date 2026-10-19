"""
test_html_parser_remove_keywords.py — 导航块删除规则的误删回归
============================================================
_should_remove 原先是「class/id 里出现关键词子串就删」，于是：
    research-content  ⊃ search   → 正文容器被当搜索框删掉
    unrelated        ⊃ related  → 被当相关推荐删掉
    photonews        ⊃ hotnews  → 被当热门新闻删掉
删完正文只剩 0 字，scrape_article 会当成「无正文」静默跳过——文章永远进不了库，
而且日志里只有一行 debug 级跳过，很难发现。

改为「按非字母数字切 token + token 前缀匹配」后，必须同时保住两件事：
  1. 动易无分隔符的 searchstyle1092668 仍要删（前缀）；
  2. research / unrelated / photonews / shared 不能再删。
"""

import sys
from pathlib import Path

from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.html_parser import _should_remove


def _tag(html):
    return BeautifulSoup(html, "html.parser").find()


def _removed(class_=None, id_=None):
    attrs = ""
    if class_:
        attrs += f' class="{class_}"'
    if id_:
        attrs += f' id="{id_}"'
    return _should_remove(_tag(f"<div{attrs}>正文</div>"))


# ── 必须仍然删除 ────────────────────────────────────────────

def test_无分隔符的搜索容器仍被删除():
    """动易模板实测类名：searchstyle1092668（依赖前缀匹配，不能改成纯 token 相等）。"""
    assert _removed(class_="searchstyle1092668")


def test_常见导航类名仍被删除():
    assert _removed(class_="nav-menu")
    assert _removed(class_="main_header")
    assert _removed(class_="footer-wrap")
    assert _removed(id_="sidebar")
    assert _removed(class_="news-share")
    assert _removed(class_="breadcrumb-area")
    assert _removed(class_="pagination")
    assert _removed(class_="pagebar")


def test_热门新闻与相关推荐仍被删除():
    assert _removed(class_="hotnews-list")
    assert _removed(class_="recommend-box")
    assert _removed(class_="related-posts")


# ── 不能再误删 ──────────────────────────────────────────────

def test_research不再被当成search():
    assert not _removed(class_="research-content")
    assert not _removed(class_="research")


def test_unrelated不再被当成related():
    assert not _removed(class_="unrelated")


def test_photonews不再被当成hotnews():
    """photoNews.lower() = photonews，旧子串规则会命中 hotnews。"""
    assert not _removed(class_="photonews")


def test_share开头的普通词不过度删除():
    assert not _removed(class_="shared-link")
    assert not _removed(class_="sharing-settings")
    assert not _removed(class_="shareholder-info")


def test_正常正文容器不受影响():
    assert not _removed(class_="vsb_content")
    assert not _removed(id_="vsb_content")
    assert not _removed(class_="article-body")
