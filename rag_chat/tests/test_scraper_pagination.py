"""
test_scraper_pagination.py — 分页 URL 拼接回归测试
====================================================
修复 bug：动易 CMS 的"下一页"链接常是裸路径（如 "310.htm"、"tzgg/81.htm"），
必须基于"当前列表页 URL"拼接（保留 /tzgg、/index 等目录前缀），
不能基于站点根目录 urljoin，否则会拼成 /310.htm → 404 提前停爬。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.scraper import resolve_next_url


def test_bare_page_number_keeps_tzgg_prefix():
    """裸路径页号：/tzgg/311.htm 的下一页是 /tzgg/310.htm，而非 /310.htm"""
    assert resolve_next_url(
        "https://www.hait.edu.cn/tzgg/311.htm", "310.htm"
    ) == "https://www.hait.edu.cn/tzgg/310.htm"


def test_bare_page_number_keeps_xyxw_prefix():
    """学校新闻：/xyxw/227.htm 的下一页是 /xyxw/226.htm"""
    assert resolve_next_url(
        "https://www.hait.edu.cn/xyxw/227.htm", "226.htm"
    ) == "https://www.hait.edu.cn/xyxw/226.htm"


def test_subdir_next_keeps_index_prefix():
    """子站带目录：教务处 /index/tzgg.htm 的下一页是 /index/tzgg/81.htm"""
    assert resolve_next_url(
        "https://jwc.hait.edu.cn/index/tzgg.htm", "tzgg/81.htm"
    ) == "https://jwc.hait.edu.cn/index/tzgg/81.htm"


def test_first_page_next_with_full_prefix():
    """首页下一页带完整前缀：/tzgg.htm → /tzgg/311.htm"""
    assert resolve_next_url(
        "https://www.hait.edu.cn/tzgg.htm", "/tzgg/311.htm"
    ) == "https://www.hait.edu.cn/tzgg/311.htm"


def test_absolute_url_passthrough():
    """绝对 URL 直接透传"""
    assert resolve_next_url(
        "https://www.hait.edu.cn/tzgg.htm",
        "https://www.hait.edu.cn/tzgg/310.htm",
    ) == "https://www.hait.edu.cn/tzgg/310.htm"
