"""
test_html_parser_title.py — <title> 站点后缀的截断边界
====================================================
背景（2026-10-17）：补回历史缺失文章时发现，教务处多篇标题只剩半句
（「关于公布2025」「关于组织开展2025」）。

根因是 `parse_html` 用 `re.sub(r"[-—|].*$", ...)` 从**第一个**连字符截断，
而标题里的学年区间「2025-2026」也含连字符：

    关于公布2025-2026学年第二学期开放实验项目结项名单的通知-教务处（学位办、语委办）
    └─ 按第一个 - 截 → 「关于公布2025」  ✗

站点后缀只出现在**最后一个**分隔符之后，且不含数字。按这个判据截断，
学年区间能保住，后缀照样去掉。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.html_parser import _strip_site_suffix, parse_html


def _page(title: str) -> str:
    return f"<html><head><title>{title}</title></head><body><p>正文</p></body></html>"


def test_year_range_in_title_survives_suffix_strip():
    """标题里的 2025-2026 不能被当成站点后缀截掉。"""
    raw = "关于公布2025-2026学年第二学期开放实验项目结项名单的通知-教务处（学位办、语委办）"

    title = _strip_site_suffix(raw)

    assert title == "关于公布2025-2026学年第二学期开放实验项目结项名单的通知"


def test_plain_site_suffix_is_stripped():
    """普通站点后缀（欢迎访问河南工学院）仍要删掉。"""
    raw = "关于组织开展2025-2026学年第一学期学生评教工作的通知-欢迎访问河南工学院"

    assert _strip_site_suffix(raw) == "关于组织开展2025-2026学年第一学期学生评教工作的通知"


def test_title_without_suffix_is_unchanged():
    """没有站点后缀、且连字符后带数字的标题，一个字都不能动。"""
    raw = "关于征集2025-2026学年教育教学改革项目的通知"

    assert _strip_site_suffix(raw) == raw


def test_parse_html_uses_fixed_title():
    """parse_html 走 <title> 兜底时同样不能腰斩。"""
    _, meta = parse_html(
        _page("关于公布2025-2026学年第二学期开放实验项目结项名单的通知-教务处"),
        "https://jwc.hait.edu.cn/info/1887/11211.htm",
    )

    assert meta["title"] == "关于公布2025-2026学年第二学期开放实验项目结项名单的通知"


# ── 栏目名 h1 不能顶替文章标题（2026-10-19）─────────────────────────────
# 电缆工程学院模板：<h1>通知公告</h1>，真标题只在 <title>。
# 修之前 27 篇文章（118 块）的 source 是「通知公告」「学院新闻」。


def _page_with_h1(title: str, h1: str | None) -> str:
    body = f"<h1>{h1}</h1>" if h1 is not None else ""
    return f"<html><head><title>{title}</title></head><body>{body}<p>正文</p></body></html>"


def test_column_h1_does_not_override_title():
    """h1 是栏目名时，标题取 <title>（真实文章标题）。"""
    _, meta = parse_html(
        _page_with_h1(
            "学术讲座——《探索原子世界：原子成像与精准操纵》-电缆工程学院",
            "通知公告",
        ),
        "https://dlxy.hait.edu.cn/info/1003/8412.htm",
    )

    assert meta["title"] == "学术讲座——《探索原子世界：原子成像与精准操纵》"


def test_real_h1_title_still_wins():
    """h1 是正常文章标题、<title> 只是站名时，保留 h1。"""
    _, meta = parse_html(
        _page_with_h1("河南工学院", "关于开展2025年实验室安全检查的通知"),
        "https://example.edu.cn/info/1.htm",
    )

    assert meta["title"] == "关于开展2025年实验室安全检查的通知"


def test_long_notice_title_is_not_treated_as_column():
    """「关于…的通知」这类完整标题即便以「通知」结尾也不能当栏目名。"""
    real = "关于组织开展2025-2026学年第一学期学生评教工作的通知"
    _, meta = parse_html(
        _page_with_h1(real + "-教务处", real),
        "https://jwc.hait.edu.cn/info/1887/1.htm",
    )

    assert meta["title"] == real
