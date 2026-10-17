"""test_time_range.py — 相对时间 → 日期区间。

为什么要有这一层：用户问「昨天有没有新的通知」时，向量检索只会按「通知」
捞旧文，BM25 也没有「昨天」对应的文档；必须先把相对时间落成 publish_date
区间，检索才知道该看哪一段。
"""

import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.time_range import resolve_time_range, strip_time_phrase

NOW = date(2026, 10, 17)  # 星期六


def _range(query):
    return resolve_time_range(query, now=NOW)


def test_yesterday_is_exact_day():
    tr = _range("昨天有没有新的通知")
    assert (tr.start, tr.end, tr.kind) == ("2026-10-16", "2026-10-16", "exact")
    assert tr.phrase == "昨天"


def test_today_and_day_before_yesterday():
    assert _range("今天有什么新闻").start == "2026-10-17"
    assert _range("前天有什么通知").start == "2026-10-15"


def test_recent_defaults_to_30_days():
    tr = _range("最近的通知消息是什么")
    assert (tr.start, tr.end) == ("2026-09-18", "2026-10-17")
    assert tr.kind == "recent"


def test_recent_n_days_and_weeks():
    assert _range("最近7天有什么通知").start == "2026-10-11"
    assert _range("近两周的新闻").start == "2026-10-04"


def test_week_month_year_windows():
    assert (_range("本周的通知").start, _range("本周的通知").end) == ("2026-10-12", "2026-10-17")
    assert (_range("上周的新闻").start, _range("上周的新闻").end) == ("2026-10-05", "2026-10-11")
    assert (_range("本月的通知").start, _range("本月的通知").end) == ("2026-10-01", "2026-10-17")
    assert (_range("上个月的通知").start, _range("上个月的通知").end) == ("2026-09-01", "2026-09-30")
    assert (_range("今年的新闻").start, _range("今年的新闻").end) == ("2026-01-01", "2026-10-17")
    assert (_range("去年的新闻").start, _range("去年的新闻").end) == ("2025-01-01", "2025-12-31")


def test_explicit_date_and_month():
    tr = _range("2026年10月16日有什么通知")
    assert (tr.start, tr.end, tr.kind) == ("2026-10-16", "2026-10-16", "exact")
    tr = _range("2026年9月的新闻")
    assert (tr.start, tr.end, tr.kind) == ("2026-09-01", "2026-09-30", "month")


def test_no_time_word_returns_none():
    assert _range("河南工学院有哪些二级学院") is None
    assert _range("转专业的报名条件") is None


def test_strip_time_phrase_leaves_topic_words():
    tr = _range("昨天有没有新的通知")
    assert "昨天" not in strip_time_phrase("昨天有没有新的通知", tr)
    assert "通知" in strip_time_phrase("昨天有没有新的通知", tr)
