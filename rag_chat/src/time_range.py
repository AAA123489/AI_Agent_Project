"""相对时间解析 —— 把「昨天 / 最近 / 本周」换成具体日期区间。

## 解决什么问题

检索层原本只把 query 原样丢给向量 + BM25。用户问「昨天有没有新的通知」时：

- 向量检索按语义找「通知」，返回的是一堆旧通知；
- BM25 把「昨天」当成普通词，没有任何文档命中；
- 文档元数据里明明有 `publish_date`，却从没参与过滤。

结果 LLM 只能对着旧通知说「无法确认昨天有没有新通知」。同样的问题也出现在
「最近有什么新闻」「上周的通知」上——**相对时间词必须先落到日期区间**，
检索才知道该看哪一段。

## 口径

- 今天/昨天/前天 → 单日精确匹配
- 本周/上周/本月/上月/今年/去年 → 自然周期
- 最近/近来/近期/近日/最新 → 最近 `RECENT_WINDOW_DAYS` 天（默认 30，可用 .env 调）
- 最近 7 天 / 近两周 / 最近一个月 → 按用户说的窗口
- 2026-10-16 / 2026年10月16日 / 2026年10月 → 显式日期

返回的 `TimeRange.phrase` 是命中的原始词，调用方可以把它从检索 query 里删掉：
「昨天」这类词对向量和 BM25 都只是噪声，日期过滤已经承担了它的语义。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# 兜底窗口：用户只说「最近/最新」时，往前看多少天
RECENT_WINDOW_DAYS = int(os.getenv("RECENT_WINDOW_DAYS", "30"))


@dataclass(frozen=True)
class TimeRange:
    """闭区间 [start, end]，日期均为 YYYY-MM-DD。"""

    start: str
    end: str
    label: str          # 展示用：昨天 / 最近30天 / 2026年10月
    kind: str           # exact | week | month | year | recent
    phrase: str         # 命中的原始词，供调用方从 query 中删除

    def contains(self, iso_date: str) -> bool:
        """ISO 日期字符串是否落在区间内（空日期一律 False）。"""
        return bool(iso_date) and self.start <= iso_date <= self.end


def _iso(d: date) -> str:
    return d.isoformat()


def date_to_int(iso_date: str) -> int:
    """YYYY-MM-DD → YYYYMMDD，供 Chroma 数值范围过滤；空/非法日期返回 0。

    Chroma 的元数据比较只认数字，$gte/$lte 传日期字符串会直接报
    「Expected operand value to be an int or a float」。所以入库时多存一个
    date_int，检索时用数字区间过滤；publish_date 仍保留给展示和 BM25。
    """
    if not iso_date:
        return 0
    digits = str(iso_date).replace("-", "").strip()
    return int(digits) if digits.isdigit() and len(digits) == 8 else 0


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _prev_month(d: date) -> date:
    return _month_start(d) - timedelta(days=1)


def resolve_time_range(
    query: str, now: datetime | date | None = None
) -> TimeRange | None:
    """从 query 中解析相对/显式时间，返回日期区间；没有时间词则 None。

    `now` 仅用于测试注入，生产不传。
    """
    if not query:
        return None
    today = now.date() if isinstance(now, datetime) else (now or datetime.now().date())

    # ── 1. 最近 N 天 / 近 N 天 / 过去 N 天 ──
    m = re.search(r"(?:最近|近|过去)\s*(\d{1,3})\s*天", query)
    if m:
        n = max(1, min(int(m.group(1)), 3650))
        start = today - timedelta(days=n - 1)
        return TimeRange(_iso(start), _iso(today), f"最近{n}天", "recent", m.group(0))

    # ── 2. 最近一周 / 近两周 / 最近一个月 ──
    m = re.search(r"(?:最近|近)\s*(一|两|二|三|1|2|3|7)\s*(周|个?星期|个月|月)", query)
    if m:
        num = {"一": 1, "两": 2, "二": 2, "三": 3, "1": 1, "2": 2, "3": 3, "7": 7}[m.group(1)]
        unit = m.group(2)
        if "月" in unit:
            # 自然月窗口：最近 N 个自然月（含本月）
            y, mo = today.year, today.month
            mo -= num - 1
            while mo <= 0:
                mo += 12
                y -= 1
            start = date(y, mo, 1)
            label = f"最近{num}个月"
        else:
            start = today - timedelta(days=7 * num - 1)
            label = f"最近{num}周"
        return TimeRange(_iso(start), _iso(today), label, "recent", m.group(0))

    # ── 3. 单日 ──
    for word, offset, kind in (
        ("今天", 0, "exact"), ("今日", 0, "exact"),
        ("昨天", 1, "exact"), ("昨日", 1, "exact"),
        ("前天", 2, "exact"),
    ):
        if word in query:
            d = today - timedelta(days=offset)
            return TimeRange(_iso(d), _iso(d), word, kind, word)

    # ── 4. 周 ──
    if any(w in query for w in ("本周", "这周", "本星期", "这个星期")):
        start = today - timedelta(days=today.weekday())
        return TimeRange(_iso(start), _iso(today), "本周", "week", "本周")
    if any(w in query for w in ("上周", "上星期", "上个星期")):
        this_monday = today - timedelta(days=today.weekday())
        start = this_monday - timedelta(days=7)
        end = this_monday - timedelta(days=1)
        return TimeRange(_iso(start), _iso(end), "上周", "week", "上周")

    # ── 5. 月 ──
    if any(w in query for w in ("本月", "这个月")):
        return TimeRange(
            _iso(_month_start(today)), _iso(today), "本月", "month", "本月"
        )
    if any(w in query for w in ("上月", "上个月")):
        prev = _prev_month(today)
        return TimeRange(
            _iso(_month_start(prev)), _iso(prev), "上个月", "month", "上个月"
        )

    # ── 6. 年 ──
    if any(w in query for w in ("今年", "本年")):
        return TimeRange(
            _iso(date(today.year, 1, 1)), _iso(today), "今年", "year", "今年"
        )
    if "去年" in query:
        y = today.year - 1
        return TimeRange(_iso(date(y, 1, 1)), _iso(date(y, 12, 31)), "去年", "year", "去年")

    # ── 7. 显式日期：2026-10-16 / 2026年10月16日 ──
    m = re.search(r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?", query)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            d = None
        if d:
            return TimeRange(_iso(d), _iso(d), _iso(d), "exact", m.group(0))

    # ── 8. 显式月份：2026年10月 ──
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", query)
    if m:
        try:
            y, mo = int(m.group(1)), int(m.group(2))
            start = date(y, mo, 1)
            end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        except ValueError:
            start = end = None
        if start and end:
            return TimeRange(_iso(start), _iso(end), f"{y}年{mo}月", "month", m.group(0))

    # ── 9. 泛化的「最近/最新」：兜底窗口 ──
    m = re.search(r"最近|近来|近期|近日|这几天|这阵子|这段时间|最新", query)
    if m:
        start = today - timedelta(days=RECENT_WINDOW_DAYS - 1)
        return TimeRange(
            _iso(start), _iso(today),
            f"最近{RECENT_WINDOW_DAYS}天", "recent", m.group(0),
        )

    return None

def strip_time_phrase(query: str, tr: TimeRange) -> str:
    """从 query 删掉命中的时间词，留下主题词给检索用。"""
    if not tr or not tr.phrase:
        return query.strip()
    return query.replace(tr.phrase, " ", 1).strip(" ，。！？、,.!?")
