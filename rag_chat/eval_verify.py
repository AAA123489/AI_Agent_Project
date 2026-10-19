"""eval_verify.py — 用规则核对回答里的「可验证事实」是否落在检索来源里

## 为什么不用 LLM 打分

`eval_baseline.py` + `eval_score.py` 那套要过**第二个 LLM**，样本一多就贵、
而且同一份代码跑两遍分数会抖（实测 60/60 与 50/60）。日期、数字、链接这三类
事实**不需要语义判断**：答案里出现的日期/数字/URL，要么能在检索来源的原文里
找到，要么就是模型自己编的。规则检查确定、免费、可进 CI。

## 检查项

1. **日期**：`2026-06-16`、`2026年6月16日`、`6月16日` → 来源正文/元数据里必须存在
   （无年份的「6月16日」按 MM-DD 匹配，来源可能只写了月日）。
2. **数量**：`42%`、`3 天`、`1000 元`、`12 人` 等带单位数字 → 来源里必须存在。
   问题里本来就有的数字（用户给定条件）不算模型主张，跳过。
3. **链接**：答案里引用的 URL 必须出现在检索来源的 `url` 里。

## 用法

    ../.venv/Scripts/python.exe eval_verify.py eval_results_baseline_topk8.json
    ../.venv/Scripts/python.exe eval_verify.py results.json --allow-rate 0.95 --json out.json

退出码：未落地的条目超过 `--allow-rate` 允许范围时为 1（可直接用于 CI）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 日期 ──
_RE_DATE_CN = re.compile(
    r"(?P<y>20\d{2})\s*年\s*(?P<m>\d{1,2})\s*月(?:\s*(?P<d>\d{1,2})\s*日)?"
)
_RE_DATE_ISO = re.compile(r"(?P<y>20\d{2})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})")
_RE_DATE_MD = re.compile(r"(?<![\d年])(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日")

# ── 数量：只收带单位的，避免把「第3条」这类序数误判成事实 ──
_RE_QUANTITY = re.compile(
    r"(?<![\d.])(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>%|％|万元|亿元|元|人|名|天|小时|分钟|公里|千米)"
)

_RE_URL = re.compile(r"https?://[^\s<>()\"'）]+")


def _norm_date(y: str | None, m: str, d: str | None) -> str:
    """统一成 YYYY-MM-DD；没有年份/日时分别退化为 YYYY-MM / MM-DD。"""
    mm = f"{int(m):02d}"
    if not y:
        return f"{mm}-{int(d):02d}" if d else mm
    if not d:
        return f"{y}-{mm}"
    return f"{y}-{mm}-{int(d):02d}"


def extract_dates(text: str) -> set[str]:
    """抽取可比较的日期集合（YYYY-MM-DD / YYYY-MM / MM-DD）。"""
    dates: set[str] = set()
    for m in _RE_DATE_CN.finditer(text):
        dates.add(_norm_date(m.group("y"), m.group("m"), m.group("d")))
    for m in _RE_DATE_ISO.finditer(text):
        dates.add(_norm_date(m.group("y"), m.group("m"), m.group("d")))
    for m in _RE_DATE_MD.finditer(text):
        dates.add(_norm_date(None, m.group("m"), m.group("d")))
    return dates


def extract_quantities(text: str) -> set[str]:
    """抽取「数字+单位」的事实，返回去掉千分位逗号的规范化字符串。"""
    out: set[str] = set()
    for m in _RE_QUANTITY.finditer(text):
        num = m.group("num").replace(",", "")
        unit = "%" if m.group("unit") == "％" else m.group("unit")
        out.add(f"{num}{unit}")
    return out


def _source_corpus(result: dict) -> str:
    """把一条评测结果的所有来源（正文 + 元数据日期/链接）拼成可检索文本。"""
    parts: list[str] = []
    for src in result.get("sources") or []:
        parts.append(str(src.get("text") or ""))
        parts.append(str(src.get("date") or ""))
        parts.append(str(src.get("url") or ""))
    return "\n".join(parts)


def _source_urls(result: dict) -> set[str]:
    return {str(s.get("url")) for s in (result.get("sources") or []) if s.get("url")}


# 拒答文案的特征：没有来源是**正确**的，不能算「答案无出处」。
_REFUSAL_MARKERS = (
    "没有", "未找到", "无法回答", "无法提供", "不在知识库", "知识库中收录",
    "建议您前往", "请前往该校官网", "只收录河南工学院",
)


def looks_like_refusal(answer: str) -> bool:
    return any(m in answer for m in _REFUSAL_MARKERS)


@dataclass
class ResultCheck:
    """单题核对结果。"""
    index: int
    question: str
    ungrounded_dates: list[str] = field(default_factory=list)
    ungrounded_quantities: list[str] = field(default_factory=list)
    bad_urls: list[str] = field(default_factory=list)
    no_source: bool = False
    refusal: bool = False
    exempt_dates: list[str] = field(default_factory=list)  # 运行日期，非知识库事实

    @property
    def ok(self) -> bool:
        return not (self.ungrounded_dates or self.ungrounded_quantities
                    or self.bad_urls or self.no_source)


def check_result(index: int, result: dict, run_date: str | None = None) -> ResultCheck:
    answer = str(result.get("answer") or "")
    question = str(result.get("question") or "")
    corpus = _source_corpus(result)
    corpus_dates = extract_dates(corpus)
    # 数字与单位之间常夹空格/千分位逗号（「3 天」「1,200 人」），比较前统一去掉
    corpus_joined = re.sub(r"[\s,]", "", corpus)

    check = ResultCheck(index=index, question=question)
    if not result.get("sources"):
        # 拒答本来就不需要来源；只有「声称有答案却拿不出来源」才算失败
        if looks_like_refusal(answer):
            check.refusal = True
        else:
            check.no_source = True

    # 1) 日期：答案里每个日期都要在来源中出现；来源只写月日时按 MM-DD 匹配
    for d in sorted(extract_dates(answer)):
        if run_date and d == run_date:
            # 「当前已是 2026-09-16」来自 system prompt 的当天日期，不是知识库事实
            check.exempt_dates.append(d)
            continue
        candidates = {d}
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            candidates.add(d[5:])          # 来源可能省略年份
        if not (candidates & corpus_dates):
            check.ungrounded_dates.append(d)

    # 2) 数量：跳过问题里给定/本身含有的数字
    question_nums = {m.group("num").replace(",", "") for m in _RE_QUANTITY.finditer(question)}
    for q in sorted(extract_quantities(answer)):
        num = re.match(r"[\d.]+", q).group(0)
        if num in question_nums:
            continue
        if q not in corpus_joined:
            check.ungrounded_quantities.append(q)

    # 3) 链接：答案里引用的 URL 必须是检索来源
    urls = _source_urls(result)
    for url in sorted(set(_RE_URL.findall(answer))):
        url = url.rstrip(".,;，。；）)】]\"'")
        if url not in urls:
            check.bad_urls.append(url)

    return check


def verify_file(path: Path) -> list[ResultCheck]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise SystemExit(f"{path} 不是评测结果格式（缺少 results 数组）")

    # 运行日期：答案里的「当前已是 YYYY-MM-DD」来自 system prompt，不是知识库事实
    run_date = None
    if isinstance(data, dict):
        ts = str(data.get("timestamp") or "")[:10]
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", ts):
            run_date = ts
    return [check_result(i, r, run_date=run_date) for i, r in enumerate(results, 1)]


def _print_report(checks: list[ResultCheck], source_path: Path) -> float:
    total = len(checks)
    passed = sum(1 for c in checks if c.ok)
    refusals = sum(1 for c in checks if c.refusal)
    print(f"核对 {source_path}：{passed}/{total} 题全部事实可追溯"
          f"（其中 {refusals} 题为拒答，无需来源）\n")
    for c in checks:
        if c.ok:
            continue
        print(f"[{c.index}] {c.question[:60]}")
        if c.no_source:
            print("  ✗ 答案没有检索来源")
        for d in c.ungrounded_dates:
            print(f"  ✗ 日期不在来源里: {d}")
        for q in c.ungrounded_quantities:
            print(f"  ✗ 数字不在来源里: {q}")
        for u in c.bad_urls:
            print(f"  ✗ 链接不是检索来源: {u}")
    return passed / total if total else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="规则核对评测结果的事实可追溯性")
    parser.add_argument("path", type=Path, help="eval_baseline 产出的 JSON")
    parser.add_argument("--allow-rate", type=float, default=1.0,
                        help="允许的题目通过率下限（默认 1.0，全部可追溯）")
    parser.add_argument("--json", type=Path, default=None, help="把逐题结果写成 JSON")
    args = parser.parse_args()

    checks = verify_file(args.path)
    rate = _print_report(checks, args.path)

    if args.json:
        args.json.write_text(
            json.dumps([{
                "index": c.index, "question": c.question, "ok": c.ok,
                "ungrounded_dates": c.ungrounded_dates,
                "ungrounded_quantities": c.ungrounded_quantities,
                "bad_urls": c.bad_urls, "no_source": c.no_source,
                "refusal": c.refusal, "exempt_dates": c.exempt_dates,
            } for c in checks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\n逐题结果已写入 {args.json}")

    if rate < args.allow_rate:
        print(f"\n通过率 {rate:.0%} 低于阈值 {args.allow_rate:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
