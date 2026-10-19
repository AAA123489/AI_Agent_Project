"""stress_retrieval.py — 检索层并发压测（不调用 LLM）

## 为什么单独压检索

一次问答的成本里，LLM 是主要开销、也是限流点；但**并发下先崩的通常是检索**：
Chroma 客户端、sentence-transformers 的 encode、jieba 分词、BM25 索引都是
进程内单例，多线程共享。这里绕开 LLM，直接压 `_retrieve_candidates`，
把「检索本身能不能扛住并发」与「模型 API 的额度/限流」分开测。

## 测什么

| 用例 | 期望 |
|---|---|
| 普通主题问题 | 有候选，不抛异常 |
| 带年份问题 | 有候选 |
| 相对时间泛问（最近有什么新闻） | 有候选（走日期倒序短接） |
| 外校问题 | 无候选 + `short_circuit` 含「库外题拦截」 |
| 罕见词问题 | 有候选 |

## 用法

    ../.venv/Scripts/python.exe stress_retrieval.py
    ../.venv/Scripts/python.exe stress_retrieval.py --workers 16 --rounds 5
    ../.venv/Scripts/python.exe stress_retrieval.py --json stress.json

只读 Chroma，**不需要停服务**。首次运行要加载嵌入模型和 BM25 索引（约 10~30 秒），
这部分在计时之外先做一次预热。

**注意**：这是进程内线程并发，不是多进程。要测多实例/网络层，请另测 HTTP。
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app_backend import _retrieve_candidates, TOP_K  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)  # 默认 DEBUG 会刷屏

# (查询, 期望)：normal = 必须有候选；refuse = 必须库外拦截
_CASES: list[tuple[str, str]] = [
    ("2026年端午节放假是怎么安排的？", "normal"),
    ("学生评教工作什么时候开始", "normal"),
    ("转专业的条件和流程是什么", "normal"),
    ("最近有什么新闻", "normal"),
    ("2026年教坛新秀评选有哪些要求", "normal"),
    ("挑战杯获奖名单", "normal"),
    ("河北工学院的录取分数线是多少？", "refuse"),
    ("清华大学有哪些专业", "refuse"),
]

_errors: list[str] = []
_lock = threading.Lock()


def _run_case(query: str, expect: str) -> float:
    t0 = time.perf_counter()
    try:
        out = _retrieve_candidates(query, top_k=TOP_K)
    except Exception as e:  # 并发下任何异常都要暴露，而不是被线程池吞掉
        with _lock:
            _errors.append(f"异常 {query!r}: {e!r}")
        return time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    candidates = out.get("candidates") or []
    short_circuit = out.get("short_circuit") or ""

    with _lock:
        if expect == "refuse":
            if "库外题拦截" not in short_circuit:
                _errors.append(f"外校题未被拦截: {query!r} → {short_circuit[:60]!r}")
        elif not candidates:
            _errors.append(f"正常题无候选: {query!r} → {short_circuit[:60]!r}")
    return elapsed


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p)))
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层并发压测（不调 LLM）")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    parser.add_argument("--rounds", type=int, default=3, help="每个用例重复轮数（默认 3）")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索 top_k")
    parser.add_argument("--json", type=Path, default=None, help="把统计结果写成 JSON")
    args = parser.parse_args()

    print(f"预热检索（加载嵌入模型 + BM25 索引）...")
    warm = time.perf_counter()
    _retrieve_candidates("河南工学院", top_k=args.top_k)
    print(f"预热完成: {time.perf_counter() - warm:.1f}s\n")

    tasks = [q for q, _ in _CASES] * args.rounds
    expects = {q: e for q, e in _CASES}
    print(f"开始压测: {len(tasks)} 次检索 / {args.workers} 并发 "
          f"（{len(_CASES)} 用例 × {args.rounds} 轮）\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        latencies = list(pool.map(lambda q: _run_case(q, expects[q]), tasks))
    wall = time.perf_counter() - t0

    latencies.sort()
    summary = {
        "workers": args.workers,
        "rounds": args.rounds,
        "calls": len(tasks),
        "wall_seconds": round(wall, 3),
        "qps": round(len(tasks) / wall, 2) if wall else 0,
        "p50_ms": round(_pct(latencies, 0.50) * 1000, 1),
        "p95_ms": round(_pct(latencies, 0.95) * 1000, 1),
        "p99_ms": round(_pct(latencies, 0.99) * 1000, 1),
        "max_ms": round(max(latencies) * 1000, 1) if latencies else 0,
        "mean_ms": round(statistics.fmean(latencies) * 1000, 1) if latencies else 0,
        "errors": len(_errors),
    }

    print(f"耗时: 墙钟 {summary['wall_seconds']}s，吞吐 {summary['qps']} 次/秒")
    print(f"延迟: p50 {summary['p50_ms']}ms  p95 {summary['p95_ms']}ms  "
          f"p99 {summary['p99_ms']}ms  max {summary['max_ms']}ms")
    if _errors:
        print(f"\n❌ {len(_errors)} 个问题：")
        for e in _errors[:20]:
            print("  -", e)
    else:
        print("\n✅ 全部用例通过（外校拦截生效，正常题均有候选）")

    if args.json:
        args.json.write_text(json.dumps({**summary, "issues": _errors},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n统计已写入 {args.json}")

    sys.exit(1 if _errors else 0)


if __name__ == "__main__":
    main()
