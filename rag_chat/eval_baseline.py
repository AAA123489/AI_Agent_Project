"""
eval_baseline.py — 6 题基准评测（真实链路：检索 + LLM）
====================================================
对固定 6 题跑 AgentLoop（真实系统链路：检索 top_k 读 .env TOP_K + LLM），
收集每题最终回答 + 检索来源，落盘成 json 供 eval_score.py 打分。

用法:
    python eval_baseline.py                      # 跑基准 6 题（默认 rerank=off，对齐 60/60 那次的配置）
    python eval_baseline.py --rerank on          # 按当前 .env 的配置跑（重排开）
    python eval_baseline.py --top-k 5 --tag topk5   # 消融对照，输出 eval_results_baseline_topk5.json
    python eval_baseline.py --out my.json        # 自定义输出文件

注意：TOP_K / RETRIEVAL_MODE / RETRIEVAL_RERANK 是 app_backend 的模块级常量，
import 时一次性求值，所以必须在 import 之前写进 os.environ 才生效（import 后再改没用）。
"""

import os

# ── 离线加载：必须在 import 模型相关模块之前设置 ──
# 不设的话 sentence-transformers 每次都会请求 hf-mirror 做在线校验，
# 网络异常时挂起甚至 segfault（见 docs/改进记录.md）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="6 题基准评测（真实链路）")
    p.add_argument(
        "--rerank",
        choices=["on", "off"],
        default="off",
        help="重排开关，默认 off（对齐 60/60 那次的配置；当前 .env 里是 on）",
    )
    p.add_argument("--top-k", type=int, default=None, help="检索返回片段数，默认读 .env（8）")
    p.add_argument(
        "--mode",
        choices=["hybrid", "vector"],
        default=None,
        help="检索模式，默认读 .env（hybrid=向量+BM25 RRF 融合）",
    )
    p.add_argument("--tag", default=None, help="实验标签，输出文件名后缀（如 topk8_gate）")
    p.add_argument("--out", default=None, help="输出 json 路径，覆盖 --tag 推导的名字")
    return p.parse_args()


_ARGS = _parse_args()

# 必须在 import app_backend 之前落到 os.environ：load_dotenv 默认不覆盖已存在的环境变量
os.environ["RETRIEVAL_RERANK"] = _ARGS.rerank
if _ARGS.top_k is not None:
    os.environ["TOP_K"] = str(_ARGS.top_k)
if _ARGS.mode is not None:
    os.environ["RETRIEVAL_MODE"] = _ARGS.mode

if __name__ == "__main__":
    # Windows 控制台 GBK 兼容（只在作为脚本运行时改 stdout；放模块级会劫持
    # import 本模块者的 stdout，pytest 拆捕获流时崩）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

from app_backend import AgentLoop, ThinkingEvent, ToolCallEvent, ToolResultEvent, TextEvent, DoneEvent, ErrorEvent, get_active_params  # noqa: E402


QUESTIONS = [
    "2026年端午节放假是怎么安排的？",
    "河南工学院第九届运动会是什么时候举办？",
    "中国国际大学生创新大赛（2026）河南工学院校赛的报名截止时间是什么时候？",
    "学校最近有安排暑期社会实践吗？具体怎么做？",
    "2026年春季学期学校在学生心理健康方面有哪些安排？",
    "河北工学院的录取分数线是多少？",
]


async def run_one(loop: AgentLoop, q: str) -> dict:
    thinking = []
    sources = []
    answer = ""
    async for ev in loop.run_stream(q):
        if isinstance(ev, ThinkingEvent):
            thinking.append(ev.step)
        elif isinstance(ev, ToolCallEvent):
            pass
        elif isinstance(ev, ToolResultEvent):
            if ev.sources:
                sources = ev.sources
        elif isinstance(ev, TextEvent):
            # TextEvent 是逐 chunk 推送的，必须累加（照抄 AgentLoop.run 的做法）；
            # 直接赋值只会留下最后一片，评测结果会失真
            answer += ev.content
        elif isinstance(ev, ErrorEvent):
            answer = f"[错误] {ev.message}"
    return {
        "question": q,
        "answer": answer,
        "sources": sources,
        "thinking": thinking,
    }


async def main() -> None:
    if _ARGS.out:
        out_path = _ARGS.out
    elif _ARGS.tag:
        out_path = f"eval_results_baseline_{_ARGS.tag}.json"
    else:
        out_path = "eval_results_baseline.json"

    print(f"配置: {get_active_params()}")
    loop = AgentLoop()
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'='*60}\n[{i}/{len(QUESTIONS)}] {q}\n{'='*60}")
        r = await run_one(loop, q)
        print(f"回答: {r['answer'][:300]}")
        print(f"来源数: {len(r['sources'])}")
        results.append(r)

    payload = {
        "config": get_active_params(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 完成，结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
