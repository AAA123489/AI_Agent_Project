"""
eval_baseline.py — 6 题纯向量基线评测
===============================
对每组测试题跑 AgentLoop（真实系统链路：检索 top_k 读 .env TOP_K + LLM），
收集每题最终回答 + 检索来源，落盘到 eval_results/ 供后续对比。

用法:
    python eval_baseline.py                  # 跑基准 6 题
    python eval_baseline.py <json输出路径>   # 自定义输出文件
"""

import asyncio
import io
import json
import logging
import sys
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

from app_backend import AgentLoop, ThinkingEvent, ToolCallEvent, ToolResultEvent, TextEvent, DoneEvent, ErrorEvent, get_active_params


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
            answer = ev.content
        elif isinstance(ev, ErrorEvent):
            answer = f"[错误] {ev.message}"
    return {
        "question": q,
        "answer": answer,
        "sources": sources,
        "thinking": thinking,
    }


async def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "eval_results_baseline.json"
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
