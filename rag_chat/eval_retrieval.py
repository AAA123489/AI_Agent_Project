"""
eval_retrieval.py — 向量 vs 混合检索的确定性消融
================================================
为什么需要它（2026-10-19）：
  README 里「混合检索优于纯向量」只有一句结论，`eval_hybrid.py` 已丢失；
  25/50/100 题随机评测的题库和脚本也都没了。论文里要写消融，必须有能复算的
  数据：固定题库 + 固定指标 + 一条命令跑出两档对照。

口径：
  - 题库 = eval_sets/qa_50.json（build_eval_set.py 生成，seed 固定）
  - 只打检索层 `_retrieve_candidates`，**不调回答模型、不调打分模型**——
    结果是确定性的，重跑不会因为 LLM 抖动而变化。
  - 正例 40 题按「期望原文 URL」判定命中：同 URL 的任意块进前 K 名即算命中。
    召回自检不参与（它会在检索后拒答/重写，属于另一层，见 eval_guard_probe.py）。
  - 指标：
      Hit@1/3/5/8  期望文章是否出现在前 K 个候选块里
      MRR          期望文章第一次出现的名次倒数（按 URL 去重后算，块级会虚高）
      Evidence@8   answer_span 是否逐字出现在前 8 个候选块文本里
                   （Hit 只说明找到文章，Evidence 说明答案句真的进了上下文）
      误拦率       正例被「库外主体」闸门拦下的数量（正常应为 0）
  - 负例 10 题只观察库外闸门命中数；是否拒答由召回自检负责。

两档怎么跑：
  RETRIEVAL_MODE 是 app_backend 的模块级常量，`_retrieve_candidates` 在调用时
  读全局，所以同一进程里改 `app_backend.RETRIEVAL_MODE` 即可切换，不用起两次。
  进程跑完即退出，不需要恢复。

用法:
    python eval_retrieval.py                  # 跑 vector + hybrid 两档
    python eval_retrieval.py --mode hybrid    # 只跑一档
    python eval_retrieval.py --questions eval_sets/qa_50.json --tag final
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SET = PROJECT_ROOT / "eval_sets" / "qa_50.json"
K_LIST = (1, 3, 5, 8)


def _norm(s: str) -> str:
    """答案片段比对：去空白与全半角标点差异（与 build_eval_set.py 一致）。"""
    return re.sub(r"\s+", "", s or "").replace("，", ",").replace("。", ".")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="向量 vs 混合检索消融")
    p.add_argument("--questions", default=str(DEFAULT_SET))
    p.add_argument("--mode", choices=("vector", "hybrid"), default=None,
                   help="只跑一档；默认两档都跑")
    p.add_argument("--tag", default="", help="结果文件名后缀")
    return p.parse_args()


def _run_mode(mode: str, positives: list[dict], negatives: list[dict]) -> dict:
    import app_backend

    app_backend.RETRIEVAL_MODE = mode
    rows = []
    for i, q in enumerate(positives, 1):
        t0 = time.perf_counter()
        env = app_backend._retrieve_candidates(q["question"], top_k=app_backend.TOP_K)
        elapsed = time.perf_counter() - t0
        cands = env["candidates"]
        texts = [c.get("text", "") for c in cands]
        urls = [(c.get("metadata") or {}).get("url", "") for c in cands]

        hit_rank = next((r for r, u in enumerate(urls, 1)
                         if u and u == q["expected_url"]), None)
        # 文档级名次：同一篇文章的多个块只算第一次出现
        doc_rank = None
        seen_urls: list[str] = []
        for u in urls:
            if u and u not in seen_urls:
                seen_urls.append(u)
        if q["expected_url"] in seen_urls:
            doc_rank = seen_urls.index(q["expected_url"]) + 1

        span = _norm(q.get("answer_span", ""))
        evidence_rank = next((r for r, t in enumerate(texts, 1)
                              if span and span in _norm(t)), None)

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "category": q.get("category", ""),
            "hit_rank": hit_rank,
            "doc_rank": doc_rank,
            "evidence_rank": evidence_rank,
            "short_circuit": env.get("short_circuit"),
            "n_candidates": len(cands),
            "elapsed_ms": round(elapsed * 1000, 1),
            # 失败样本的排查线索：命中了哪些文章（按名次）
            "top_urls": urls[:10],
            "expected_url": q["expected_url"],
        })
        flag = "✓" if hit_rank else "✗"
        print(f"  [{i:02d}/{len(positives)}] {flag} hit={hit_rank} "
              f"evid={evidence_rank} {q['question'][:42]}")

    # ── 负例：只观察库外闸门（非本校问题应该被 short_circuit 拦下）──
    neg_rows = []
    for q in negatives:
        env = app_backend._retrieve_candidates(q["question"], top_k=app_backend.TOP_K)
        neg_rows.append({
            "id": q["id"],
            "question": q["question"],
            "kind": q.get("kind", ""),
            "short_circuit": env.get("short_circuit"),
            "n_candidates": len(env["candidates"]),
        })

    def _rate(pred) -> float:
        return sum(1 for r in rows if pred(r)) / max(1, len(rows))

    summary = {
        "mode": mode,
        "top_k": app_backend.TOP_K,
        "n_positive": len(rows),
        "hit_at": {k: round(_rate(lambda r, k=k: r["hit_rank"] is not None and r["hit_rank"] <= k), 4)
                   for k in K_LIST},
        "mrr": round(sum(1.0 / r["doc_rank"] for r in rows if r["doc_rank"]) / max(1, len(rows)), 4),
        "evidence_at_8": round(_rate(lambda r: r["evidence_rank"] is not None and r["evidence_rank"] <= 8), 4),
        "positive_short_circuit": sum(1 for r in rows if r["short_circuit"]),
        "negative_gate_blocked": sum(1 for r in neg_rows if r["short_circuit"]),
        "avg_ms": round(sum(r["elapsed_ms"] for r in rows) / max(1, len(rows)), 1),
    }
    return {"summary": summary, "positives": rows, "negatives": neg_rows}


def main() -> None:
    args = _parse_args()
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    positives = [q for q in data["questions"] if q["type"] == "positive"]
    negatives = [q for q in data["questions"] if q["type"] == "negative"]
    modes = [args.mode] if args.mode else ["vector", "hybrid"]

    print(f"题库 {args.questions}：正例 {len(positives)} + 负例 {len(negatives)}"
          f"（seed={data.get('seed')}，构建时 {data.get('kb_chunks_at_build')} 块）")

    results = {}
    for mode in modes:
        print(f"\n──── RETRIEVAL_MODE={mode} ────")
        results[mode] = _run_mode(mode, positives, negatives)

    print("\n指标对比")
    print(f"{'':<14}" + "".join(f"{m:>12}" for m in modes))
    keys = [("Hit@1", lambda s: s["hit_at"][1]), ("Hit@3", lambda s: s["hit_at"][3]),
            ("Hit@5", lambda s: s["hit_at"][5]), ("Hit@8", lambda s: s["hit_at"][8]),
            ("MRR", lambda s: s["mrr"]), ("Evidence@8", lambda s: s["evidence_at_8"]),
            ("正例误拦", lambda s: s["positive_short_circuit"]),
            ("负例闸门", lambda s: f"{s['negative_gate_blocked']}/{len(negatives)}"),
            ("平均耗时ms", lambda s: s["avg_ms"])]
    for label, fn in keys:
        vals = []
        for m in modes:
            v = fn(results[m]["summary"])
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{label:<14}" + "".join(f"{v:>12}" for v in vals))

    tag = args.tag or datetime.now().strftime("%Y%m%d")
    out = PROJECT_ROOT / f"eval_results_retrieval_{tag}.json"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_set": str(args.questions),
        "seed": data.get("seed"),
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
