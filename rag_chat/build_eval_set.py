"""
build_eval_set.py — 构建可复现的 50 题评测题库
=============================================
为什么要单独建题库（2026-10-19）：
  25/50/100 题那三套随机评测的题库与脚本从未入库，已彻底丢失，论文里只剩
  「当时跑过一次」的数字；60/60 那套只有 6 题，且同一代码两次跑出 60/60 与
  50/60（两级 LLM 抖动）。没有固定题库，任何结论都无法复算。

本脚本产出 `eval_sets/qa_50.json`：
  - 40 道正例题：从知识库**确定性采样**（固定 seed）40 篇文档，每篇让 LLM 依据
    原文片段写一个问题 + 一段逐字答案。答案在写入前会**程序化校验**确实出自该片段
    （不是 LLM 说了算），校验失败就换片段重试/丢弃。
  - 10 道负例题：6 道本校主题硬负例（库里确实没有）+ 4 道无关题（库外）。
    口径与 eval_guard_probe.py 一致，便于两份评测互相印证。

题库一旦生成就提交进仓库，评测时不再调用 LLM —— 这是可复现的前提。
采样和生成过程都是确定性的（seed 写死），但**题目内容是 LLM 生成的**，
所以题库本身要人工过一遍再当论文依据。

用法:
    python build_eval_set.py --dry-run          # 只看采样结果，不调 LLM
    python build_eval_set.py                    # 生成 eval_sets/qa_50.json
    python build_eval_set.py --limit 5          # 先小样本试跑
    python build_eval_set.py --out eval_sets/my_set.json
"""

import os

# ── 离线加载：必须在 import 模型相关模块之前设置（照 eval_baseline.py）──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 固定 seed：换机器/换时间采样结果一致，题库才有「可复现」的意义
SEED = 20261019
DEFAULT_OUT = PROJECT_ROOT / "eval_sets" / "qa_50.json"

# 每类至少几篇，剩余名额按分类规模补给大分类。
# 小分类（材料学院 19 块）也要有代表，否则题库全是通知公告。
MIN_PER_CATEGORY = 2

# 负例题：与 eval_guard_probe.py 的硬负例/无关题同口径。
HARD_NEGATIVE = [
    "河南工学院食堂几点开门？",
    "河南工学院图书馆的开放时间是什么？",
    "河南工学院宿舍有空调吗？",
    "河南工学院宿舍几点熄灯？",
    "河南工学院校车几点发车？",
    "河南工学院的校历在哪里可以查到？",
]
OUT_OF_KB = [
    "今天天气怎么样？",
    "推荐几部好看的电影",
    "用 Python 写一个快速排序",
    "把「你好世界」翻译成英文",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 50 题可复现评测题库")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出 json 路径")
    p.add_argument("--limit", type=int, default=None, help="只生成前 N 道正例题（试跑）")
    p.add_argument("--dry-run", action="store_true", help="只打印采样，不调 LLM")
    return p.parse_args()


_ARGS = _parse_args()

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _norm(s: str) -> str:
    """比较答案片段时去掉空白/全半角差异，避免「逐字」被排版干扰。"""
    return re.sub(r"\s+", "", s or "").replace("，", ",").replace("。", ".")


def _load_sources() -> dict[str, dict]:
    """从 Chroma 读 (url -> 元数据)，按分类分组后确定性采样。

    直接用 chromadb 客户端，不经过 VectorStore：避免为了采样加载 420MB
    embedding 模型（这里只需要元数据）。
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
    col = client.get_collection("my_rag_collection")
    data = col.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for m in data["metadatas"]:
        if not m or not m.get("url"):
            continue
        key = m["url"]
        if key not in docs:
            docs[key] = {
                "url": key,
                "source": m.get("source", ""),
                "category": m.get("category", ""),
                "publish_date": m.get("publish_date", ""),
            }
    return docs


def _sample_sources(docs: dict[str, dict], total: int = 40) -> list[dict]:
    """按分类分层采样：每类先取 MIN_PER_CATEGORY 篇，剩余名额按类规模最大的补。

    采样在每类内部按 url 排序后用同一个 RNG 抽，结果与 Chroma 的返回顺序无关。
    """
    by_cat: dict[str, list[dict]] = {}
    for d in docs.values():
        by_cat.setdefault(d["category"], []).append(d)

    rng = random.Random(SEED)
    for cat in sorted(by_cat):
        by_cat[cat].sort(key=lambda x: x["url"])
        rng.shuffle(by_cat[cat])

    picked: list[dict] = []
    for cat in sorted(by_cat):
        picked.extend(by_cat[cat][:MIN_PER_CATEGORY])

    # 剩余名额：按「剩余未选文档数」从大到小补，保证大分类有足够样本
    remaining = total - len(picked)
    if remaining > 0:
        pool = [d for cat in sorted(by_cat) for d in by_cat[cat][MIN_PER_CATEGORY:]]
        pool.sort(key=lambda x: x["url"])
        rng.shuffle(pool)
        picked.extend(pool[:remaining])

    return picked[:total]


def _best_chunk(col, url: str) -> str:
    """取该 URL 下最适合出题的一块：文字够长、含数字/日期（可验证的事实多）。"""
    data = col.get(where={"url": url}, include=["documents"])
    chunks = [c for c in (data.get("documents") or []) if c and len(c) >= 120]
    if not chunks:
        return ""
    # 含日期/数字的块优先；同分取更长的（信息量更大）
    return max(chunks, key=lambda c: (len(re.findall(r"\d", c)) > 0, len(c)))


_GEN_SYSTEM = (
    "你是 RAG 评测题库编写者。只输出一个 JSON 对象，不要解释、不要 markdown 代码块。"
)

_GEN_PROMPT = """下面是一篇河南工学院官网文档的正文片段：

---
{chunk}
---

请基于这段文字写一个**用户真的可能问出口**的中文问题，并给出片段中能逐字回答它的一段原文。

要求：
1. 问题不要照抄标题，也不要出现「根据上述片段」这类元话语；像学生/教职工在问答系统里提问。
2. answer_span 必须**逐字**摘自上面的片段（10~80 字，可含日期、数字），不要改写。
3. 问题必须能由 answer_span 直接回答，不能问片段里没有的信息。
4. 只输出 JSON：{{"question": "...", "answer_span": "..."}}"""


async def _gen_one(chunk: str, api_key: str) -> dict | None:
    """调 LLM 生成一题；JSON 解析失败返回 None（由调用方重试/跳过）。"""
    from app_backend import _call_llm  # 延迟 import：读 .env 内的 key

    try:
        resp = await _call_llm(
            [{"role": "user", "content": _GEN_PROMPT.format(chunk=chunk)}],
            api_key=api_key,
            system=_GEN_SYSTEM,
            tools=[],
            max_tokens=400,
            temperature=0.2,
        )
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return None

    text = (resp.get("text") or "").strip()
    # 容忍模型偶尔包一层 ```json
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    q = (obj.get("question") or "").strip()
    span = (obj.get("answer_span") or "").strip()
    if not q or not span:
        return None
    return {"question": q, "answer_span": span}


async def main() -> None:
    docs = _load_sources()
    sampled = _sample_sources(docs)
    print(f"知识库 {len(docs)} 篇，分层采样 {len(sampled)} 篇")
    for d in sampled:
        print(f"  [{d['category']}] {d['publish_date']} {d['source'][:45]}")

    if _ARGS.dry_run:
        return

    from app_backend import _call_llm, API_KEY
    import chromadb

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
    col = client.get_collection("my_rag_collection")

    positives = []
    for i, d in enumerate(sampled, 1):
        if _ARGS.limit and len(positives) >= _ARGS.limit:
            break
        chunk = _best_chunk(col, d["url"])
        if not chunk:
            print(f"[{i}] 跳过（无可用正文块）: {d['source'][:40]}")
            continue

        got = None
        for attempt in range(3):
            got = await _gen_one(chunk, API_KEY)
            if got and _norm(got["answer_span"]) in _norm(chunk):
                break
            got = None
        if not got:
            print(f"[{i}] 跳过（3 次生成均无法逐字校验）: {d['source'][:40]}")
            continue

        positives.append({
            "id": f"P{len(positives)+1:02d}",
            "type": "positive",
            "question": got["question"],
            "answer_span": got["answer_span"],
            "expected_url": d["url"],
            "source": d["source"],
            "category": d["category"],
            "publish_date": d["publish_date"],
        })
        print(f"[{i}] ✓ {got['question'][:50]}")

    negatives = [
        {"id": f"N{i:02d}", "type": "negative", "kind": "hard",
         "question": q, "expected": "refuse"}
        for i, q in enumerate(HARD_NEGATIVE, 1)
    ] + [
        {"id": f"N{len(HARD_NEGATIVE)+i:02d}", "type": "negative", "kind": "out_of_kb",
         "question": q, "expected": "refuse"}
        for i, q in enumerate(OUT_OF_KB, 1)
    ]

    out = {
        "version": 1,
        "seed": SEED,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "builder": "build_eval_set.py",
        "kb_chunks_at_build": col.count(),
        "questions": positives + negatives,
    }
    out_path = Path(_ARGS.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_path}：正例 {len(positives)} + 负例 {len(negatives)} = {len(out['questions'])} 题")


if __name__ == "__main__":
    asyncio.run(main())
