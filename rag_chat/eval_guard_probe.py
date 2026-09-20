"""
eval_guard_probe.py — 召回自检能力探测（硬负例 + 正例，真实检索链路）
=====================================================================
6 题基准（eval_baseline.py）只覆盖「库里有答案」的题，测不出自检层的价值：
那 6 题改造前就是 60/60，自检开着也只是别把分掉下去。

本脚本补的是**另一半口径**——把三类题混在一起量：

  1. 硬负例：本校主题但库里真没有答案（食堂几点开门 / 招生办电话）
     → 期望：自检拒答。guard=off 时会走满 20 块把无关内容喂给 LLM 编。
  2. 无关题：跟本校没关系的一般问题（今天天气 / 推荐电影）
     → 期望：拒答（任一机制）。**注意这里不能期望「库外拦截」**——既有的
     `_refuse_out_of_kb_school` 只做**校名**校验，题目里没有校名它根本不管，
     guard=off 时这两题是**放行**的（实测）。它被拦下来只可能是自检的功劳。
  3. 灰区题：主题词在库、具体事实不在库（招生办电话 / 现任校长）
     → **不设期望，只观察**。正确行为本身有争议（见 GRAY 注释），不参与达标统计。
  4. 正例：库里有答案（6 题基准里除「河北工学院」外的 5 题）
     → 期望：放行，且**耗时几乎不涨**（距离够近走快通过通道，不调判官）。
     这一项是防误拒的对照组，比拒答率更重要。

耗时口径：每组首个查询会背上 embedding 模型加载（实测约 12s），所以先跑一次预热，
下面的 per-question 耗时才是自检本身的成本。

只打检索层（`_search_knowledge_base`）——自检图就在这一层，结果是确定性的、可复现的。
加 `--answers` 再跑一遍完整 AgentLoop，看最终回答是不是真的拒答而非编造（花 LLM 调用）。

用法:
    python eval_guard_probe.py                          # guard 按 .env（默认 off）
    RECALL_GUARD=on python eval_guard_probe.py --tag guard_on
    RECALL_GUARD=on RECALL_GUARD_MAX_ATTEMPTS=1 python eval_guard_probe.py --tag ring_off  # 消融：环关
    RECALL_GUARD=on python eval_guard_probe.py --answers --tag guard_on_full

注意：RECALL_GUARD / TOP_K / RETRIEVAL_MODE / RETRIEVAL_RERANK 是 app_backend 的模块级
常量，import 时一次性求值，必须在 import 之前写进 os.environ（照 eval_baseline.py 的做法）。

硬负例的「库里确实没有」是人工扫全库（6297 块）确认过的（见 docs/改进记录.md）：
「营业时间 / 开饭 / 就餐时间 / 宿舍有空调 / 校车 / 班车 / 通勤 / 熄灯 / 开馆 / 闭馆」全库 0 命中；
「校历」的 3 处命中是**子串误命中**（"建校历史"里含"校历"），真词 0 命中；
含「食堂」的块共 11 个，全部是安全排查/征兵宣传通知里的**地点罗列**，没有一条讲营业时间。

⚠️ 一条被撤下来的负例：「招生办咨询电话」原本在硬负例里，后来发现库里有官方座机
0373-3691179——但那是**校团委**的，不是招生办的。照抄给用户是答错，拒答才对，
所以它仍留在负例侧，只是移进了灰区组、不设硬期望。另一条「现任校长」同理撤下：
库里确有「校长+人名」形态，断言它「库里没有」不严谨。
"""

import os

# ── 离线加载：必须在 import 模型相关模块之前设置（照 eval_baseline.py）──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="召回自检能力探测（硬负例 + 正例）")
    p.add_argument("--rerank", choices=["on", "off"], default="off",
                   help="重排开关，默认 off（对齐 60/60 那次配置；.env 里现在是 on）")
    p.add_argument("--tag", default=None, help="实验标签，输出文件名后缀")
    p.add_argument("--answers", action="store_true",
                   help="额外跑完整 AgentLoop，记录最终回答（花 LLM 调用）")
    return p.parse_args()


_ARGS = _parse_args()
os.environ["RETRIEVAL_RERANK"] = _ARGS.rerank
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if __name__ == "__main__":
    # Windows 控制台 GBK 兼容（只在作为脚本运行时改 stdout；放模块级会劫持
    # import 本模块者的 stdout，pytest 拆捕获流时崩）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

from app_backend import (  # noqa: E402
    AgentLoop,
    ErrorEvent,
    RECALL_GUARD,
    RECALL_GUARD_MAX_ATTEMPTS,
    TextEvent,
    _search_knowledge_base,
)


# ── 题库（分三类，口径互不混用）────────────────────────────────

HARD_NEGATIVE = [
    ("食堂几点开门", "河南工学院食堂几点开门？"),
    ("图书馆开放时间", "河南工学院图书馆的开放时间是什么？"),
    ("宿舍有无空调", "河南工学院宿舍有空调吗？"),
    ("宿舍熄灯时间", "河南工学院宿舍几点熄灯？"),
    ("校车发车时间", "河南工学院校车几点发车？"),
    ("校历", "河南工学院的校历在哪里可以查到？"),
]

# 灰区题：主题词在库里、但**具体事实**不在。**不设期望，只观察**。
# 这两题都是「判官按相关性判、而不是按可答性判」的地方，正确行为本身有争议：
#   - 招生办电话：库里有官方座机 0373-3691179，但那是**校团委**的，不是招生办的。
#     照抄给用户就是答错，所以拒答更好——但判官看到"招生"+号码可能认为可答。
#   - 现任校长：库里确有「校长+人名」形态（多为外单位领导，如乡镇中心学校校长刘东方），
#     能否据此认定本校校长存疑。断言它「库里没有」是不严谨的，所以不放进硬负例。
GRAY = [
    ("招生办电话", "河南工学院招生办的咨询电话是多少？"),
    ("现任校长", "河南工学院现任校长是谁？"),
]

OUT_OF_KB = [
    ("天气", "今天天气怎么样？"),
    ("电影推荐", "推荐几部好看的电影"),
    ("写代码", "用 Python 写一个快速排序"),
    ("翻译", "把「你好世界」翻译成英文"),
]

POSITIVE = [
    ("端午放假", "2026年端午节放假是怎么安排的？"),
    ("运动会", "河南工学院第九届运动会是什么时候举办？"),
    ("报名截止", "中国国际大学生创新大赛（2026）河南工学院校赛的报名截止时间是什么时候？"),
    ("暑期实践", "学校最近有安排暑期社会实践吗？具体怎么做？"),
    ("心理健康", "2026年春季学期学校在学生心理健康方面有哪些安排？"),
]

# 拒答类结果的前缀 → 归类。三种来源必须分开数：
#   自检拒答是本层新增能力；库外拦截是改造前就有的闸门；检索失败/未找到是既有早退。
_GUARD_PREFIX = "【召回自检"
_OOB_PREFIX = "【库外题拦截"


def _classify(text: str) -> str:
    if text.startswith(_GUARD_PREFIX):
        return "自检拒答"
    if text.startswith(_OOB_PREFIX):
        return "库外拦截"
    if text.startswith("检索失败") or text.startswith("知识库中未找到"):
        return "检索早退"
    return "放行"


def _probe(kind: str, label: str, question: str) -> dict:
    t0 = time.perf_counter()
    text = _search_knowledge_base(question)
    elapsed = time.perf_counter() - t0
    return {
        "kind": kind,
        "label": label,
        "question": question,
        "result": _classify(text),
        "elapsed_s": round(elapsed, 2),
        "chars": len(text),
        "head": text[:70].replace("\n", " "),
    }


async def _answer(question: str) -> str:
    """跑完整 AgentLoop，拿最终回答（--answers 才用）。"""
    loop = AgentLoop()
    out = ""
    async for ev in loop.run_stream(question):
        if isinstance(ev, TextEvent):
            out += ev.content
        elif isinstance(ev, ErrorEvent):
            out = f"[错误] {ev.message}"
    return out


async def main() -> None:
    print("=" * 78)
    print(f"召回自检探测 | RECALL_GUARD={RECALL_GUARD} "
          f"MAX_ATTEMPTS={RECALL_GUARD_MAX_ATTEMPTS if RECALL_GUARD == 'on' else '-'} "
          f"| rerank={_ARGS.rerank}")
    print("=" * 78)

    # 预热：首个检索要加载 embedding 模型 + 建 BM25 索引，实测约 18s。
    # 不预热的话每组的第一题都会背上这笔一次性开销，耗时均值没法看。
    t0 = time.perf_counter()
    _search_knowledge_base("河南工学院")
    print(f"\n（预热完成 {time.perf_counter() - t0:.1f}s：embedding 模型 + BM25 索引 + 自检图）")

    rows = []
    for kind, group in (("硬负例", HARD_NEGATIVE), ("灰区", GRAY),
                        ("无关题", OUT_OF_KB), ("正例", POSITIVE)):
        print(f"\n──── {kind} ────")
        for label, q in group:
            r = _probe(kind, label, q)
            rows.append(r)
            print(f"  {label:14s} {r['result']:6s} {r['elapsed_s']:5.2f}s  {r['chars']:6d}字  {r['head']}")

    # ── 汇总：拒答率按类分开算，正例单独算误拒 ──
    # 「无关题」的可接受结果有两种：既有闸门拦下（题里有别的校名时）或自检拒答。
    # 两者都算达标，但要分别计数——是自检在起作用还是老闸门在起作用，是两回事。
    print("\n" + "=" * 78)
    for kind, want in (("硬负例", ("自检拒答",)),
                       ("无关题", ("自检拒答", "库外拦截")),
                       ("正例", ("放行",))):
        sub = [r for r in rows if r["kind"] == kind]
        ok = sum(1 for r in sub if r["result"] in want)
        avg = sum(r["elapsed_s"] for r in sub) / max(1, len(sub))
        flag = "✅" if ok == len(sub) else "⚠️"
        print(f"{flag} {kind}: {ok}/{len(sub)} 达到期望「{'/'.join(want)}」  平均 {avg:.2f}s")
        for r in sub:
            if r["result"] not in want:
                print(f"     ↳ 未达期望: {r['label']} → {r['result']}")
        if kind == "无关题":
            print(f"     其中：自检拒答 {sum(1 for r in sub if r['result'] == '自检拒答')} 条 / "
                  f"库外闸门 {sum(1 for r in sub if r['result'] == '库外拦截')} 条")

    # 灰区题不设期望：拒答和放行都有说法，只看它落在哪边，不参与达标统计。
    sub = [r for r in rows if r["kind"] == "灰区"]
    if sub:
        avg = sum(r["elapsed_s"] for r in sub) / len(sub)
        tally = " / ".join(f"{r['label']}→{r['result']}" for r in sub)
        print(f"➖ 灰区（不设期望，仅观察）  平均 {avg:.2f}s  {tally}")

    if _ARGS.answers:
        print("\n──── 硬负例的最终回答（完整链路，看是否编造）────")
        for label, q in HARD_NEGATIVE[:3]:
            ans = await _answer(q)
            rows.append({"kind": "硬负例-回答", "label": label, "question": q, "answer": ans})
            print(f"\n  【{label}】{q}\n  → {ans[:300]}")

    tag = _ARGS.tag or ("guard_on" if RECALL_GUARD == "on" else "guard_off")
    out_path = f"eval_results_guard_probe_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "recall_guard": RECALL_GUARD,
            "max_attempts": RECALL_GUARD_MAX_ATTEMPTS,
            "rerank": _ARGS.rerank,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
