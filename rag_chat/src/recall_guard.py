"""召回自检 —— 判断检索回来的内容是否真的能支撑回答。

## 解决什么问题

加这一层之前，检索链路没有任何"召回质量"概念：

- `VectorStore.distance_threshold=0.85` 存了但**从未被读过**（已于 2026-09-17 删除）
- 唯一能拒答的是 `app_backend._refuse_out_of_kb_school`，而它只做**主体校校验**
  （query 里出现"河南工学院"就无条件放行），**不做主题存在性校验**
- 检索永远返回 top_k 个块，哪怕全是无关的

结果：问「河南工学院食堂几点开门」这种**本校但库里真没有**的题，系统会把最不相关的
块塞进 LLM 上下文让它编。实测该 query 的检索返回 6211 字符 / 15 个块。

## 三种去向，其中一种要回到检索

充分 → 格式化放行；不充分 → 改写 query 回检索；主题不符/重试用尽 → 拒答。
第二与第三种的区别就是"要不要重来一次"，所以这是一个**带环的循环**。

## 为什么是手写循环

2026-09 之前这里是一张 LangGraph `StateGraph`（`retrieve → grade →条件边`），
改成手写**只为去掉框架依赖，不改行为**——判定顺序、每个默认值、拒答文案都逐项
保留，30 条单元测试里 29 条一行未动。

框架那版的环有个隐患：唯一的外层保险 `recursion_limit` 默认值是 **10007**
（不是旧版的 25），等于没有兜底，环跑飞会空转到一万步、每圈都在烧 LLM 调用。
手写循环不需要这个二次保险——`attempt` 每圈 +1，`attempt < max_attempts` 不成立
即退出，**循环上界由计数器直接决定**。条件边那侧的 `path_map` 白名单同理不再需要：
原来路由函数返回未映射的 key 会直接 KeyError（而 SSE 侧没有 try/except，会打断
整条流），现在 `return` 就是 `return`。

## 两条必须知道的事实（都实测过，别凭印象改）

1. **没有一个跨检索来源统一的相似度分数可以卡阈值。**
   `rrf_fuse` 融合后 `distance` 可能是 BM25 路的归一化分（`1.0 - score/best`，
   第一名恒为 0.0 → 展示"相似度 100%"），也可能是同文档补块硬编码继承来的。
   **唯一可信的是向量路的原始余弦距离**，由 `rrf_fuse` 单独保留在 `vector_distance`
   字段里（只有向量路命中的块才有这个字段）。

2. **阈值必须从真实库标定，不能拍。** 线上库（my_rag_collection）实测的向量余弦距离：

   | query | 最佳距离 |
   |---|---|
   | Q5 心理健康 / Q3 报名截止 / Q4 暑期实践 / Q1 端午 / Q2 运动会 | 0.143 / 0.203 / 0.249 / 0.255 / 0.307 |
   | 转专业（BM25 独立价值案例） | 0.384 |
   | 河南工学院食堂几点开门 | 0.465 |
   | 今天天气怎么样 / 推荐几部电影 | 0.603 / 0.613 |

   整个动态范围是 0.14~0.61 —— 所以那个 `0.85` 阈值**永远不可能触发**。
   `PASS_DIST = 0.30` 就是照这张表定的：基线 5 题里 4 题 ≤0.30 免检，只有 Q2 运动会
   （0.307）差 0.007 落到判官——**别写成"5 题全免检"**，它确实会多打一次 LLM。
   **换 embedding 模型后必须重新标定。**

## 环默认不开（消融结论）

环（`RECALL_GUARD_MAX_ATTEMPTS=2`）实测**没有收益、且有反效果**，所以默认值是 1。
详见 app_backend 里该常量的注释与 docs/改进记录.md 第 15 条。代码保留可开关。
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# ── 阈值（照上文表格标定；换 embedding 模型后必须重新标定）──
PASS_DIST = 0.30  # 最佳向量余弦距离 ≤ 此值 → 直接放行，不调判官

# 判官的合法判定（只认这三个，其余一律归一化成 sufficient）
_VERDICTS = ("sufficient", "insufficient", "off_topic")

# 自检拒答文案。前缀必须是「【召回自检」——AgentLoop 靠它区分本层拒答与
# 库外闸门的「【库外题拦截】」，用宽泛的「【」会把库外拦截误标成自检未通过。
REFUSE_TEXT = (
    "【召回自检未通过】知识库中没有与这个问题相关的内容，无法作答。"
    "严禁用你自己的知识或推测补充任何信息（包括数字、日期、人名），"
    "只说明无法回答，不要展开。"
)

JUDGE_PROMPT = """你是 RAG 系统的召回质量判定器。用户提了一个问题，检索系统返回了若干片段。
你的唯一任务是：判断这些片段与问题的关系，据此给出一行判定。

【判定标准】
- sufficient：片段包含与问题相关的信息。**大多数情况应判这一项**，哪怕片段不完整、不够精确。
- insufficient：片段看着相关，但**缺少回答问题所需的关键事实**（提到了主题却没有具体答案）。
- off_topic：片段与问题**明显无关**（问的是这件事，返回的全是另一件事）。

【三条硬约束】
1. 宁可判 sufficient，也不要因为片段"不完整"就往下判。只有你确信片段撑不起回答时，才判后两项。
2. 不要用你自己的知识去判断正确答案是什么，只看片段与问题的**相关性**。
3. 无法确定时一律判 sufficient。

【输出格式】严格输出一行，不要有任何其他内容：
VERDICT: sufficient|insufficient|off_topic | REWRITE: <判 insufficient 时给一个更可能检索到答案的查询，否则填 -> | EVIDENCE: <从片段里逐字抄一句你的判断依据，20 字以内>"""


def _parse_verdict(raw: str) -> tuple[str, str]:
    """宽松解析判官输出 → (verdict, rewritten_query)。

    **双默认兜底**：解析不出、判定值不在白名单、或文本为空 → 一律 sufficient 放行。
    没有这条兜底，任何解析 bug / API 抖动都会直接把正常问题判成拒答掉分。
    """
    if not raw:
        logger.warning("召回自检判官返回空文本，按放行处理")
        return "sufficient", ""
    m = re.search(r"VERDICT\s*[:：]\s*([A-Za-z_]+)", raw)
    if not m:
        # 也覆盖 _call_llm 的失败返回值（"API 调用失败 (401)" 等，其中没有 VERDICT）
        logger.warning("召回自检判官输出无法解析，按放行处理: %r", raw[:120])
        return "sufficient", ""
    verdict = m.group(1).strip().lower()
    if verdict not in _VERDICTS:
        logger.warning("召回自检判官返回未知判定 %r，按放行处理", verdict)
        return "sufficient", ""
    rewrite = ""
    # 用 [^|\n]+ 而不是 (.+)：判官输出是「VERDICT: .. | REWRITE: .. | EVIDENCE: ..」，
    # (.+) 会把 " | EVIDENCE: xxx" 一起吞进改写 query，污染下一圈检索。
    m2 = re.search(r"REWRITE\s*[:：]\s*([^|\n]+)", raw)
    if m2:
        rewrite = m2.group(1).splitlines()[0].strip()
        if rewrite in ("-", "无", "none", "None", "N/A", "n/a"):
            rewrite = ""
    return verdict, rewrite


def _judge_input(query: str, candidates: list[dict], limit: int = 5, chars: int = 200) -> str:
    """判官输入：只给 top5 × 200 字。

    不给全部候选（补块后常 15~20 块 × 800 字 ≈ 16k 字符）——判官只需要判断相关性，
    全量喂进去每问都要多付一大笔 token，而且卡在 SSE 的关键路径上。
    """
    parts = [f"【用户问题】{query}", "", "【检索返回的片段】"]
    for i, doc in enumerate(candidates[:limit], 1):
        src = (doc.get("metadata") or {}).get("source", "未知文档")
        parts.append(f"[{i}] 来源: {src}\n{(doc.get('text') or '')[:chars]}")
    return "\n\n".join(parts)


def _best_vec_dist(candidates: list[dict]) -> float | None:
    """候选里最小的**向量路原始余弦距离**；一个都没有则 None。

    只看 `vector_distance`：融合后的 `distance` 被 BM25 归一化分污染过（见模块开头）。
    一个块都没有、或全部来自 BM25 路时返回 None —— 此时必走判官，不做距离快通过。
    """
    vec_dists = [c["vector_distance"] for c in candidates if "vector_distance" in c]
    return min(vec_dists) if vec_dists else None


async def _judge(llm_fn, api_key: str, query: str, candidates: list[dict], best: float | None):
    """调 LLM 判官 → (verdict, rewrite)。判官出错按放行处理，绝不中断问答。"""
    try:
        resp = await llm_fn(
            [{"role": "user", "content": _judge_input(query, candidates)}],
            api_key,
            system=JUDGE_PROMPT,
            tools=[],          # 判官不需要工具，走 [] 让 _call_llm 不下发 tools 字段
            max_tokens=150,
            temperature=0,
        )
        raw = resp.get("text", "")
    except Exception:
        # 判官本身出错绝不能中断问答——按放行处理（宁可少拒答，不可误拒答）
        logger.exception("召回自检判官调用异常，按放行处理")
        raw = ""
    verdict, rewrite = _parse_verdict(raw)
    logger.info("召回自检：best=%s 判官=%s rewrite=%r", best, verdict, rewrite[:40])
    return verdict, rewrite


async def run_recall_guard(
    *,
    retrieve_fn,
    format_fn,
    llm_fn,
    api_key: str,
    query: str,
    top_k: int = 8,
    max_attempts: int = 1,
) -> str:
    """跑一次召回自检，返回最终文本（检索上下文 / 检索侧早退文案 / 自检拒答文案）。

    依赖全部注入而**不 import app_backend**：一是避免循环 import（app_backend 要
    import 本模块），二是让测试不必拉起 chromadb / torch。

    - `retrieve_fn(query, top_k)` → 信封 dict `{"candidates", "short_circuit", "timings"}`，
      同步阻塞（Chroma 查询 + BM25 全库打分 + 补块），故显式丢线程池。
    - `format_fn(candidates)` → str
    - `llm_fn(messages, api_key, **kw)` → `{"text", ...}`，async

    `max_attempts` 默认 1 = **不重检**。它既是生产的默认
    （`RECALL_GUARD_MAX_ATTEMPTS`），也是消融结论（环无收益且有反效果）。
    形参默认值与生产默认值取同一个数，是因为改成手写后只剩这一个入口——
    原来图里 `_route` 另有一个"state 里没写 max_attempts 就保守取 1"的默认，
    那是给直接 `ainvoke` 的调用方兜底的，现在没有那种调用方了。
    """
    current_query = query
    attempt = 0

    while True:
        # ── 检索 ──
        env = await asyncio.to_thread(retrieve_fn, current_query, top_k)
        candidates = env.get("candidates") or []
        short_circuit = env.get("short_circuit")
        attempt += 1

        # ── 判定 ──
        if short_circuit:
            # 检索侧已早退（库外拦截/失败/未找到）：文案是现成的，不必判
            verdict = "sufficient"
        else:
            best = _best_vec_dist(candidates)
            # 快通过：向量路都认为高度相关，直接放行，省掉一次 LLM 往返。
            # 基线 5 题的最佳距离全在 0.14~0.31，大部分问题走这条路。
            if best is not None and best <= PASS_DIST:
                logger.info("召回自检：向量距离 %.3f ≤ %.2f，快通过", best, PASS_DIST)
                verdict = "sufficient"
            else:
                # 模糊带 → LLM 判官
                verdict, rewrite = await _judge(llm_fn, api_key, current_query, candidates, best)
                if verdict == "insufficient" and rewrite:
                    current_query = rewrite

        # ── 去向（原条件边的三个分支）──
        if short_circuit:
            # 检索侧早退：文案原样透传（库外拦截的「【库外题拦截】」前缀靠它区分类别）
            return short_circuit
        if verdict == "sufficient":
            return format_fn(candidates)
        if verdict == "insufficient" and attempt < max_attempts:
            continue           # ← 这就是那条环（默认进不来，max_attempts=1）
        return REFUSE_TEXT     # off_topic，或 insufficient 但重试用尽
