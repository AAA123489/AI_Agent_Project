"""召回自检图 —— 判断检索回来的内容是否真的能支撑回答。

## 解决什么问题

加这一层之前，检索链路没有任何"召回质量"概念：

- `VectorStore.distance_threshold=0.85` 存了但**从未被读过**（search_similar 不引用它）
- 唯一能拒答的是 `app_backend._refuse_out_of_kb_school`，而它只做**主体校校验**
  （query 里出现"河南工学院"就无条件放行），**不做主题存在性校验**
- 检索永远返回 top_k 个块，哪怕全是无关的

结果：问「河南工学院食堂几点开门」这种**本校但库里真没有**的题，系统会把最不相关的
块塞进 LLM 上下文让它编。实测该 query 的检索返回 6211 字符 / 15 个块。

## 为什么用 LangGraph 而不是一个 if

判完之后有三种去向，其中一种**要回到检索节点重来**：
充分 → 格式化放行；不充分 → 改写 query 回检索（环）；主题不符/重试用尽 → 拒答。
带环的编排是 if/else 表达不了的部分——这才是引入状态机的理由。

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

## 终止保证

LangGraph 1.2.10 的 `recursion_limit` 默认是 **10007**（不是旧版的 25），
靠它兜底等于没有兜底——环真跑飞了会空转到一万步才报错，中间每圈都在烧 LLM 调用。
所以真正的终止条件是 state 里的 `attempt` 计数器；`recursion_limit` 由
`max_attempts` 推导（`4*n+6`），**必须始终宽于计数器**——拍个死数会让它从"二次保险"
变成真正的终止条件（`max_attempts=5` 配死数 10 就先炸 GraphRecursionError 了，实测）。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ── 阈值（照上文表格标定；换 embedding 模型后必须重新标定）──
PASS_DIST = 0.30  # 最佳向量余弦距离 ≤ 此值 → 直接放行，不调判官

# 判官的合法判定（路由函数只认这三个，其余一律归一化成 sufficient）
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


class RecallState(TypedDict, total=False):
    """图状态。

    `attempt` 用覆盖语义（不加 reducer）——它就是终止计数器，累加语义在这里是错的。
    对比项目三的 `messages` 用 `operator.add` 累加：什么时候该用 reducer、
    什么时候该覆盖，取决于字段表达的是"累积历史"还是"当前值"。
    """

    query: str           # 用户原始问题（改写后不变）
    current_query: str   # 当前检索用的 query（判官改写后会变）
    top_k: int           # 检索片段数（MCP 侧会传非默认值，所以放进 state 而不是闭包里）
    candidates: list[dict]
    short_circuit: str | None  # 检索侧早退文案（库外拦截/检索失败/未找到）
    verdict: str
    attempt: int
    max_attempts: int
    context: str         # 最终返回给调用方的文本


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


def build_recall_graph(*, retrieve_fn, format_fn, llm_fn, api_key: str):
    """构造召回自检图（DI 工厂）。

    依赖全部注入而**不 import app_backend**：一是避免循环 import
    （app_backend 要 import 本模块），二是让图测试不必拉起 chromadb / torch。

    - retrieve_fn(query, top_k) -> 信封 dict {"candidates", "short_circuit", "timings"}
    - format_fn(candidates) -> str
    - llm_fn(messages, api_key, **kw) -> {"text", ...}，async

    这里**不收** max_attempts 参数：图的形状与它无关，它属于「怎么跑」而不是「图长什么样」，
    由 run_recall_guard 写进 state。之前放在工厂签名里，是个改了没反应的静默失效参数。
    """

    async def _retrieve(state: RecallState) -> dict:
        query = state.get("current_query") or state["query"]
        # 检索是同步阻塞的（Chroma 查询 + BM25 全库打分 + 补块），显式丢线程池，
        # 不依赖 langgraph 对同步节点的隐式 executor 行为
        env = await asyncio.to_thread(retrieve_fn, query, state.get("top_k") or 8)
        return {
            "candidates": env.get("candidates") or [],
            "short_circuit": env.get("short_circuit"),
            "attempt": state.get("attempt", 0) + 1,
        }

    async def _grade(state: RecallState) -> dict:
        # 检索侧已早退（库外拦截/失败/未找到）：文案是现成的，不必判，直接透传
        if state.get("short_circuit"):
            return {"verdict": "sufficient"}

        candidates = state.get("candidates") or []
        # 只看向量路的原始余弦距离——融合后的 distance 被 BM25 归一化分污染过，不能用
        vec_dists = [c["vector_distance"] for c in candidates if "vector_distance" in c]
        best = min(vec_dists) if vec_dists else None

        # 快通过：向量路都认为高度相关，直接放行，省掉一次 LLM 往返。
        # 基线 5 题的最佳距离全在 0.14~0.31，大部分问题走这条路。
        if best is not None and best <= PASS_DIST:
            logger.info("召回自检：向量距离 %.3f ≤ %.2f，快通过", best, PASS_DIST)
            return {"verdict": "sufficient"}

        # 模糊带 → LLM 判官
        query = state.get("current_query") or state["query"]
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
        out: dict = {"verdict": verdict}
        if verdict == "insufficient" and rewrite:
            out["current_query"] = rewrite
        return out

    async def _format(state: RecallState) -> dict:
        return {"context": format_fn(state.get("candidates") or [])}

    async def _refuse(state: RecallState) -> dict:
        # 两个来源共用一个出口：检索侧早退有现成文案（库外拦截/未找到），原样透传；
        # 否则是自检判出来的拒答
        return {"context": state.get("short_circuit") or REFUSE_TEXT}

    def _route(state: RecallState) -> str:
        """条件边。返回值**必须**是下方 path_map 里的 key——
        langgraph 的 _branch.py 是 self.ends[r]，返回未映射的 key 直接 KeyError，
        而 AgentLoop 那侧没有 try/except，异常会打断整条 SSE 流。
        所以这里的每个 return 都是字面量，且 verdict 已在 _parse_verdict 里做过白名单归一化。
        """
        if state.get("short_circuit"):
            return "refuse"
        verdict = state.get("verdict", "sufficient")
        if verdict == "sufficient":
            return "format"
        # 默认 1 = 不重检。直接 ainvoke 的调用方没写 max_attempts 时的保守取值：
        # 不明确要求重检，就不要自作主张多打一枪。
        if verdict == "insufficient" and state.get("attempt", 0) < state.get("max_attempts", 1):
            return "retrieve"  # ← 这就是那条环
        return "refuse"        # off_topic，或 insufficient 但重试用尽

    graph = StateGraph(RecallState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_node("format", _format)
    graph.add_node("refuse", _refuse)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _route,
        {"format": "format", "retrieve": "retrieve", "refuse": "refuse"},
    )
    graph.add_edge("format", END)
    graph.add_edge("refuse", END)
    return graph.compile()


async def run_recall_guard(graph, query: str, top_k: int = 8, max_attempts: int = 2) -> str:
    """跑一次自检图，返回最终文本（检索上下文 或 拒答文案）。"""
    final = await graph.ainvoke(
        {
            "query": query,
            "current_query": query,
            "top_k": top_k,
            "candidates": [],
            "verdict": "",
            "attempt": 0,
            "max_attempts": max_attempts,
        },
        # 显式传：真正的终止靠 state 里的 attempt，这里只是「计数器万一写错」的二次保险。
        # 不显式传的话默认是 10007，等于没有兜底。
        # 但也不能拍一个死数——上限必须**始终宽于** attempt 计数器，否则它就从
        # 二次保险变成了真正的终止条件（max_attempts=5 配死数 10 就会先炸
        # GraphRecursionError，而不是按计数正常转拒答）。每圈 retrieve→grade
        # 两个 superstep，4*n+6 留足余量。
        {"recursion_limit": 4 * max(1, max_attempts) + 6},
    )
    return final.get("context") or ""
