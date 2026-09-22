"""Agent 主循环的 LangGraph 状态机 —— 把 `AgentLoop` 的 for-step 手写循环改成显式图。

## 为什么

原 `AgentLoop.run_stream`（app_backend.py）用 `for rnd in range(MAX_ROUNDS)`
手写了一个 ReAct 循环：调 LLM → 有 tool_use 就执行工具 → 回到 LLM → 无 tool_use 就收工。
这段控制流本身就是一张图，只是用 `for` 写出来的：

    START → prepare → llm ─┬─(有 tool_uses)→ tools ─┬─(round < max_rounds)→ llm
                           │                       └─(轮数用尽)──────────→ exhausted → END
                           └─(无 tool_uses)────────────────────────────→ finalize → END

两个判定问的是不同问题，所以分在两条边上：**「LLM 要工具吗」在 llm 之后问**，
**「轮数用尽吗」在 tools 之后问**。这样最后一轮仍会执行工具再兜底，
与原来 `for rnd in range(max_rounds)` 的语义逐轮对齐（原实现最后一轮同样跑工具）。

写成状态机与 `src/recall_guard.py` 同源的理由：**分支与环显式化**。工具的二次校验、
拒答分类、轮数终止都从 `for` 循环体里的 if 变成图上的节点与边，再往里加环节
（更多工具路由、多轮改写、并行工具）不必继续往循环体里塞分支。

## 与原实现的逐项对齐（改的是编排，不是行为）

- 轮数语义：`round` 计数器，从 0 起，`llm` 节点每跑一次 +1；
  `round == max_rounds` 时不再进 `tools` 而走 `exhausted` —— 等价 `for rnd in range(6)`。
- 事件顺序：`第 N 轮推理` → 流式文本 → ToolCall → ToolResult → 思考步骤，
  与 `run_stream` 逐行一致。
- `thinking_steps` 累积；`sources` **覆盖**（不是累积）—— 保持原语义。
- 收尾两步（`✅ 生成完成` / `⚠️ 达到最大轮数`）只进 `thinking_steps`、
  **不单独 yield**，随 `DoneEvent` 一次性下发 —— 与原来一致。

## 不做的事（刻意）

- **不用 `add_messages` reducer。** `messages` 里装的是 Anthropic 原生块格式
  （`{"type": "tool_use"}` / `{"type": "tool_result"}`），`add_messages` 会把 dict
  转成 LangChain Message 对象并改写 content 结构，破坏发出去的请求体。
  这里用覆盖语义、节点自己拼列表（同 `recall_guard` 里 "计数器用覆盖、历史用累加" 的取舍）。
- **不 import app_backend。** 依赖全部注入：一是 app_backend 要 import 本模块（循环 import），
  二是让图测试不必拉起 chromadb / torch（同 `recall_guard.py` 的工厂约定）。

## 事件形状

节点经 `get_stream_writer()` 推出的 payload 是 **dict**，不是 app_backend 的事件 dataclass
—— 后者是表现层契约，归 app_backend 所有；这里是纯编排层，只产出"发生了什么"。
调用方用 `astream(..., stream_mode=["custom", "values"])` 消费。

`get_stream_writer()` 在非流式上下文（`ainvoke`）下是安全 no-op，所以节点不依赖
"一定在流式运行中"这个前提 —— 直接 `ainvoke` 也不会炸。
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.stream_gate import OpeningGate

logger = logging.getLogger(__name__)

# 拒答原因 → 前端展示的思考步骤文案。
# 与 app_backend 的字符串前缀解耦：前缀判定由注入的 classify_refusal_fn 做，
# 这里只负责"原因 → 显示什么"，全是字面量。
_REFUSAL_STEP = {
    "recall_guard": "🛡️ 召回自检未通过，已拒答",
    "out_of_kb": "🚫 库外题拦截，已拒答",
}


class AgentState(TypedDict, total=False):
    """图的状态。`total=False` → 调用方只传少数字段即可启动，节点内一律 `state.get`。

    字段语义（覆盖 vs 累加）是按"它是当前值还是历史"选的，与 `RecallState` 同一取舍：
    - `messages` / `thinking_steps` —— 历史，只增不减
    - `round` / `sources` / `answer` / `refusal` / `pending_tool_uses` —— 当前值
    """

    # ── 输入 ──
    question: str          # 用户原话（库外二次校验用，全程不变）
    history: list[dict]    # 多轮历史（仅 prepare 读一次）

    # ── 会话状态 ──
    messages: list[dict]   # Anthropic 原生消息列表（含 tool_use / tool_result 块）
    thinking_steps: Annotated[list[str], operator.add]  # 累积，最终随 done 事件下发

    # ── 循环控制 ──
    round: int             # 终止计数器：llm 节点每跑一次 +1
    max_rounds: int
    pending_tool_uses: list[dict]  # 本轮 LLM 请求的工具调用（路由依据）

    # ── 产出 ──
    answer: str
    sources: list[dict]
    refusal: str | None    # "recall_guard" | "out_of_kb" | None
    stream_error: bool     # 流式失败（已回退非流式重发）
    streamed_chunks: int   # 已流式下发的文本块数（0 = 整段补发）

    # ── 计时 ──
    started_at: float


def _fmt_step(idx: int, text: str) -> str:
    """与 app_backend 原格式逐字一致：`[HH:MM:SS] Step N: 文本`。"""
    return f"[{datetime.now().strftime('%H:%M:%S')}] Step {idx}: {text}"


def build_agent_graph(
    *,
    llm_fn,
    tool_fn,
    guard_fn,
    parse_sources_fn,
    classify_refusal_fn,
    max_rounds: int = 6,
    max_history_turns: int = 10,
    max_history_chars: int = 2000,
    search_tool_name: str = "search_knowledge_base",
):
    """构建并编译 Agent 主循环图。

    依赖全部注入，本模块不 import app_backend（避免循环 import + 让测试免于拉起重依赖）：

    - `llm_fn(messages, on_delta)` → 异步，返回 `{"text", "tool_uses", "stop_reason"}`；
      传 `on_delta` 时启用流式，每个文本增量回调一次。
    - `tool_fn(name, input)` → 同步阻塞，返回结果文本；节点内丢线程池执行。
    - `guard_fn(name, user_question)` → 同步，返回拒答文案或 None（库外主体二次校验）。
    - `parse_sources_fn(result_text)` → 同步，从检索结果里解析结构化来源。
    - `classify_refusal_fn(result_text)` → 同步，返回 `"recall_guard"` / `"out_of_kb"` / None。
      拒答的字符串前缀属于 app_backend 的契约，判定留在那边，这里只消费结论。

    `max_rounds` 是**形状参数**（决定图上环的终止上限），进工厂签名；
    `round` 是运行期计数器，进 state —— 与 `recall_guard` 里
    "形状参数进工厂、运行参数进 state" 的区分一致。
    """

    def _emit(payload: dict) -> None:
        """把事件推给 `astream(stream_mode="custom")`。节点内调用。"""
        get_stream_writer()(payload)

    def _normalize_content(content) -> str:
        """历史消息 content 可能是 list[dict] 或 str，统一转成 str。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content) if content else ""

    # ── 节点 ──────────────────────────────────────────────

    async def _prepare(state: AgentState) -> dict:
        """构建消息列表（截断历史防上下文超长），发出首个思考步骤。"""
        question = state.get("question", "")
        messages: list[dict] = []
        for msg in (state.get("history") or [])[-max_history_turns:]:
            role = msg.get("role", "user")
            content = _normalize_content(msg.get("content", ""))
            # 跳过空的与前端占位消息（⏳ = 正在生成）
            if not content or content.startswith("⏳"):
                continue
            messages.append({"role": role, "content": content[:max_history_chars]})
        messages.append({"role": "user", "content": question})

        step = _fmt_step(1, f"接收 Query: [{question[:60]}{'...' if len(question) > 60 else ''}]")
        _emit({"type": "thinking", "step": step})
        return {
            "messages": messages,
            "thinking_steps": [step],
            "round": 0,
            "max_rounds": max_rounds,
            "started_at": time.time(),
            "sources": [],
            "refusal": None,
        }

    async def _llm(state: AgentState) -> dict:
        """调 LLM，边生成边把文本增量推给前端；返回本轮的工具调用请求。"""
        rnd = state.get("round", 0)
        base = len(state.get("thinking_steps", []))
        step = _fmt_step(base + 1, f"第 {rnd + 1} 轮推理 — 调用 LLM...")
        _emit({"type": "thinking", "step": step})

        messages = state.get("messages", [])
        thinking = [step]

        # 流式调用：on_delta 把增量塞进队列，本协程并发排空队列实时转发。
        # 流式解析失败时回退非流式重发（stream_error 置位），答案交给补发路径整段输出。
        chunk_queue: asyncio.Queue = asyncio.Queue()
        stream_error = [False]
        streamed_chunks = 0

        async def _llm_with_stream() -> dict:
            try:
                return await llm_fn(messages, chunk_queue.put_nowait)
            except Exception:
                stream_error[0] = True
                logger.exception("流式调用失败，回退非流式")
                return await llm_fn(messages, None)
            finally:
                await chunk_queue.put(None)  # 哨兵：排空循环结束

        llm_task = asyncio.create_task(_llm_with_stream())
        # 每轮一个闸门：扣住轮首的英文旁白（详见 src/stream_gate.py）
        gate = OpeningGate()
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                if stream_error[0]:
                    continue  # 流式已失败：丢弃残留缓冲，不输出残缺片段
                out = gate.feed(chunk)
                if out:
                    streamed_chunks += 1
                    _emit({"type": "text", "content": out})
            response = await llm_task
            # 收尾：本轮扣着没发的，有 tool_use 就是旁白（丢），没有就是纯英文回答（补发）
            tail = gate.finish(had_tool_use=bool(response.get("tool_uses")))
            if tail:
                streamed_chunks += 1
                _emit({"type": "text", "content": tail})
        finally:
            if not llm_task.done():
                llm_task.cancel()

        return {
            "thinking_steps": thinking,
            "round": rnd + 1,
            "answer": response.get("text", ""),
            "pending_tool_uses": list(response.get("tool_uses") or []),
            "stream_error": stream_error[0],
            "streamed_chunks": streamed_chunks,
        }

    async def _tools(state: AgentState) -> dict:
        """逐个执行本轮的工具调用：先过库外二次校验，再丢线程池执行。"""
        question = state.get("question", "")
        messages = list(state.get("messages", []))
        sources = state.get("sources", [])
        refusal = state.get("refusal")
        thinking: list[str] = []
        base = len(state.get("thinking_steps", []))

        def _next_step(text: str) -> str:
            thinking.append(_fmt_step(base + len(thinking) + 1, text))
            return thinking[-1]

        for tu in state.get("pending_tool_uses", []):
            name = tu["name"]
            inp = tu.get("input", {})
            tid = tu.get("id", f"tool_{state.get('round', 0)}")

            _emit({"type": "tool_call", "tool": name, "args": inp})

            # 同步阻塞的工具执行（ChromaDB 查询 / BM25 打分 / weather 请求）
            # 丢线程池执行，避免卡住事件循环（多人并发问答互不阻塞）。
            # ⚠️ 必须保持 asyncio.to_thread：工作线程里没有 running loop，
            # `_search_knowledge_base` 的同步桥才会走安全的 asyncio.run 分支；
            # 若改成直接 await 调用，会落到 ThreadPoolExecutor 降级分支并阻塞事件循环数秒。
            blocked = guard_fn(name, question)
            result_text = (
                blocked if blocked
                else await asyncio.to_thread(tool_fn, name, inp)
            )

            if name == search_tool_name:
                ev_sources = parse_sources_fn(result_text)
                sources = ev_sources
                _emit({
                    "type": "tool_result", "tool": name,
                    "success": True, "output": result_text, "sources": ev_sources,
                })
                # 两种拒答都要在"检索到 N 条"判定之前，否则会出现
                # "检索到 0 条" + "已拒答" 两个自相矛盾的步骤。
                reason = classify_refusal_fn(result_text)
                if reason:
                    refusal = reason
                    step = _next_step(_REFUSAL_STEP.get(reason, "已拒答"))
                else:
                    step = _next_step(f"📋 检索到 {len(ev_sources)} 条相关结果")
                _emit({"type": "thinking", "step": step})
            else:
                _emit({
                    "type": "tool_result", "tool": name,
                    "success": True, "output": result_text, "sources": None,
                })
                step = _next_step(f"📋 返回: {result_text[:120]}")
                _emit({"type": "thinking", "step": step})

            # 每个工具单独成对 append（不是合并进一条 assistant 消息）—— 与原实现一致
            messages.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
            })
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tid, "content": result_text}],
            })

        return {
            "messages": messages,
            "thinking_steps": thinking,
            "sources": sources,
            "refusal": refusal,
        }

    async def _finalize(state: AgentState) -> dict:
        """正常收工：无待执行的工具调用，本轮文本即最终答案。"""
        elapsed = time.time() - state.get("started_at", time.time())
        rnd = state.get("round", 0)
        base = len(state.get("thinking_steps", []))
        # 这一步只进 thinking_steps、不单独 emit —— 随 done 事件一次性下发（原行为）
        step = _fmt_step(base + 1, f"✅ 生成完成 | {rnd} 轮 | 耗时 {elapsed:.1f}s")

        answer = state.get("answer", "")
        # 已流式输出的（有增量且无失败）不必重复发整段；
        # 未流式/流式失败场景在此整段补发，保证答案完整。
        if state.get("stream_error") or state.get("streamed_chunks", 0) == 0:
            _emit({"type": "text", "content": answer})

        steps = list(state.get("thinking_steps", [])) + [step]
        _emit({"type": "done", "thinking": steps, "sources": list(state.get("sources", []))})
        return {"thinking_steps": [step], "answer": answer}

    async def _exhausted(state: AgentState) -> dict:
        """轮数用尽仍有工具调用：强制终止，给出兜底文案。"""
        base = len(state.get("thinking_steps", []))
        step = _fmt_step(base + 1, f"⚠️ 达到最大轮数 {state.get('max_rounds', max_rounds)}，强制终止")
        fallback = "抱歉，处理超时，请简化问题后重试。"
        _emit({"type": "text", "content": fallback})

        steps = list(state.get("thinking_steps", [])) + [step]
        _emit({"type": "done", "thinking": steps, "sources": list(state.get("sources", []))})
        return {"thinking_steps": [step], "answer": fallback}

    # ── 路由 ──────────────────────────────────────────────

    def _route_after_llm(state: AgentState) -> str:
        """LLM 要工具吗？要就进工具环，不要就是本轮文本即最终答案。

        返回值**必须是**下方 path_map 里的 key。langgraph 的 `_branch.py` 是
        `self.ends[r]`，返回未映射的 key 直接 KeyError，而调用侧（SSE 流）没有
        兜住一切的 try/except。所以每个 return 都是字面量（同 `recall_guard._route`）。
        """
        return "tools" if state.get("pending_tool_uses") else "finalize"

    def _route_after_tools(state: AgentState) -> str:
        """轮数用尽吗？用尽则强制终止，否则回 LLM 下一轮。

        放在 tools **之后**而不是 llm 之前，是为了对齐原实现：原来的
        `for rnd in range(max_rounds)` 在最后一轮**仍然执行工具**再走兜底，
        若提到 llm 之前判定，最后一轮的工具就不会执行了（行为差异）。
        """
        if state.get("round", 0) < state.get("max_rounds", max_rounds):
            return "llm"
        return "exhausted"

    # ── 装配 ──────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("prepare", _prepare)
    graph.add_node("llm", _llm)
    graph.add_node("tools", _tools)
    graph.add_node("finalize", _finalize)
    graph.add_node("exhausted", _exhausted)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "llm")
    graph.add_conditional_edges(
        "llm", _route_after_llm,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools,
        {"llm": "llm", "exhausted": "exhausted"},  # ← tools→llm 就是那个环
    )
    graph.add_edge("finalize", END)
    graph.add_edge("exhausted", END)

    return graph.compile()


def recursion_limit(max_rounds: int) -> int:
    """由轮数上限推导 `recursion_limit`，**不要拍死数**。

    与 `recall_guard` 同一个坑：LangGraph 1.2.10 的 `recursion_limit` 默认是 10007
    （不是旧版的 25），靠它兜底等于没有兜底——环真跑飞会空转到一万步才报错。
    真正的终止条件是 state 里的 `round` 计数器，这个值只是"计数器万一写错"的二次保险。

    它必须**始终宽于**计数器：上限推导（而非死数）才能保证 `max_rounds` 调大时
    不会反过来先炸 `GraphRecursionError`。每圈 `llm → tools` 两个 superstep，
    再加 prepare/finalize 的头尾，`4*n+6` 留足余量。
    """
    return 4 * max(1, max_rounds) + 6
