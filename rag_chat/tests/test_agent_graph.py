"""测试 Agent 主循环图（src/agent_graph.py）—— 节点逻辑、条件边、工具环、终止保证。

全部通过 DI 注入 fake `llm_fn` / `tool_fn` / `guard_fn`，**零网络、零 LLM 调用、
不依赖 chromadb/torch**。图本身也不 import app_backend，所以本文件不需要拉起向量库。

不用 `@pytest.mark.asyncio`（.venv 没装 pytest-asyncio），照 `test_recall_guard.py`
的写法用 `asyncio.run`。
"""
import asyncio

import pytest

from src.agent_graph import build_agent_graph, recursion_limit

SEARCH = "search_knowledge_base"


# ── 辅助 ──────────────────────────────────────────────────────


class _Spy:
    """记录调用次数与入参。"""

    def __init__(self):
        self.llm = 0
        self.tool = 0
        self.guard = 0
        self.llm_calls: list[list[dict]] = []      # 每轮发给 LLM 的 messages
        self.tool_calls: list[tuple] = []          # (name, input)
        self.guard_calls: list[tuple] = []         # (name, question)


def _make_llm(spy, responses):
    """造一个假 LLM。responses 按调用次序取用，最后一项会被重复使用。

    每项：`{"text", "tool_uses", "deltas"}`；`{"raise_": True}` 表示这次调用抛异常
    （模拟流式解析失败，触发回退非流式）。
    """

    async def _llm(messages, on_delta):
        i = spy.llm
        spy.llm += 1
        spy.llm_calls.append(messages)
        r = responses[min(i, len(responses) - 1)]
        if r.get("raise_"):
            raise RuntimeError("模拟流式解析失败")
        if on_delta:
            for d in r.get("deltas", []):
                on_delta(d)
        return {
            "text": r.get("text", ""),
            "tool_uses": r.get("tool_uses", []),
            "stop_reason": "end_turn",
        }

    return _llm


def _tool_use(name=SEARCH, input_=None, tid="t1"):
    return {"name": name, "input": input_ or {"query": "测试"}, "id": tid}


def _run(
    spy,
    responses,
    *,
    question="测试问题",
    history=None,
    max_rounds=6,
    guard=None,
    sources=None,
    classify=None,
    tool_result="【1】片段内容\n来源: doc.md",
    max_history_turns=10,
):
    """建图跑一遍，返回 (custom 事件列表, 终态 state)。"""
    tool_result = tool_result

    def _tool_fn(name, input_):
        spy.tool += 1
        spy.tool_calls.append((name, input_))
        return tool_result

    def _guard_fn(name, user_question):
        spy.guard += 1
        spy.guard_calls.append((name, user_question))
        return guard(name, user_question) if guard else None

    graph = build_agent_graph(
        llm_fn=_make_llm(spy, responses),
        tool_fn=_tool_fn,
        guard_fn=_guard_fn,
        parse_sources_fn=lambda text: sources if sources is not None else [{"source": "doc.md"}],
        classify_refusal_fn=classify or (lambda text: None),
        max_rounds=max_rounds,
        max_history_turns=max_history_turns,
    )

    async def _go():
        events, final = [], None
        async for mode, chunk in graph.astream(
            {"question": question, "history": history, "max_rounds": max_rounds},
            {"recursion_limit": recursion_limit(max_rounds)},
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                events.append(chunk)
            else:
                final = chunk
        return events, final

    return asyncio.run(_go())


def _of(events, etype):
    return [e for e in events if e["type"] == etype]


def _text_of(events):
    return "".join(e["content"] for e in _of(events, "text"))


# ── 主干：无工具调用 → 直接收工 ────────────────────────────────


def test_无工具调用时直接收工():
    spy = _Spy()
    events, final = _run(spy, [{"text": "你好，这是回答", "deltas": ["你好", "，这是回答"]}])

    assert spy.llm == 1
    assert spy.tool == 0
    assert _text_of(events) == "你好，这是回答"
    assert final["answer"] == "你好，这是回答"


def test_事件顺序与思考步骤对齐原实现():
    spy = _Spy()
    events, final = _run(spy, [{"text": "答案", "deltas": ["答案"]}])

    types = [e["type"] for e in events]
    assert types[0] == "thinking"           # 接收 Query
    assert types[1] == "thinking"           # 第 1 轮推理
    assert "text" in types
    assert types[-1] == "done"

    thinking = _of(events, "done")[0]["thinking"]
    assert "接收 Query" in thinking[0]
    assert "第 1 轮推理" in thinking[1]
    assert "✅ 生成完成" in thinking[-1]


def test_未流式时整段补发答案():
    """LLM 没给增量（非流式路径）→ streamed_chunks==0 → 收尾整段补发。"""
    spy = _Spy()
    events, _ = _run(spy, [{"text": "整段答案", "deltas": []}])

    assert _text_of(events) == "整段答案"
    assert len(_of(events, "text")) == 1


# ── 工具环 ────────────────────────────────────────────────────


def test_工具调用后回到LLM再收工():
    """一轮工具 → 回 LLM → 无工具 → 收工。这是那个环。"""
    spy = _Spy()
    events, final = _run(spy, [
        {"tool_uses": [_tool_use()], "deltas": []},
        {"text": "根据检索，答案是A", "deltas": ["根据检索，答案是A"]},
    ])

    assert spy.llm == 2, "应该调了两次 LLM（一轮工具后回到 LLM）"
    assert spy.tool == 1
    assert final["answer"] == "根据检索，答案是A"

    assert len(_of(events, "tool_call")) == 1
    assert len(_of(events, "tool_result")) == 1
    steps = [e["step"] for e in _of(events, "thinking")]
    assert any("📋 检索到 1 条相关结果" in s for s in steps)


def test_工具结果的助手消息与用户消息成对append():
    """每个工具单独成对 append（assistant tool_use + user tool_result），与原实现一致。"""
    spy = _Spy()
    _run(spy, [
        {"tool_uses": [_tool_use(tid="a")], "deltas": []},
        {"text": "完", "deltas": ["完"]},
    ])

    second_call = spy.llm_calls[1]
    # 首条是用户问题，其后是 tool_use / tool_result 对
    assert second_call[0]["role"] == "user"
    assert second_call[-2]["content"][0]["type"] == "tool_use"
    assert second_call[-2]["content"][0]["id"] == "a"
    assert second_call[-1]["content"][0]["type"] == "tool_result"
    assert second_call[-1]["content"][0]["tool_use_id"] == "a"


def test_非检索工具的思考步骤走返回分支():
    spy = _Spy()
    events, _ = _run(spy, [
        {"tool_uses": [_tool_use(name="get_current_time", input_={})], "deltas": []},
        {"text": "完", "deltas": ["完"]},
    ])

    steps = [e["step"] for e in _of(events, "thinking")]
    assert any("📋 返回:" in s for s in steps)
    assert not any("检索到" in s for s in steps)


def test_sources是覆盖语义不是累积():
    """两轮检索，sources 应等于最后一次的结果（原实现 `sources = ev_sources`）。"""
    spy = _Spy()
    _, final = _run(
        spy,
        [
            {"tool_uses": [_tool_use(tid="t1")], "deltas": []},
            {"tool_uses": [_tool_use(tid="t2")], "deltas": []},
            {"text": "完", "deltas": ["完"]},
        ],
        sources=[{"source": "last.md"}],
    )
    assert final["sources"] == [{"source": "last.md"}]
    assert len(final["sources"]) == 1, "不该把两轮结果累积起来"


# ── 终止保证 ──────────────────────────────────────────────────


def test_轮数用尽走兜底文案():
    spy = _Spy()
    events, final = _run(spy, [{"tool_uses": [_tool_use()], "deltas": []}], max_rounds=3)

    assert spy.llm == 3, "恰好跑满 max_rounds 轮 LLM"
    assert final["answer"] == "抱歉，处理超时，请简化问题后重试。"
    steps = _of(events, "done")[0]["thinking"]
    assert any("达到最大轮数 3" in s for s in steps)


@pytest.mark.parametrize("max_rounds", [1, 3, 5, 9])
def test_判官永远要工具也能按计数器停住(max_rounds):
    """终止责任在 state 的 round 计数器，不靠 recursion_limit 兜底。

    LangGraph 1.2.10 的 `recursion_limit` 默认是 10007（不是旧版的 25），靠它兜底
    等于没有兜底——真跑飞会空转到一万步、每圈都在烧 LLM。本用例把上限放大到 9
    验证它确实按计数停，而不是靠外层兜底，也不会先炸 GraphRecursionError。
    """
    spy = _Spy()
    _, final = _run(spy, [{"tool_uses": [_tool_use()], "deltas": []}], max_rounds=max_rounds)

    assert spy.llm == max_rounds
    assert spy.tool == max_rounds
    assert final["answer"].startswith("抱歉")


@pytest.mark.parametrize("max_rounds", [1, 2, 6, 12])
def test_recursion_limit始终宽于计数器(max_rounds):
    """上限必须由计数器推导而非死数，否则调大 max_rounds 会先炸 GraphRecursionError。"""
    assert recursion_limit(max_rounds) > max_rounds


def test_recursion_limit对非法值也不返回0():
    assert recursion_limit(0) > 0
    assert recursion_limit(-5) > 0


# ── 库外闸门与拒答分类 ────────────────────────────────────────


def test_库外闸门命中时不执行工具():
    spy = _Spy()
    events, final = _run(
        spy,
        [{"tool_uses": [_tool_use()], "deltas": []}, {"text": "完", "deltas": ["完"]}],
        guard=lambda name, q: "【库外题拦截】该问题超出知识库范围。",
        classify=lambda text: "out_of_kb" if text.startswith("【库外题拦截") else None,
    )

    assert spy.tool == 0, "被闸门拦下就不该跑检索"
    assert final["refusal"] == "out_of_kb"
    assert _of(events, "tool_result")[0]["output"].startswith("【库外题拦截")
    steps = [e["step"] for e in _of(events, "thinking")]
    assert any("🚫 库外题拦截，已拒答" in s for s in steps)


def test_闸门收到的是用户原话():
    """二次校验必须用用户原话，不是 LLM 改写过的工具参数。"""
    spy = _Spy()
    _run(
        spy,
        [{"tool_uses": [_tool_use(input_={"query": "招生录取分数"})], "deltas": []},
         {"text": "完", "deltas": ["完"]}],
        question="河北工学院的录取分数线是多少",
    )
    assert spy.guard_calls[0] == (SEARCH, "河北工学院的录取分数线是多少")


def test_闸门只对检索工具生效():
    spy = _Spy()
    _run(
        spy,
        [{"tool_uses": [_tool_use(name="calculate", input_={})], "deltas": []},
         {"text": "完", "deltas": ["完"]}],
    )
    assert spy.guard_calls == [("calculate", "测试问题")]


def test_召回自检拒答被分类并显示对应步骤():
    spy = _Spy()
    events, final = _run(
        spy,
        [{"tool_uses": [_tool_use()], "deltas": []}, {"text": "完", "deltas": ["完"]}],
        tool_result="【召回自检未通过】知识库中没有足够依据。",
        classify=lambda text: "recall_guard" if text.startswith("【召回自检") else None,
    )

    assert final["refusal"] == "recall_guard"
    steps = [e["step"] for e in _of(events, "thinking")]
    assert any("🛡️ 召回自检未通过，已拒答" in s for s in steps)


def test_拒答时不显示检索到N条():
    """两种拒答都要在「检索到 N 条」判定之前，否则出现自相矛盾的两个步骤。"""
    spy = _Spy()
    events, _ = _run(
        spy,
        [{"tool_uses": [_tool_use()], "deltas": []}, {"text": "完", "deltas": ["完"]}],
        tool_result="【库外题拦截】超出范围。",
        classify=lambda text: "out_of_kb" if text.startswith("【库外题拦截") else None,
    )
    steps = [e["step"] for e in _of(events, "thinking")]
    assert not any("检索到" in s for s in steps)


def test_未知拒答原因也有兜底文案():
    """分类函数返回了没登记的原因时，不能 KeyError 打断流。"""
    spy = _Spy()
    events, _ = _run(
        spy,
        [{"tool_uses": [_tool_use()], "deltas": []}, {"text": "完", "deltas": ["完"]}],
        classify=lambda text: "some_new_reason",
    )
    steps = [e["step"] for e in _of(events, "thinking")]
    assert any(s.endswith("已拒答") for s in steps)


# ── 首句闸门（OpeningGate）接线 ───────────────────────────────


def test_工具轮的英文旁白被丢弃():
    spy = _Spy()
    events, _ = _run(spy, [
        {"deltas": ["I'll search the knowledge base for information."],
         "tool_uses": [_tool_use()]},
        {"text": "答案", "deltas": ["答案"]},
    ])

    assert "I'll search" not in _text_of(events)
    assert _text_of(events) == "答案"


def test_纯英文回答不被吃掉():
    """整轮无汉字且无 tool_use → 是一段正常英文回答，必须整段补发。"""
    spy = _Spy()
    events, _ = _run(spy, [{"text": "Hello there", "deltas": ["Hello there"]}])

    assert _text_of(events) == "Hello there"


def test_中文正文几乎无延迟放行():
    """正文以汉字开头时，第一个增量就应放行（扣住的是空前缀）。"""
    spy = _Spy()
    events, _ = _run(spy, [{"text": "你好", "deltas": ["你好"]}])

    assert len(_of(events, "text")) == 1
    assert _of(events, "text")[0]["content"] == "你好"


# ── 流式失败回退 ──────────────────────────────────────────────


def test_流式失败回退非流式并整段补发():
    spy = _Spy()
    events, final = _run(spy, [{"raise_": True}, {"text": "回退后的答案"}])

    assert spy.llm == 2, "第一次流式抛异常后应回退重发一次"
    assert final["answer"] == "回退后的答案"
    assert _text_of(events) == "回退后的答案", "回退路径必须整段补发，不能丢答案"
    assert final["stream_error"] is True


def test_流式失败不输出残缺片段():
    """流式已失败时，队列里的残留缓冲要丢弃，不能吐半截。"""
    spy = _Spy()
    events, _ = _run(spy, [
        {"raise_": True, "deltas": ["半截英文narration"]},
        {"text": "完整答案"},
    ])
    assert "半截" not in _text_of(events)


# ── 历史处理 ──────────────────────────────────────────────────


def test_历史截断且跳过占位消息():
    spy = _Spy()
    history = (
        [{"role": "user", "content": f"旧问题{i}"} for i in range(12)]
        + [{"role": "assistant", "content": "⏳ 正在生成..."}]
        + [{"role": "user", "content": ""}]
    )
    _run(spy, [{"text": "新答案", "deltas": ["新答案"]}], history=history, max_history_turns=10)

    sent = spy.llm_calls[0]
    # 截断发生在过滤**之前**（与原实现一致）：先取最近 10 条 =
    # 8 条旧问题 + ⏳ 占位 + 空串，再丢掉后两条 → 8 条历史 + 当前问题 = 9。
    assert len(sent) == 9
    assert sent[-1] == {"role": "user", "content": "测试问题"}
    assert not any("⏳" in str(m.get("content")) for m in sent)


def test_历史里的列表型content被拍平成字符串():
    spy = _Spy()
    history = [{"role": "user", "content": [{"type": "text", "text": "块一"}, {"type": "text", "text": "块二"}]}]
    _run(spy, [{"text": "答", "deltas": ["答"]}], history=history)

    assert spy.llm_calls[0][0] == {"role": "user", "content": "块一块二"}


# ── 非流式入口 ────────────────────────────────────────────────


def test_非流式ainvoke下节点不炸():
    """get_stream_writer 在非流式上下文里是安全 no-op，节点不该因此抛异常。"""
    spy = _Spy()
    graph = build_agent_graph(
        llm_fn=_make_llm(spy, [{"text": "非流式答案"}]),
        tool_fn=lambda name, inp: "结果",
        guard_fn=lambda name, q: None,
        parse_sources_fn=lambda text: [],
        classify_refusal_fn=lambda text: None,
    )
    final = asyncio.run(
        graph.ainvoke({"question": "问题", "max_rounds": 6}, {"recursion_limit": 30})
    )

    assert final["answer"] == "非流式答案"
    assert spy.llm == 1
