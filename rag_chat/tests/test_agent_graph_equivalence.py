"""新旧两条主循环路径的等价性 —— 「改的是编排，不是行为」的可执行定义。

同一条对话、同一个假 LLM、同一个假工具，分别跑：

- `AGENT_GRAPH=off` → `AgentLoop._run_stream_legacy`（原手写 for-step 循环）
- `AGENT_GRAPH=on`  → `AgentLoop._run_stream_graph`（LangGraph 状态机）

断言两条路径产出的**事件序列逐项相等**（时间戳与耗时归一化后）。

这层测试是重构的安全网：图重写里最容易出的错不是崩溃，而是
「少发了一个思考步骤」「多跑了一轮工具」这种静默的行为漂移 ——
那些不会让任何单测变红，只会让线上输出悄悄变了样。

零网络、零 LLM 调用：`_call_llm` 与 `execute_tool` 都被替换成假件。
"""
import asyncio
import re

import pytest

import app_backend
from app_backend import AgentLoop

# 思考步骤里带时间戳与耗时，两条路径跑在不同时刻，比对前先归一化
_TS = re.compile(r"\[\d{2}:\d{2}:\d{2}\] ")
_ELAPSED = re.compile(r"耗时 [\d.]+s")


def _norm(step: str) -> str:
    return _ELAPSED.sub("耗时 Xs", _TS.sub("", step))


def _scripted_llm(calls: list, rounds: list[dict]):
    """假 LLM：按 rounds 顺序返回；gallery 里的 deltas 经 on_delta 推给闸门。"""

    async def _fake(messages, api_key, on_delta=None, **kw):
        calls.append([dict(m) for m in messages])
        r = rounds[min(len(calls) - 1, len(rounds) - 1)]
        for d in r.get("deltas", []):
            if on_delta:
                on_delta(d)
        return {
            "text": r.get("text", ""),
            "tool_uses": r.get("tool_uses", []),
            "stop_reason": r.get("stop_reason", "end_turn"),
        }

    return _fake


# 真实检索结果的格式（_parse_sources 按这个正则解析，用假文本会被解析成 0 条来源）
_FAKE_RETRIEVAL = (
    "[1] 相似度: 85.0% | 来源: 通知.txt | 日期: 2026-04-01 | 年份: 2026"
    " | 分类: 通知 | 站点: 官网\n"
    "   原文链接: http://example.com/a\n"
    "   片段: 河南工学院运动会 4 月 16-17 日举行...\n"
)


def _fake_tool(spy: list):
    def _tool(name, input_):
        spy.append((name, input_))
        return _FAKE_RETRIEVAL

    return _tool


def _collect(events) -> list:
    """把事件对象压成可比较的元组序列。"""
    out = []
    for ev in events:
        name = type(ev).__name__
        if name == "ThinkingEvent":
            out.append(("thinking", _norm(ev.step)))
        elif name == "ToolCallEvent":
            out.append(("tool_call", ev.tool, ev.args))
        elif name == "ToolResultEvent":
            out.append(("tool_result", ev.tool, ev.success, ev.output, ev.sources))
        elif name == "TextEvent":
            out.append(("text", ev.content))
        elif name == "DoneEvent":
            out.append(("done", [_norm(s) for s in ev.thinking], ev.sources))
        else:
            out.append((name,))
    return out


def _drive(monkeypatch, rounds, *, question="运动会什么时候开", switch, history=None):
    """跑一条路径，返回 (归一化事件序列, 事件对象列表, LLM 收到的 messages)。"""
    calls: list = []
    tool_calls: list = []
    monkeypatch.setattr(app_backend, "_call_llm", _scripted_llm(calls, rounds))
    monkeypatch.setattr(app_backend, "execute_tool", _fake_tool(tool_calls))
    monkeypatch.setattr(app_backend, "AGENT_GRAPH", switch)

    loop = AgentLoop(api_key="fake-key")
    events = []

    async def _go():
        async for ev in loop.run_stream(question, history):
            events.append(ev)

    asyncio.run(_go())
    return _collect(events), events, calls, tool_calls


# ── 场景 ──────────────────────────────────────────────────────

SCENARIO_TOOL_THEN_ANSWER = [
    {"deltas": ["I'll search the knowledge base for it."],  # 英文旁白：应被闸门丢弃
     "tool_uses": [{"id": "t1", "name": "search_knowledge_base",
                    "input": {"query": "运动会"}}],
     "stop_reason": "tool_use"},
    {"text": "根据检索，运动会 4 月 16-17 日举行。",
     "deltas": ["根据检索，", "运动会 4 月 16-17 日举行。"]},
]

SCENARIO_PLAIN_ANSWER = [
    {"text": "你好，有什么可以帮你？", "deltas": ["你好，", "有什么可以帮你？"]},
]

SCENARIO_NON_STREAM = [
    {"text": "非流式整段答案"},  # 无 deltas → 收尾整段补发
]

SCENARIO_MULTI_TOOL = [
    {"tool_uses": [
        {"id": "t1", "name": "search_knowledge_base", "input": {"query": "运动会"}},
        {"id": "t2", "name": "get_current_time", "input": {}},
    ]},
    {"text": "答案", "deltas": ["答案"]},
]

SCENARIO_TOOL_ONLY_EXHAUST = [
    {"tool_uses": [{"id": "t1", "name": "search_knowledge_base", "input": {"query": "x"}}]},
]


@pytest.mark.parametrize("rounds,name", [
    (SCENARIO_TOOL_THEN_ANSWER, "工具轮+回答轮"),
    (SCENARIO_PLAIN_ANSWER, "纯回答"),
    (SCENARIO_NON_STREAM, "非流式补发"),
    (SCENARIO_MULTI_TOOL, "单轮多工具"),
    (SCENARIO_TOOL_ONLY_EXHAUST, "轮数用尽兜底"),
])
def test_两条路径事件序列完全一致(monkeypatch, rounds, name):
    old, _, calls_old, tools_old = _drive(monkeypatch, rounds, switch="off")
    new, _, calls_new, tools_new = _drive(monkeypatch, rounds, switch="on")

    assert old == new, f"[{name}] 事件序列漂移"
    assert calls_old == calls_new, f"[{name}] 发给 LLM 的 messages 漂移"
    assert tools_old == tools_new, f"[{name}] 工具调用参数漂移"


def test_等价性测试真的覆盖了分支(monkeypatch):
    """防止上面的用例因为场景太简单而空转：确认工具轮确实走了工具。"""
    _, _, _, tools = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="on")
    assert len(tools) == 1
    assert tools[0][0] == "search_knowledge_base"

    _, events, calls, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="on")
    assert len(calls) == 2, "一轮工具 + 一轮回答 = 两次 LLM 调用"
    assert [e.content for e in events if type(e).__name__ == "TextEvent"] == [
        "根据检索，", "运动会 4 月 16-17 日举行。",
    ]


def test_两条路径都在英文旁白后丢弃它(monkeypatch):
    """首句闸门的接线不能只在一条路径上生效。"""
    for switch in ("off", "on"):
        _, events, _, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch=switch)
        text = "".join(e.content for e in events if type(e).__name__ == "TextEvent")
        assert "I'll search" not in text, f"agent_graph={switch} 时旁白漏了"
        assert text.startswith("根据检索")


def test_两条路径的思考步骤内容一致(monkeypatch):
    old, _, _, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="off")
    new, _, _, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="on")

    steps_old = [s for kind, *rest in old if kind == "thinking" for s in rest]
    steps_new = [s for kind, *rest in new if kind == "thinking" for s in rest]
    assert steps_old == steps_new
    assert any("检索到 1 条相关结果" in s for s in steps_new)


def test_两条路径的收尾步骤数一致(monkeypatch):
    """DoneEvent 里的 thinking 列表长度与内容都要一致（含"生成完成"那一步）。"""
    old, _, _, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="off")
    new, _, _, _ = _drive(monkeypatch, SCENARIO_TOOL_THEN_ANSWER, switch="on")

    done_old = [rest[0] for kind, *rest in old if kind == "done"][0]
    done_new = [rest[0] for kind, *rest in new if kind == "done"][0]
    assert len(done_old) == len(done_new)
    assert done_old[-1] == done_new[-1]
    assert "生成完成" in done_new[-1]


def test_轮数用尽时两条路径都给兜底文案(monkeypatch):
    for switch in ("off", "on"):
        _, events, _, _ = _drive(monkeypatch, SCENARIO_TOOL_ONLY_EXHAUST, switch=switch)
        text = "".join(e.content for e in events if type(e).__name__ == "TextEvent")
        assert "抱歉" in text, f"agent_graph={switch} 没给兜底文案"


def test_历史消息两条路径处理一致(monkeypatch):
    history = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "⏳ 正在生成..."},
        {"role": "user", "content": [{"type": "text", "text": "块式历史"}]},
    ]
    _, _, calls_old, _ = _drive(monkeypatch, SCENARIO_PLAIN_ANSWER,
                                switch="off", history=history)
    _, _, calls_new, _ = _drive(monkeypatch, SCENARIO_PLAIN_ANSWER,
                                switch="on", history=history)
    assert calls_old == calls_new
    assert not any("⏳" in str(m.get("content")) for m in calls_new[0])


def test_默认开关是off(monkeypatch):
    """默认必须走老路径 —— 新路径先并存，验证过了再谈默认打开。"""
    assert app_backend.AGENT_GRAPH == "off"
