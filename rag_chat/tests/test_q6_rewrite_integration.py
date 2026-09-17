"""端到端复现 Q6 失败路径：LLM 把工具参数里的校名改写掉了。

## 为什么要这层测试

`test_out_of_kb_guard.py` 测的是闸门函数本身（纯字符串）。这一层测的是**接线**：
`_guard_tool_call` 有没有真的接在 `AgentLoop.run_stream` 的工具调用前。
少了这层，闸门写得再对也可能是死代码。

## 复现的路径

用户问「河北工学院的录取分数线是多少？」，回答那级 LLM 某次把工具参数写成
`query='招生录取分数'`——校名没了，检索层闸门（只看得到工具参数）拦不住，
返回的本校分数被如实列进回答。实测同一份代码两遍得到 60/60 与 50/60。

这里用一个**假 LLM** 把这次改写钉死：它第一轮必定返回那个丢了校名的 tool_use，
不再依赖模型随机性。零网络。
"""
import asyncio

import pytest

import app_backend
from app_backend import AgentLoop, ThinkingEvent

QUESTION = "河北工学院的录取分数线是多少？"
# 实测那次 LLM 实际写出来的工具参数：校名被改写掉了
REWRITTEN_QUERY = "招生录取分数"


def _fake_llm_returning_rewritten_query(calls: list):
    """假 LLM：第一轮返回丢了校名的 tool_use，第二轮给最终答案。"""

    async def _fake(messages, api_key, on_delta=None, **kw):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "text": "",
                "tool_uses": [
                    {
                        "id": "t1",
                        "name": "search_knowledge_base",
                        "input": {"query": REWRITTEN_QUERY},
                    }
                ],
                "stop_reason": "tool_use",
            }
        # 第二轮：把上一轮的工具结果原样转述（真实模型也会照抄拒答文案）
        tool_result = messages[-1]["content"][0]["content"]
        return {"text": tool_result, "tool_uses": [], "stop_reason": "end_turn"}

    return _fake


async def _drive(monkeypatch, question: str) -> tuple[str, list[str]]:
    calls: list = []
    monkeypatch.setattr(app_backend, "_call_llm", _fake_llm_returning_rewritten_query(calls))

    loop = AgentLoop(api_key="fake-key")
    answer, steps = "", []
    async for ev in loop.run_stream(question):
        if isinstance(ev, ThinkingEvent):
            steps.append(ev.step)
        elif type(ev).__name__ == "TextEvent":
            answer += ev.content
    return answer, steps


@pytest.mark.parametrize("chunk", [["河北工学院的录取分数线是多少？"]])
def test_校名被改写掉时仍被拒绝而非返回本校数字(monkeypatch, chunk):
    """工具参数里没有校名，但用户原话里有——必须拒答，不能把本校分数答出来。"""
    answer, steps = asyncio.run(_drive(monkeypatch, chunk[0]))

    assert "河北工学院" in answer, "拒答文案里要点名用户问的那所学校"
    assert "无法回答" in answer
    # 关键：没有把本校分数当答案端出去
    assert "河南工学院" not in answer or "没有" in answer


def test_库外拦截步骤真的流到了客户端(monkeypatch):
    """只 add_step 不 yield 的话这个步骤在流式过程中根本看不到——
    评测回放和实时日志都查不出走没走这条分支。"""
    _, steps = asyncio.run(_drive(monkeypatch, QUESTION))

    assert any("库外题拦截" in s for s in steps), steps


def test_命中拦截后不再跑检索(monkeypatch):
    """二次校验在 execute_tool 之前——命中就不必白跑一次 Chroma + BM25。"""
    called = []
    monkeypatch.setattr(app_backend, "execute_tool", lambda *a, **k: called.append(a) or "")

    asyncio.run(_drive(monkeypatch, QUESTION))

    assert called == []


def test_正常问题照常检索(monkeypatch):
    """别把闸门做成"凡事先拒"——主体校问题必须正常走到检索。"""
    called = []
    monkeypatch.setattr(app_backend, "execute_tool", lambda *a, **k: called.append(a) or "检索结果")

    asyncio.run(_drive(monkeypatch, "河南工学院食堂几点开门"))

    assert len(called) == 1
    assert called[0][0] == "search_knowledge_base"
