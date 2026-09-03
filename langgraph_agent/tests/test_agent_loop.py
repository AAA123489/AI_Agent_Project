"""集成测试：mock call_llm，验证「Router → 子代理 → 工具 → 答案」全链路。

关键 mock 策略：所有 LLM 调用点都裸调用 agent 模块级 call_llm，
monkeypatch.setattr(agent, "call_llm", fake) 一处替换即可拦下所有网络调用。
"""
import asyncio

import agent as agent_module
from agent import MAX_ROUNDS, build_agent, extract_final_answer


# ── 假 LLM 工厂 ──────────────────────────────────────────────

def make_fake_llm(script):
    """script: 按调用顺序返回的 response 列表（每个形如 call_llm 的返回 dict）。"""
    queue = list(script)
    calls = []

    async def fake(messages, api_key="", api_url="", model="", system=None, tools=None):
        calls.append({"system": system, "tools": tools, "messages": messages})
        assert queue, f"fake LLM 被调用 {len(calls)} 次，超过 script 长度"
        return queue.pop(0)

    return fake, calls


def _text(text: str) -> dict:
    """纯文本回复。"""
    return {"content": [{"type": "text", "text": text}], "text": text, "tool_uses": [], "stop_reason": "end_turn"}


def _tool_use(name: str, input_: dict, tool_id: str = "toolu_1") -> dict:
    """工具调用回复。"""
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": input_}
    return {"content": [block], "text": "", "tool_uses": [block], "stop_reason": "tool_use"}


def _ainvoke(agent, message: str) -> dict:
    """同步跑一次 agent.ainvoke（测试里避免引入 pytest-asyncio）。"""
    async def run():
        return await agent.ainvoke({
            "messages": [{"role": "user", "content": message}],
            "round_count": 0,
            "route": "",
        })
    return asyncio.run(run())


# ── 测试 ─────────────────────────────────────────────────────

class TestCalcFlow:
    def test_calc_question_routes_and_answers(self, monkeypatch):
        script = [
            _text("calc"),                                                    # ① Router 分类
            _tool_use("calculate", {"expression": "(100+200)*3"}),            # ② calc 子代理要调工具
            _text("结果是 900"),                                               # ③ calc 子代理给答案
        ]
        fake, calls = make_fake_llm(script)
        monkeypatch.setattr(agent_module, "call_llm", fake)

        agent = build_agent()
        final = _ainvoke(agent, "(100+200)*3 等于多少")

        assert extract_final_answer(final) == "结果是 900"
        # Router 不调工具
        assert calls[0]["tools"] is None
        # 后续子代理调用带了 calculate 工具
        assert any(
            "calculate" in {t["name"] for t in (c["tools"] or [])} for c in calls[1:]
        )


class TestChatFallback:
    def test_garbage_route_falls_back_to_chat(self, monkeypatch):
        script = [
            _text("??? 完全无法识别"),      # Router 输出噪声 → _parse_route 兜底 chat
            _text("你好呀，我是闲聊子代理"),
        ]
        fake, _ = make_fake_llm(script)
        monkeypatch.setattr(agent_module, "call_llm", fake)

        agent = build_agent()
        final = _ainvoke(agent, "你是谁")

        assert extract_final_answer(final) == "你好呀，我是闲聊子代理"


class TestMaxRoundsSafety:
    def test_agent_stops_at_max_rounds(self, monkeypatch):
        # 假 LLM 永远要求调工具 → 应被 MAX_ROUNDS 强制终止，不死循环
        async def always_tool(messages, api_key="", api_url="", model="", system=None, tools=None):
            return _tool_use("calculate", {"expression": "1+1"})

        monkeypatch.setattr(agent_module, "call_llm", always_tool)

        agent = build_agent()
        final = _ainvoke(agent, "一直算下去")

        assert final["round_count"] == MAX_ROUNDS


class TestKbFlow:
    def test_kb_question_uses_hybrid_search(self, monkeypatch):
        script = [
            _text("kb"),                                                     # ① Router 分类
            _tool_use("search_knowledge_base", {"query": "河南工学院"}, "toolu_kb"),  # ② kb 要检索
            _text("河南工学院位于新乡市"),                                      # ③ kb 给答案
        ]
        fake, _ = make_fake_llm(script)
        monkeypatch.setattr(agent_module, "call_llm", fake)

        # kb 工具复用项目一混合检索：mock 委托点返回其展示文本
        def fake_p1_search(query, top_k):
            assert query == "河南工学院"
            return (
                "[1] 相似度: 90.0% | 来源: 官网.md | 日期: 2026-01-01 | 年份: 2026\n"
                "   原文链接: https://www.hait.edu.cn/info/1.htm\n"
                "   片段: 河南工学院位于新乡市"
            )

        monkeypatch.setattr("tools._p1_hybrid_search", fake_p1_search)

        agent = build_agent()
        final = _ainvoke(agent, "学校在哪里")

        assert extract_final_answer(final) == "河南工学院位于新乡市"
