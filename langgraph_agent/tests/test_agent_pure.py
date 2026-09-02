"""纯函数单元测试：不依赖网络 / Chroma / Redis，直接验证核心逻辑。

覆盖：
- _parse_route        Router 输出解析 + 噪声兜底
- should_continue     轮数上限 / tool_use 判定
- extract_final_answer 只取最后一条 assistant 文本（防思考过程泄漏）
- _safe_gbk           Windows GBK 终端防崩
"""
from agent import MAX_ROUNDS, _parse_route, _safe_gbk, extract_final_answer, should_continue


class TestParseRoute:
    def test_kb(self):
        assert _parse_route("kb") == "kb"
        assert _parse_route("KB") == "kb"
        assert _parse_route("  kb\n") == "kb"

    def test_calc(self):
        assert _parse_route("calc") == "calc"
        assert _parse_route("cALc") == "calc"

    def test_chat(self):
        assert _parse_route("chat") == "chat"

    def test_noise_fallback_to_chat(self):
        # 识别不出 → 兜底走 chat，绝不拒绝用户
        assert _parse_route("") == "chat"
        assert _parse_route("???!!!") == "chat"
        assert _parse_route("我不太确定") == "chat"

    def test_keyword_inside_noise(self):
        # 只要文本里含 kb/calc 就命中（先判 calc，避免 kb 先命中）
        assert _parse_route("请调用kb子代理") == "kb"
        assert _parse_route("这个需要calc处理") == "calc"
        assert _parse_route("先用calc再用kb") == "calc"


class TestShouldContinue:
    def test_round_limit_wins(self):
        # 达到上限时即使还要调工具也必须结束
        state = {
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "1"}]}],
            "round_count": MAX_ROUNDS,
        }
        assert should_continue(state) == "__end__"

    def test_tool_use_goes_to_tools(self):
        state = {
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "1"}]}],
            "round_count": 1,
        }
        assert should_continue(state) == "tools"

    def test_text_ends(self):
        state = {
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "你好"}]}],
            "round_count": 1,
        }
        assert should_continue(state) == "__end__"

    def test_empty_state_ends(self):
        assert should_continue({"messages": [], "round_count": 0}) == "__end__"


class TestExtractFinalAnswer:
    def test_only_last_assistant_text(self):
        state = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "中间思考过程，不应泄漏"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "这才是最终答案"}]},
        ]}
        assert extract_final_answer(state) == "这才是最终答案"

    def test_text_after_tool_use(self):
        state = {"messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "name": "calculate"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "结果是 900"}]},
        ]}
        assert extract_final_answer(state) == "结果是 900"

    def test_no_assistant_message(self):
        assert extract_final_answer({"messages": [{"role": "user", "content": "q"}]}) == "（Agent 未生成回复）"


class TestSafeGbk:
    def test_plain_text_unchanged(self):
        assert _safe_gbk("中文 abc 123") == "中文 abc 123"

    def test_emoji_replaced_with_question_mark(self):
        assert "?" in _safe_gbk("你好😀")


class TestMaxRoundsConstant:
    def test_safety_cap_is_10(self):
        assert MAX_ROUNDS == 10
