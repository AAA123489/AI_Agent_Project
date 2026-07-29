"""测试 RAG Prompt 模板 —— build_rag_prompt。"""
import pytest
from prompts import build_rag_prompt


class TestWithContext:
    """有检索文档时的 prompt"""

    def test_with_context_and_history(self):
        query = "FastAPI 有什么特点？"
        context_docs = ["FastAPI 基于 ASGI，性能高", "它内置 Pydantic 数据校验"]
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ]
        prompt = build_rag_prompt(query, context_docs, history)

        assert "参考上下文" in prompt
        assert "FastAPI 基于 ASGI" in prompt
        assert "Pydantic 数据校验" in prompt
        assert "历史对话" in prompt
        assert "你好" in prompt
        assert "FastAPI 有什么特点？" in prompt
        assert "【你的回答】" in prompt

    def test_with_context_empty_history(self):
        query = "什么是 RAG？"
        context_docs = ["RAG 是检索增强生成的缩写"]
        history = []
        prompt = build_rag_prompt(query, context_docs, history)

        assert "参考上下文" in prompt
        assert "RAG 是检索增强生成" in prompt
        assert "历史对话" in prompt  # 模板中始终有这个标签
        assert "什么是 RAG？" in prompt

    def test_with_context_history_is_none(self):
        """history 为 None 时不应崩溃"""
        query = "测试问题"
        context_docs = ["一段上下文"]
        prompt = build_rag_prompt(query, context_docs, None)  # type: ignore
        assert "测试问题" in prompt
        assert "一段上下文" in prompt


class TestWithoutContext:
    """无检索文档时的纯对话模式"""

    def test_without_context_with_history(self):
        query = "今天天气怎么样？"
        context_docs = []
        history = [
            {"role": "user", "content": "嗨"},
            {"role": "assistant", "content": "嗨！"},
        ]
        prompt = build_rag_prompt(query, context_docs, history)

        assert "参考上下文" not in prompt
        assert "历史对话" in prompt
        assert "嗨" in prompt
        assert "今天天气怎么样？" in prompt
        assert "【你的回答】" in prompt

    def test_without_context_empty_history(self):
        query = "你是谁？"
        context_docs = []
        history = []
        prompt = build_rag_prompt(query, context_docs, history)

        assert "参考上下文" not in prompt
        assert "你是谁？" in prompt
        assert isinstance(prompt, str)

    def test_context_docs_is_none(self):
        """context_docs 为 None 时不应崩溃，应走纯对话模式"""
        query = "测试"
        prompt = build_rag_prompt(query, None, [])  # type: ignore
        assert "参考上下文" not in prompt
        assert "测试" in prompt

    def test_context_docs_with_empty_strings(self):
        """context_docs 包含空字符串时行为"""
        query = "测试"
        context_docs = ["", "  ", "\n"]
        prompt = build_rag_prompt(query, context_docs, [])
        # 白色空格应该被视为空上下文，走纯对话
        # 或者至少不崩溃
        assert "测试" in prompt


class TestPromptStructure:
    """prompt 结构完整性"""

    def test_ends_with_answer_marker(self):
        """每次调用都必须以「你的回答」结尾，方便 LLM 接续"""
        for context_docs, history in [
            (["doc"], [{"role": "user", "content": "hi"}]),
            ([], []),
            (["doc"], []),
            ([], [{"role": "user", "content": "hi"}]),
        ]:
            prompt = build_rag_prompt("测试", context_docs, history)
            assert prompt.rstrip().endswith("【你的回答】："), (
                f"Prompt should end with answer marker, got: ...{prompt[-30:]}"
            )

    def test_query_always_present(self):
        query = "一个独一无二的问题_XYZ123"
        for context_docs, history in [
            (["doc"], [{"role": "user", "content": "hi"}]),
            ([], []),
        ]:
            prompt = build_rag_prompt(query, context_docs, history)
            assert query in prompt, f"Query missing from prompt: {prompt[:100]}"

    def test_no_none_in_output(self):
        """None 不应出现在 prompt 文本中"""
        prompt = build_rag_prompt("你好", [], [])
        assert "None" not in prompt


class TestHistoryEdgeCases:
    """历史消息边界情况"""

    def test_history_missing_role_key(self):
        query = "测试"
        history = [{"content": "没有 role 字段的消息"}]
        prompt = build_rag_prompt(query, [], history)
        assert "没有 role 字段的消息" in prompt

    def test_history_missing_content_key(self):
        query = "测试"
        history = [{"role": "user"}]
        prompt = build_rag_prompt(query, [], history)
        assert "测试" in prompt  # 不崩溃即可

    def test_history_with_extra_fields(self):
        """多出的字段应被忽略，不报错"""
        query = "测试"
        history = [{"role": "user", "content": "hello", "timestamp": "2026-01-01", "id": 123}]
        prompt = build_rag_prompt(query, [], history)
        assert "hello" in prompt
