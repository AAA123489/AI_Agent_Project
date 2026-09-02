"""工具单元测试：安全计算 / 知识库检索 / 主动工作记忆（mock Redis 与 VectorStore）。

工作记忆测试用内存版 FakeRedis 替换 tools._get_memory_client，
不依赖真实 Redis；VectorStore 测试替换 tools._get_vector_store。
"""
import pytest

from tools import (
    _calculate,
    _safe_eval,
    clear_memory,
    execute_tool,
    save_memory,
    search_memory,
    set_session,
)


# ── 假 Redis（内存实现，模拟 set/get/keys/delete）──

class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)

    def keys(self, pattern):
        prefix = pattern[:-1]  # 去掉尾部 '*'
        return [k for k in self.data if k.startswith(prefix)]

    def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.data:
                del self.data[k]
                n += 1
        return n


# ── 安全计算 ─────────────────────────────────────────────────

class TestSafeEval:
    def test_basic_ops(self):
        assert _safe_eval("1+2") == 3
        assert _safe_eval("(100+200)*3") == 900
        assert _safe_eval("2 ** 10") == 1024
        assert _safe_eval("-5 + 3") == -2

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            _safe_eval("1/0")

    def test_reject_code_execution(self):
        # AST 白名单：函数调用、导入、语句都不允许
        for bad in ["__import__('os')", "open('x')", "lambda: 1", "1; print(1)"]:
            with pytest.raises(Exception):
                _safe_eval(bad)


class TestCalculate:
    def test_ok_and_error_strings(self):
        assert "[OK]" in _calculate("1+1")
        assert "2" in _calculate("1+1")

    def test_division_by_zero_friendly(self):
        # 除零不再抛异常，返回友好错误
        assert "[X]" in _calculate("1/0")

    def test_empty_expression(self):
        assert "[WARN]" in _calculate("")


# ── 工作记忆 ─────────────────────────────────────────────────

class TestMemoryTools:
    @pytest.fixture(autouse=True)
    def fake_redis(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr("tools._get_memory_client", lambda: fake)
        set_session("test_user")
        return fake

    def test_save_and_search(self):
        result = save_memory("用户学校", "张同学在河南工学院")
        assert "[OK]" in result
        result = search_memory("河南工学院")
        assert "张同学在河南工学院" in result
        assert "用户学校" in result

    def test_search_no_hit(self):
        assert "没有找到" in search_memory("不存在的关键词")

    def test_save_empty_key(self):
        assert "[WARN]" in save_memory("", "内容")

    def test_clear_one(self):
        save_memory("a", "内容A")
        save_memory("b", "内容B")
        assert "[OK]" in clear_memory("a")
        assert "没有找到" in search_memory("内容A")
        assert "内容B" in search_memory("内容B")

    def test_clear_all(self):
        save_memory("a", "内容A")
        assert "[OK]" in clear_memory("*")
        assert "没有找到" in search_memory("内容A")

    def test_execute_tool_dispatch(self):
        result = execute_tool("save_memory", {"key": "姓名", "content": "张三"})
        assert "[OK]" in result
        result = execute_tool("search_memory", {"query": "张三"})
        assert "张三" in result

    def test_session_isolation(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr("tools._get_memory_client", lambda: fake)

        set_session("user1")
        save_memory("学校", "user1 的记忆")
        set_session("user2")
        # user2 查不到 user1 的记忆
        assert "没有找到" in search_memory("学校")
        # 切回 user1 还在
        set_session("user1")
        assert "user1 的记忆" in search_memory("学校")

    def test_memory_unavailable_degrades(self, monkeypatch):
        # Redis 连不上 → 降级返回提示，不抛异常
        monkeypatch.setattr("tools._get_memory_client", lambda: None)
        assert "[WARN]" in save_memory("k", "v")
        assert "[WARN]" in search_memory("q")
        assert "[WARN]" in clear_memory("k")


# ── 知识库检索工具 ───────────────────────────────────────────

class TestSearchKnowledgeBase:
    def test_empty_query(self, monkeypatch):
        assert "[WARN]" in execute_tool("search_knowledge_base", {"query": " "})

    def test_empty_library(self, monkeypatch):
        class FakeVectorStore:
            def search_similar(self, query, n_results=5):
                return []

        monkeypatch.setattr("tools._get_vector_store", lambda: FakeVectorStore())
        result = execute_tool("search_knowledge_base", {"query": "学校"})
        assert "[EMPTY]" in result

    def test_returns_results(self, monkeypatch):
        class FakeVectorStore:
            def search_similar(self, query, n_results=5):
                return [{
                    "text": "河南工学院位于新乡市",
                    "metadata": {"source": "官网.md"},
                    "distance": 0.2,
                }]

        monkeypatch.setattr("tools._get_vector_store", lambda: FakeVectorStore())
        result = execute_tool("search_knowledge_base", {"query": "河南工学院"})
        assert "河南工学院位于新乡市" in result
        assert "官网.md" in result
