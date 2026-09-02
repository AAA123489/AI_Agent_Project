"""测试 MCP 工具函数 —— search_knowledge_base / ingest_file / list_sources。

所有测试通过 mock VectorStore 实现，不依赖 ChromaDB 或 Embedding 模型。
"""
from unittest.mock import MagicMock, patch

import pytest


# ── 辅助函数 ──────────────────────────────────────────────────


def _make_mock_vector_store(**overrides):
    """构造一个带默认行为的 mock VectorStore。

    可以通过 overrides 覆盖任意方法：
        vs = _make_mock_vector_store(count=10, search_similar=[...])
    """
    vs = MagicMock()
    vs.count.return_value = overrides.get("count", 5)
    vs.count_by_source.return_value = overrides.get("count_by_source", 3)
    vs.search_similar.return_value = overrides.get(
        "search_similar",
        [
            {
                "text": "FastAPI 是一个现代 Python Web 框架",
                "metadata": {"source": "fastapi.md", "chunk_index": 0},
                "distance": 0.15,
            },
            {
                "text": "RAG 是检索增强生成的缩写",
                "metadata": {"source": "rag.md", "chunk_index": 1},
                "distance": 0.32,
            },
        ],
    )
    vs.collection = MagicMock()
    vs.collection.get.return_value = overrides.get(
        "collection_get",
        {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"source": "fastapi.md"},
                {"source": "fastapi.md"},
                {"source": "rag.md"},
            ],
        },
    )
    return vs


@pytest.fixture
def mock_vs():
    """注入 mock VectorStore，所有 MCP 工具共享。

    search_knowledge_base 已改为复用 app_backend._search_knowledge_base
    （混合检索），因此需要同时 patch app_backend 的 getter；
    ingest_file / list_sources 仍走 mcp_server.tools 的 getter。
    """
    vs = _make_mock_vector_store()
    with (
        patch("mcp_server.tools._get_vector_store", return_value=vs),
        patch("app_backend._get_vector_store", return_value=vs),
    ):
        yield vs


# ── search_knowledge_base ─────────────────────────────────────


class TestSearchKnowledgeBase:
    """语义检索工具"""

    def test_normal_query(self, mock_vs):
        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("FastAPI 有什么特点？")
        assert "FastAPI" in result
        assert "fastapi.md" in result
        assert "相似度" in result or "%" in result

    def test_top_k_respected(self, mock_vs):
        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("测试", top_k=1)
        # 确保只返回了 mock 中的 1 条（以 n_results=min(1,20)=1 调用）
        mock_vs.search_similar.assert_called_once()
        call_kwargs = mock_vs.search_similar.call_args
        assert call_kwargs[1]["n_results"] == 1

    def test_top_k_clamped_to_20(self, mock_vs):
        """top_k 超过 20 时应被限制为 20"""
        from mcp_server.tools import search_knowledge_base

        search_knowledge_base("测试", top_k=100)
        call_kwargs = mock_vs.search_similar.call_args
        assert call_kwargs[1]["n_results"] == 20

    def test_empty_query(self, mock_vs):
        """空查询返回提示信息，不应调用向量库"""
        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("")
        assert "为空" in result or "⚠️" in result

        result2 = search_knowledge_base("   ")
        assert "为空" in result2 or "⚠️" in result2

    def test_no_results(self, mock_vs):
        """检索无结果时返回友好提示"""
        mock_vs.search_similar.return_value = []

        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("不存在的文档")
        assert "未找到" in result or "📭" in result

    def test_api_error_handled(self, mock_vs):
        """向量库异常时不应崩溃，返回错误信息"""
        mock_vs.search_similar.side_effect = RuntimeError("Chroma 连接失败")

        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("测试")
        assert "失败" in result or "❌" in result


# ── ingest_file ───────────────────────────────────────────────


class TestIngestFile:
    """文档入库工具"""

    def test_empty_path(self, mock_vs):
        from mcp_server.tools import ingest_file

        result = ingest_file("")
        assert "提供" in result and "路径" in result

    def test_file_not_found(self, mock_vs):
        from mcp_server.tools import ingest_file

        result = ingest_file("/nonexistent/path/doc.txt")
        assert "不存在" in result

    def test_unsupported_format(self, mock_vs):
        from mcp_server.tools import ingest_file

        # 创建临时文件，但后缀不被支持
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            result = ingest_file(tmp_path)
            assert "不支持" in result
        finally:
            Path(tmp_path).unlink()

    def test_ingest_success(self, mock_vs):
        """正常入库流程（mock ingest_document 避免真实 Chroma 操作）"""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write("# Test\nHello World".encode())
            tmp_path = f.name

        with (
            patch("mcp_server.tools.ingest_document") as mock_ingest,
            patch("mcp_server.tools._get_vector_store", return_value=mock_vs),
        ):
            from mcp_server.tools import ingest_file

            mock_vs.count_by_source.return_value = 0
            mock_vs.count.side_effect = [5, 8]  # before → after

            result = ingest_file(tmp_path)
            assert "成功" in result
            mock_ingest.assert_called_once()

        Path(tmp_path).unlink()

    def test_ingest_replaces_existing(self, mock_vs):
        """已有旧数据时走覆盖模式"""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some content")
            tmp_path = f.name

        with (
            patch("mcp_server.tools.ingest_document") as mock_ingest,
            patch("mcp_server.tools._get_vector_store", return_value=mock_vs),
        ):
            from mcp_server.tools import ingest_file

            mock_vs.count_by_source.return_value = 10  # 有 10 条旧数据
            mock_vs.count.side_effect = [10, 15]

            result = ingest_file(tmp_path)
            assert "成功" in result
            assert mock_vs.delete_by_source.called

        Path(tmp_path).unlink()

    def test_ingest_exception_handled(self, mock_vs):
        """入库过程异常时不崩溃"""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"content")
            tmp_path = f.name

        with (
            patch("mcp_server.tools.ingest_document", side_effect=ValueError("解析失败")),
            patch("mcp_server.tools._get_vector_store", return_value=mock_vs),
        ):
            from mcp_server.tools import ingest_file

            mock_vs.count_by_source.return_value = 0
            result = ingest_file(tmp_path)
            assert "失败" in result

        Path(tmp_path).unlink()


# ── list_sources ──────────────────────────────────────────────


class TestListSources:
    """知识库来源统计工具"""

    def test_returns_source_list(self, mock_vs):
        from mcp_server.tools import list_sources

        result = list_sources()
        assert "fastapi.md" in result
        assert "rag.md" in result
        assert "文档块" in result or "个块" in result

    def test_empty_knowledge_base(self, mock_vs):
        mock_vs.count.return_value = 0

        from mcp_server.tools import list_sources

        result = list_sources()
        assert "为空" in result or "📭" in result

    def test_exception_handled(self, mock_vs):
        mock_vs.count.side_effect = RuntimeError("数据库挂了")

        from mcp_server.tools import list_sources

        result = list_sources()
        assert "失败" in result or "❌" in result


# ── 工具返回格式 ──────────────────────────────────────────────


class TestOutputFormat:
    """验证所有工具返回的是人类可读的字符串"""

    def test_search_returns_string(self, mock_vs):
        from mcp_server.tools import search_knowledge_base

        result = search_knowledge_base("测试")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_sources_returns_string(self, mock_vs):
        from mcp_server.tools import list_sources

        result = list_sources()
        assert isinstance(result, str)
        assert len(result) > 0
