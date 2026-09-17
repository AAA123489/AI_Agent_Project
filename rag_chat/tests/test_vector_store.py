"""测试 Chroma 向量库封装 —— VectorStore。"""
import shutil
import tempfile

import pytest

from src.vector_store import VectorStore


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def temp_db():
    """每次测试使用独立的临时 Chroma 目录，避免相互污染。

    Windows 上 ChromaDB 退出时 SQLite 文件锁可能未立即释放，
    导致 TemporaryDirectory 无法正常清理。这里用 shutil.rmtree
    手动兜底，忽略 PermissionError。
    """
    tmpdir = tempfile.mkdtemp()
    try:
        store = VectorStore(db_path=tmpdir, collection_name="test_collection")
        yield store
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def populated_db(temp_db):
    """预填充 3 条文档的向量库。"""
    temp_db.save_documents(
        documents=["FastAPI 是一个现代 Python Web 框架", "ChromaDB 是向量数据库", "RAG 是检索增强生成"],
        ids=["doc-1", "doc-2", "doc-3"],
        metadatas=[
            {"source": "fastapi.md", "chunk_index": 0},
            {"source": "chroma.md", "chunk_index": 0},
            {"source": "rag.md", "chunk_index": 0},
        ],
    )
    return temp_db


# ── 初始化 ────────────────────────────────────────────────────


class TestInit:
    """构造与初始化"""

    def test_default_values(self, temp_db):
        """默认参数应正确设置"""
        assert temp_db.collection.name == "test_collection"

    def test_collection_is_persistent(self, temp_db):
        """collection 应在创建后可立即查询"""
        assert temp_db.count() == 0

    def test_collection_reuse(self, temp_db):
        """重复 upsert 不抛异常（Chroma 自动处理）"""
        temp_db.save_document("第一条文档")
        temp_db.save_document("第一条文档")
        # 不应抛异常

    def test_custom_collection_name(self):
        """collection_name 可自定义"""
        tmpdir = tempfile.mkdtemp()
        try:
            store = VectorStore(db_path=tmpdir, collection_name="custom_collection")
            assert store.collection.name == "custom_collection"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── 增删查 ────────────────────────────────────────────────────


class TestSaveAndCount:
    """文档保存与计数"""

    def test_save_single_document(self, temp_db):
        temp_db.save_document("测试文档", metadata={"source": "test.txt"})
        assert temp_db.count() == 1

    def test_save_multiple_documents(self, temp_db):
        temp_db.save_documents(
            documents=["文档A", "文档B", "文档C"],
            ids=["a", "b", "c"],
        )
        assert temp_db.count() == 3

    def test_count_by_source(self, populated_db):
        assert populated_db.count_by_source("fastapi.md") == 1
        assert populated_db.count_by_source("chroma.md") == 1
        assert populated_db.count_by_source("rag.md") == 1
        assert populated_db.count_by_source("nonexistent.md") == 0

    def test_delete_by_source(self, populated_db):
        assert populated_db.count() == 3
        populated_db.delete_by_source("fastapi.md")
        assert populated_db.count() == 2
        assert populated_db.count_by_source("fastapi.md") == 0
        # 其他文档不受影响
        assert populated_db.count_by_source("chroma.md") == 1
        assert populated_db.count_by_source("rag.md") == 1

    def test_delete_by_source_no_match(self, populated_db):
        """删除不存在的 source 不应影响现有数据"""
        count_before = populated_db.count()
        populated_db.delete_by_source("ghost.md")
        assert populated_db.count() == count_before


class TestClearAll:
    """清空 collection"""

    def test_clear_empty(self, temp_db):
        """空 collection 清空不抛异常"""
        temp_db.clear_all()
        assert temp_db.count() == 0

    def test_clear_populated(self, populated_db):
        assert populated_db.count() == 3
        populated_db.clear_all()
        assert populated_db.count() == 0


# ── 检索 ──────────────────────────────────────────────────────


class TestSearchSimilar:
    """相似度检索"""

    def test_returns_correct_structure(self, populated_db):
        """返回结果应包含 text / metadata / distance 三个字段"""
        results = populated_db.search_similar("Python 框架", n_results=2)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "distance" in r
            assert isinstance(r["text"], str)
            assert isinstance(r["distance"], float)

    def test_returns_fewer_than_requested(self, populated_db):
        """请求数量超过库内总数时，返回实际条数"""
        results = populated_db.search_similar("某个查询", n_results=50)
        assert len(results) <= populated_db.count()
        assert len(results) >= 1

    def test_semantic_ordering(self, populated_db):
        """语义相关的文档应排在前面"""
        results = populated_db.search_similar("FastAPI", n_results=3)
        # 第一条应和 FastAPI 最相关
        assert "FastAPI" in results[0]["text"]

    def test_n_results_default(self, populated_db):
        """不传 n_results 时使用默认值 3"""
        results = populated_db.search_similar("检索")
        assert len(results) <= 3

    def test_n_results_one(self, populated_db):
        results = populated_db.search_similar("向量", n_results=1)
        assert len(results) == 1

    def test_empty_collection(self, temp_db):
        """空库检索不抛异常，返回空列表"""
        results = temp_db.search_similar("任意问题")
        assert results == []

    def test_query_with_special_characters(self, populated_db):
        """包含特殊字符的查询不应崩溃"""
        results = populated_db.search_similar("Python / RAG —— 框架？")
        assert isinstance(results, list)

    def test_query_chinese(self, populated_db):
        """中文查询"""
        results = populated_db.search_similar("什么是检索增强生成")
        assert len(results) >= 1

    def test_query_english(self, populated_db):
        """英文查询"""
        results = populated_db.search_similar("vector database")
        assert len(results) >= 1


# search_similar_async 是 asyncio.to_thread 的薄包装，核心逻辑由同步测试全覆盖。
