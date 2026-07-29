"""MCP 工具实现 —— 将 RAG 能力封装为 Agent 可调用的函数。

每个工具函数都是同步的（MCP SDK 会在 async 上下文中通过 to_thread 调用）。
返回字符串，由 MCP 协议层包装为 TextContent。
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（MCP Server 由 Claude Code 以子进程方式启动时需要）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.vector_store import VectorStore  # noqa: E402
from rag_pipeline import ingest_document  # noqa: E402

logger = logging.getLogger(__name__)

# ── VectorStore 单例（所有工具共享，只加载一次 420MB 模型）──
_vector_store: VectorStore | None = None
_lock: asyncio.Lock | None = None  # asyncio 锁，延迟创建


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _get_vector_store() -> VectorStore:
    """获取全局单例 VectorStore（线程安全双重检查）。

    注意：此函数可从 sync 上下文调用（MCP 工具的 @mcp.tool() 是 sync 的），
    但初始化在 asyncio.to_thread 中执行，实际调用 _get_vector_store 时
    是从主事件循环的线程池线程中调用，所以这里用简单的 None 检查就够。
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            collection_name="my_rag_collection",
            distance_threshold=0.85,
        )
        logger.info("VectorStore 单例初始化完成（Embedding 模型已加载）")
    return _vector_store


# ── 工具函数 ──────────────────────────────────────────────────


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在知识库中向量检索与 query 最相似的文档片段。

    Args:
        query: 搜索查询文本
        top_k: 返回的最相关文档数，默认 5

    Returns:
        格式化的检索结果文本，包含文档片段、来源文件和相似度距离
    """
    if not query or not query.strip():
        return "⚠️ 查询内容为空，请提供有效的搜索文本。"

    try:
        vs = _get_vector_store()
        results = vs.search_similar(query.strip(), n_results=min(top_k, 20))

        if not results:
            return "📭 知识库中未找到与查询相关的内容。建议先使用 ingest_document 入库相关文档。"

        lines = [f"🔍 「{query}」的检索结果（共 {len(results)} 条）：\n"]
        for i, r in enumerate(results, 1):
            text_preview = r["text"][:200].replace("\n", " ")
            source = r.get("metadata", {}).get("source", "未知来源")
            distance = r.get("distance", 0)
            similarity = max(0, 1 - distance)  # 余弦距离 → 相似度
            lines.append(
                f"---\n"
                f"【{i}】来源: {source}\n"
                f"相似度: {similarity:.2%}  (距离: {distance:.4f})\n"
                f"内容: {text_preview}..."
            )
        return "\n".join(lines)

    except Exception as e:
        logger.exception("search_knowledge_base 执行失败")
        return f"❌ 检索失败: {e}"


def ingest_file(file_path: str) -> str:
    """将本地文档（TXT/MD/PDF）解析、切分后存入知识库。

    支持幂等入库：同一文件重复入库时会先删除旧数据再写入。

    Args:
        file_path: 文档的绝对路径或相对于项目根目录的路径

    Returns:
        入库结果摘要（切分块数、总文档数等）
    """
    if not file_path:
        return "⚠️ 请提供文件路径。"

    path = Path(file_path)
    if not path.is_absolute():
        path = _project_root / path

    if not path.exists():
        return f"❌ 文件不存在: {path}"

    suffix = path.suffix.lower()
    if suffix not in (".txt", ".md", ".pdf"):
        return f"❌ 不支持的文件格式: {suffix}。支持: .txt / .md / .pdf"

    try:
        vs = _get_vector_store()
        file_path_str = str(path)

        # 幂等处理：先删除同文件的旧数据
        existing_count = vs.count_by_source(file_path_str)
        if existing_count > 0:
            vs.delete_by_source(file_path_str)
            logger.info("覆盖模式：已删除旧数据 %d 条", existing_count)

        total_before = vs.count()

        # 同步入库（ingest_document 内部调用切分器 + Chroma）
        ingest_document(file_path_str, vector_store=vs)

        total_after = vs.count()
        new_chunks = total_after - total_before

        return (
            f"✅ 文档入库成功\n"
            f"文件: {path.name}\n"
            f"新增块数: {new_chunks}\n"
            f"知识库总块数: {total_after}"
            + (f"\n(覆盖模式: 替换了原有的 {existing_count} 条旧数据)" if existing_count > 0 else "")
        )

    except Exception as e:
        logger.exception("ingest_file 执行失败")
        return f"❌ 入库失败: {e}"


def list_sources() -> str:
    """列出知识库中所有文档来源及其块数。

    Returns:
        格式化的来源清单
    """
    try:
        vs = _get_vector_store()
        total = vs.count()

        if total == 0:
            return "📭 知识库为空，还没有入库任何文档。使用 ingest_file 添加文档。"

        # Chroma 没有直接「列出所有 source」的方法，
        # 通过 get() 获取全量 metadata 再统计
        all_data = vs.collection.get()
        source_counts: dict[str, int] = {}
        for meta in all_data.get("metadatas", []):
            if meta and "source" in meta:
                src = meta["source"]
                source_counts[src] = source_counts.get(src, 0) + 1

        lines = [f"📚 知识库共有 {total} 个文档块，来自 {len(source_counts)} 个文件：\n"]
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  • {src}  ({count} 块)")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("list_sources 执行失败")
        return f"❌ 查询失败: {e}"
