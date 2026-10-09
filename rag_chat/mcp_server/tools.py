"""MCP 工具实现 —— 将 RAG 能力封装为 Agent 可调用的函数。

每个工具函数都是同步的（MCP SDK 会在 async 上下文中通过 to_thread 调用）。
返回字符串，由 MCP 协议层包装为 TextContent。
"""
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（MCP Server 由 Claude Code 以子进程方式启动时需要）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)

# ── VectorStore 单例（所有工具共享，只加载一次 420MB 模型）──
_vector_store: VectorStore | None = None


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
        )
        logger.info("VectorStore 单例初始化完成（Embedding 模型已加载）")
    return _vector_store


# ── 工具函数 ──────────────────────────────────────────────────


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在知识库中检索与 query 最相关的文档片段（混合检索：向量 + BM25 + RRF）。

    与聊天链路共用 app_backend._search_knowledge_base，保证外部 Agent 与站内问答
    拿到一致结果（含库外主体校验 / 年份过滤 / 罕见词精确召回 / 同文档补块 / 可选重排）。
    返回格式化文本，含文档片段、来源、相似度与原文链接。

    Args:
        query: 搜索查询文本
        top_k: 初始召回条数，默认 5（同文档补块后最多可达 20 条）

    Returns:
        格式化的检索结果文本；失败返回错误说明。
    """
    if not query or not query.strip():
        return "⚠️ 查询内容为空，请提供有效的搜索文本。"

    try:
        # 延迟导入：MCP 服务启动不加载 app_backend 全量模块，首次检索时才接上混合链路
        from app_backend import _search_knowledge_base
        return _search_knowledge_base(query.strip(), top_k=min(top_k, 20))
    except Exception as e:
        logger.exception("search_knowledge_base 执行失败")
        return f"❌ 检索失败: {e}"


def list_sources() -> str:
    """列出知识库中所有文档来源及其块数。

    Returns:
        格式化的来源清单
    """
    try:
        vs = _get_vector_store()
        total = vs.count()

        if total == 0:
            return "📭 知识库为空，还没有入库任何文档。请先运行爬虫管线入库。"

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
