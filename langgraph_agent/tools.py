"""
LangGraph Agent 工具定义 —— 项目三

定义 Agent 可调用的工具集合，包括：
- search_knowledge_base: 向量语义检索（对接项目一的 Chroma 知识库）
- list_knowledge_sources: 列出知识库已有文档
- calculate: 数学计算

每个工具返回 Anthropic Tool Use 格式的 tool_result。
"""

import ast
import logging
import operator as op
import sys
from pathlib import Path

# ── 路径：让 langgraph_agent 能导入 rag_chat 的模块 ──
_rag_chat_path = Path(__file__).resolve().parent.parent / "rag_chat"
if str(_rag_chat_path) not in sys.path:
    sys.path.insert(0, str(_rag_chat_path))

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── VectorStore 单例（延迟加载，避免导入时就连接 Chroma）──
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    """获取全局 VectorStore 单例，首次调用时初始化（加载 Embedding 模型）。

    关键：db_path 必须显式指向 rag_chat/chroma_db，因为 Agent 是从
    langgraph_agent/ 目录运行的，默认的 ./chroma_db 会指向错误位置。
    """
    global _vector_store
    if _vector_store is None:
        # 计算 rag_chat/chroma_db 的绝对路径（不受工作目录影响）
        chroma_path = str(_rag_chat_path / "chroma_db")
        _vector_store = VectorStore(
            db_path=chroma_path,
            collection_name="my_rag_collection",
            distance_threshold=0.85,
        )
        logger.info("VectorStore 单例初始化完成 (db_path=%s)", chroma_path)
    return _vector_store


# ═══════════════════════════════════════════════════════════════
# Anthropic Tool Use 格式的工具定义（传给 LLM 的 input_schema）
# ═══════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "在 RAG 知识库中进行向量语义检索。"
            "当用户询问技术问题、需要查阅文档、或想了解某个知识点时使用。"
            "返回最相关的文档片段、来源文件路径和相似度分数。"
            "适用场景：'FastAPI 有什么特点？'、'Redis 的用法'、'什么是 RAG？'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本，使用与用户问题最相关的关键词",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最相关文档数，默认 5，最大 10",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_knowledge_sources",
        "description": (
            "列出 RAG 知识库中已有的所有文档来源及每个文档的块数。"
            "当用户问'知识库里有哪些文档'、'目前有哪些资料'时使用。"
            "不需要任何参数。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "执行数学表达式计算。支持四则运算、幂运算、括号等。"
            "适用场景：'123 + 456 等于多少？'、'2 的 10 次方是多少？'"
            "表达式必须是纯数学表达式，不接受变量或函数调用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '(100 + 200) * 3 / 5'、'2 ** 10'",
                },
            },
            "required": ["expression"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# 工具执行函数
# ═══════════════════════════════════════════════════════════════

# 安全的数学运算符白名单
_SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(expression: str) -> float | int:
    """安全地评估数学表达式，拒绝任意代码执行。"""

    def _eval_node(node):
        """递归解析 AST 节点，只允许白名单中的运算。"""
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return _SAFE_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            return _SAFE_OPS[op_type](_eval_node(node.operand))
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree)


def execute_tool(name: str, arguments: dict) -> str:
    """根据工具名执行对应的工具函数，返回字符串结果。

    这是 Agent 的"手"——LLM 决定调用哪个工具、传什么参数，
    这个函数负责执行并返回结果。
    """
    logger.info("工具执行: %s(%s)", name, arguments)

    try:
        if name == "search_knowledge_base":
            return _search_knowledge_base(
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 5),
            )
        elif name == "list_knowledge_sources":
            return _list_sources()
        elif name == "calculate":
            return _calculate(arguments.get("expression", ""))
        else:
            return f"❌ 未知工具: {name}"
    except Exception as e:
        logger.exception("工具 %s 执行异常", name)
        return f"❌ 工具执行失败 ({name}): {e}"


def _search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在知识库中向量检索相关文档片段。"""
    if not query or not query.strip():
        return "⚠️ 查询内容为空，请提供有效的搜索文本。"

    vs = _get_vector_store()
    results = vs.search_similar(query.strip(), n_results=min(top_k, 10))

    if not results:
        return "📭 知识库中未找到与查询相关的内容。建议先使用 list_knowledge_sources 查看已有文档。"

    lines = [f"🔍 「{query}」的检索结果（共 {len(results)} 条）："]
    for i, r in enumerate(results, 1):
        text_preview = r["text"][:300].replace("\n", " ")
        source = r.get("metadata", {}).get("source", "未知来源")
        distance = r.get("distance", 0)
        similarity = max(0, 1 - distance)
        lines.append(
            f"---\n"
            f"[{i}] 来源: {source}  |  相似度: {similarity:.1%}\n"
            f"内容: {text_preview}..."
        )
    return "\n".join(lines)


def _list_sources() -> str:
    """列出知识库中所有文档来源。"""
    vs = _get_vector_store()
    total = vs.count()

    if total == 0:
        return "📭 知识库为空，还没有入库任何文档。"

    all_data = vs.collection.get()
    source_counts: dict[str, int] = {}
    for meta in all_data.get("metadatas", []):
        if meta and "source" in meta:
            src = meta["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

    lines = [f"📚 知识库共有 {total} 个文档块，来自 {len(source_counts)} 个文件："]
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  • {src}  ({count} 块)")
    return "\n".join(lines)


def _calculate(expression: str) -> str:
    """安全地计算数学表达式。"""
    if not expression or not expression.strip():
        return "⚠️ 请提供要计算的数学表达式。"

    try:
        result = _safe_eval(expression)
        return f"✅ 计算结果: {expression} = {result}"
    except (SyntaxError, ValueError, TypeError) as e:
        return f"❌ 无法计算表达式 '{expression}': {e}"
