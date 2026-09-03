"""
LangGraph Agent 工具定义 —— 项目三

定义 Agent 可调用的工具集合，包括：
- search_knowledge_base: 向量语义检索（对接项目一的 Chroma 知识库）
- list_knowledge_sources: 列出知识库已有文档
- calculate: 数学计算

每个工具返回 Anthropic Tool Use 格式的 tool_result。
"""

import ast
import contextvars
import logging
import operator as op
import os
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
# 主动工作记忆 —— Agent 自己决定存什么、查什么
#
# 存储：Redis（同步客户端。execute_tool 是同步上下文，不能直接 await 异步 Redis）
# 作用域：按会话（session_id）隔离，用 contextvars 随请求传递。
#         同一时间处理多个用户时，各请求的记忆互不串扰。
# 降级：Redis 连不上 → 记忆工具返回提示文本，Agent 照常干活，绝不抛异常。
# ═══════════════════════════════════════════════════════════════

_MEMORY_TTL = int(os.getenv("MEMORY_TTL", str(7 * 24 * 3600)))  # 记忆默认保留 7 天
_memory_client = None
_session_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_session", default="default")


def set_session(session_id: str) -> None:
    """设置当前会话 ID（工作记忆按会话隔离）。Web/CLI 每次对话前调用。"""
    _session_var.set(session_id or "default")


# ── 本次请求检索到的结构化知识库来源（按会话存）──
# 存 {source, url, similarity, text, ...}，供 Web 层以 sources 事件回传前端渲染「📎 查看原文」。
# 为什么不用 contextvar 回传：LangGraph 节点可能在子任务上下文里执行，工具里 set 的值写进
# 子上下文副本，回不到发起方。改按 session 记在模块级 dict：检索处能读到 _session_var
# （session 是从发起方随上下文传下来的），Web 结束后按同一 session 取值，天然按用户隔离。
_kb_sources_by_session: dict[str, list] = {}


def get_kb_sources(session_id: str) -> list:
    """读取某会话本次 KB 检索命中的来源（含原文 url）。"""
    return _kb_sources_by_session.get(session_id or "default", [])


def reset_kb_sources(session_id: str) -> None:
    """Web 每次对话前清空该会话的来源缓存（防止跨请求残留）。"""
    _kb_sources_by_session.pop(session_id or "default", None)


def _to_source_docs(results: list[dict]) -> list[dict]:
    """把 VectorStore 检索结果转成前端来源块需要的字段（对齐 rag_chat/static/chat.html）。"""
    docs: list[dict] = []
    seen: set[str] = set()
    for r in results:
        meta = r.get("metadata") or {}
        src = meta.get("source", "未知来源")
        if src in seen:
            continue
        seen.add(src)
        docs.append({
            "source": src,
            "url": meta.get("url", "") or "",
            "date": meta.get("publish_date", "") or "",
            "year": meta.get("year", "") or "",
            "category": meta.get("category", "") or "",
            "similarity": round(max(0, 1 - r.get("distance", 0)), 4),
            "text": (r.get("text") or "")[:200],
        })
    return docs


def _get_memory_client():
    """懒加载同步 Redis 客户端；不可用时返回 None（记忆功能降级不报错）。"""
    global _memory_client
    if _memory_client is None:
        try:
            import redis as sync_redis
            _memory_client = sync_redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True,
                socket_connect_timeout=3,
                protocol=2,  # 与项目一一致：兼容旧版 Redis（不支持 RESP3 的 HELLO）
            )
        except Exception:
            _memory_client = None
    return _memory_client


def _memory_key(key: str) -> str:
    """记忆的 Redis key：agent:memory:{会话}:{主题}"""
    return f"agent:memory:{_session_var.get()}:{key}"


def save_memory(key: str, content: str) -> str:
    """保存一条记忆。key 是记忆的主题（如'用户学校'），content 是记忆内容。"""
    if not key.strip():
        return "[WARN]  记忆主题（key）不能为空"
    if not content.strip():
        return "[WARN]  记忆内容不能为空"

    client = _get_memory_client()
    if client is None:
        return "[WARN]  记忆功能不可用（Redis 未连接），跳过保存"

    try:
        client.set(_memory_key(key), content, ex=_MEMORY_TTL)
        return f"[OK]  已保存记忆「{key}」"
    except Exception:
        return "[WARN]  记忆写入失败，跳过保存"


def search_memory(query: str) -> str:
    """检索与 query 相关的记忆（在 key 和内容里做子串匹配）。"""
    if not query.strip():
        return "[WARN]  查询内容不能为空"

    client = _get_memory_client()
    if client is None:
        return "[WARN]  记忆功能不可用（Redis 未连接）"

    try:
        prefix = f"agent:memory:{_session_var.get()}:"
        keys = client.keys(prefix + "*")
        q = query.strip().lower()
        hits = []
        for k in keys or []:
            short_key = k[len(prefix):]
            content = client.get(k) or ""
            if q in short_key.lower() or q in content.lower():
                hits.append(f"  • {short_key}: {content}")
        if not hits:
            return "[MEMORY]  没有找到相关记忆"
        return "[MEMORY]  找到相关记忆：\n" + "\n".join(hits)
    except Exception:
        return "[WARN]  记忆检索失败"


def clear_memory(key: str) -> str:
    """删除一条记忆；key 为 '*' 或 'all' 时清空当前会话全部记忆。"""
    client = _get_memory_client()
    if client is None:
        return "[WARN]  记忆功能不可用（Redis 未连接）"

    try:
        prefix = f"agent:memory:{_session_var.get()}:"
        if key in ("*", "all", "全部"):
            keys = client.keys(prefix + "*") or []
            if keys:
                client.delete(*keys)
            return f"[OK]  已清空 {len(keys)} 条记忆"
        deleted = client.delete(prefix + key)
        if deleted:
            return f"[OK]  已删除记忆「{key}」"
        return f"[WARN]  记忆「{key}」不存在"
    except Exception:
        return "[WARN]  记忆删除失败"


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
    {
        "name": "save_memory",
        "description": (
            "保存一条工作记忆。当用户主动透露个人偏好、重要信息，或你需要跨问题记住的内容时使用。"
            "key 是记忆主题（如'用户姓名'、'用户所在校区'），content 是记忆内容。"
            "之后可通过 search_memory 检索到这条记忆。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "记忆主题，如 '用户学校'"},
                "content": {"type": "string", "description": "记忆内容，如 '张同学在河南工学院'"},
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "检索之前保存的工作记忆。当用户问题可能涉及之前的对话信息或用户偏好时，先调用本工具看看。"
            "返回与 query 相关的记忆条目；没有则返回空结果。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词，如 '学校'、'偏好'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "clear_memory",
        "description": (
            "删除一条工作记忆；key 传 '*' 或 'all' 表示清空全部记忆。"
            "当用户明确要求忘记某信息时使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "要删除的记忆主题，或 '*' 清空全部"},
            },
            "required": ["key"],
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
        elif name == "save_memory":
            return save_memory(
                key=arguments.get("key", ""),
                content=arguments.get("content", ""),
            )
        elif name == "search_memory":
            return search_memory(arguments.get("query", ""))
        elif name == "clear_memory":
            return clear_memory(arguments.get("key", ""))
        else:
            return f"[X]  未知工具: {name}"
    except Exception as e:
        logger.exception("工具 %s 执行异常", name)
        return f"[X]  工具执行失败 ({name}): {e}"


def _p1_hybrid_search(query: str, top_k: int) -> str:
    """调用项目一 app_backend 的完整混合检索（延迟导入，避免启动就拖进项目一全家桶）。

    BM25+RRF 融合、人名/专有名词 $contains 精确召回（张尊舒这种人名纯向量捞不到）、
    同文档补块、年份过滤、库外拒答——全在项目一 app_backend 里按 100 题评测调优过。
    这里直接复用，保证「同一个问题，项目一和项目三检索结果一致」。
    """
    from app_backend import _search_knowledge_base as _p1_search
    return _p1_search(query, top_k=top_k)


def _parse_source_blocks(text: str) -> list[dict]:
    """从项目一检索返回的展示文本里提取结构化来源（供前端渲染『📎 查看原文』）。

    项目一每块格式固定：
        [i] 相似度: X% | 来源: 分类/文件名.txt | 日期: ... | ...
           原文链接: https://...
           片段: ...
    """
    import re
    docs: list[dict] = []
    seen: set[str] = set()
    for block in (text or "").split("\n\n"):
        src_m = re.search(r"来源:\s*([^|\n]+)", block)
        url_m = re.search(r"原文链接:\s*(\S+)", block)
        sim_m = re.search(r"相似度:\s*([\d.]+)%", block)
        if not src_m or not url_m:
            continue
        src = src_m.group(1).strip()
        if src in seen:
            continue
        seen.add(src)
        snippet = ""
        frag_m = re.search(r"片段:\s*(.*)", block, re.S)
        if frag_m:
            snippet = frag_m.group(1).split("...")[0].replace("\n", " ").strip()[:160]
        docs.append({
            "source": src,
            "url": url_m.group(1).strip(),
            "similarity": round(float(sim_m.group(1)) / 100, 4) if sim_m else 0.0,
            "text": snippet,
        })
    return docs


def _search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在知识库中检索相关文档片段。

    走项目一同源的混合检索（BM25+RRF+精确召回+补块）。项目一已按 100 题评测调优，
    纯向量对这些场景（尤其人名/专有名词）会漏召回，所以这里不再自己调 vs.search_similar。
    """
    q = (query or "").strip()
    if not q:
        return "[WARN]  查询内容为空，请提供有效的搜索文本。"

    try:
        text = _p1_hybrid_search(q, min(top_k, 10))
    except Exception as e:
        logger.warning("混合检索不可用（%s），回退纯向量", e)
        return _search_pure_vector(q, top_k)

    # 存结构化来源（含原文 url），供 Web 层以 sources 事件回传前端
    _kb_sources_by_session[_session_var.get()] = _parse_source_blocks(text)

    if not text or "知识库中未找到相关内容" in text:
        return "[EMPTY]  知识库中未找到与查询相关的内容。建议先使用 list_knowledge_sources 查看已有文档。"
    return f"[SEARCH]  「{q}」检索结果（与项目一同源混合检索）：\n\n{text}"


def _search_pure_vector(query: str, top_k: int = 5) -> str:
    """回退方案：纯向量检索。正常走不到，仅当项目一混合检索异常时兜底。"""
    vs = _get_vector_store()
    try:
        results = vs.search_similar(query, n_results=min(top_k, 10))
    except Exception as e:
        return f"[X]  知识库检索异常: {e}"
    _kb_sources_by_session[_session_var.get()] = _to_source_docs(results)
    if not results:
        return "[EMPTY]  知识库中未找到与查询相关的内容。建议先使用 list_knowledge_sources 查看已有文档。"
    lines = [f"[SEARCH]  「{query}」的检索结果（共 {len(results)} 条，纯向量回退）："]
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
        return "[EMPTY]  知识库为空，还没有入库任何文档。"

    all_data = vs.collection.get()
    source_counts: dict[str, int] = {}
    for meta in all_data.get("metadatas", []):
        if meta and "source" in meta:
            src = meta["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

    lines = [f"[LIB]  知识库共有 {total} 个文档块，来自 {len(source_counts)} 个文件："]
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  • {src}  ({count} 块)")
    return "\n".join(lines)


def _calculate(expression: str) -> str:
    """安全地计算数学表达式。"""
    if not expression or not expression.strip():
        return "[WARN]  请提供要计算的数学表达式。"

    try:
        result = _safe_eval(expression)
        return f"[OK]  计算结果: {expression} = {result}"
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"[X]  无法计算表达式 '{expression}': {e}"
