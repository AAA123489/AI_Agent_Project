"""MCP Server 入口 —— 将 RAG 知识库能力暴露为 Agent 可调用的工具。

通过 stdio 与 Claude Code 通信，使用 MCP 协议（Model Context Protocol）。

启动方式：
    python -m mcp_server.server

    Claude Code 通过 .mcp.json 配置自动启动此进程，不需要手动运行。
"""
import asyncio
import logging
import sys
import warnings
from pathlib import Path

# 抑制第三方库的无关警告（chromadb 等）
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 确保项目根目录在导入路径中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from mcp_server.tools import (  # noqa: E402
    ingest_file,
    list_sources,
    search_knowledge_base,
)

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(levelname)s %(message)s",
    stream=sys.stderr,  # MCP 走 stderr，避免污染 stdout（stdio 协议）
)
logger = logging.getLogger("mcp_server")

# ── Server 实例 ──────────────────────────────────────────────
server = Server("rag-knowledge-base")

# ── 工具定义 ─────────────────────────────────────────────────


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """告诉 Agent：我有哪些工具可用"""
    return [
        Tool(
            name="search_knowledge_base",
            description=(
                "在 RAG 知识库中向量语义检索相关文档片段。"
                "当用户问「FastAPI 的特点」「Redis 的用法」「什么是 RAG」等需要查阅技术文档的问题时使用此工具。"
                "返回最相关的文档片段、来源文件路径和相似度分数。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本，例如 'FastAPI 有什么特点'、'RAG 的实现步骤'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的最相关文档数，默认 5，最大 20",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_file",
            description=(
                "将本地文档（TXT/Markdown/PDF）导入 RAG 知识库。"
                "支持幂等入库：同一文件重复导入时自动覆盖旧数据。"
                "使用场景：用户说「把这篇文档加到知识库」「导入 XX.pdf」时调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "本地文件路径，支持绝度路径或相对于项目根目录的路径。支持 .txt / .md / .pdf",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="list_sources",
            description=(
                "列出 RAG 知识库中已有的所有文档来源及文档块数量。"
                "使用场景：用户问「知识库里有哪些文档」「目前索引了多少资料」时调用。"
                "不需要任何参数。"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ── 工具调用路由 ─────────────────────────────────────────────


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Agent 调用工具时，路由到具体实现"""
    logger.info("工具调用: %s, 参数: %s", name, arguments)

    # 所有工具函数是同步的，扔到线程池执行（避免阻塞事件循环）。
    # 特别是 ingest_file 涉及文件 IO + Embedding 计算，不能阻塞 asyncio。
    try:
        if name == "search_knowledge_base":
            result = await asyncio.to_thread(
                search_knowledge_base,
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 5),
            )
        elif name == "ingest_file":
            result = await asyncio.to_thread(
                ingest_file,
                file_path=arguments.get("file_path", ""),
            )
        elif name == "list_sources":
            result = await asyncio.to_thread(list_sources)
        else:
            result = f"❌ 未知工具: {name}"

    except Exception as e:
        logger.exception("工具 %s 执行异常", name)
        result = f"❌ 工具执行失败 ({name}): {e}"

    return [TextContent(type="text", text=result)]


# ── 启动入口 ─────────────────────────────────────────────────


async def main() -> None:
    """启动 MCP Server，监听 stdio 等待 Agent 连接。"""
    logger.info("RAG 知识库 MCP Server 启动中...")
    async with stdio_server() as (read_stream, write_stream):
        logger.info("stdio 通道已建立，等待 Agent 请求")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
