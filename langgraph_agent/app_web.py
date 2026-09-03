"""
app_web.py — 项目三 LangGraph 多 Agent Web 服务
=================================================
复用项目一 rag_chat 的 FastAPI/SSE 外壳与前端 chat.html，
把「Router 意图路由 + 三子代理」通过 SSE（节点级流式）暴露成 Web 聊天接口。

复用清单（引擎无关，原样拿自 rag_chat）：
- 前端：rag_chat/static/chat.html（只依赖 /chat 和 /health 两个接口）
- 多轮记忆：Redis（rag_chat/redis_client.py，30 分钟过期）
- 鉴权：X-API-Key（读 rag_chat/.env，与项目一同 key）
- 输入校验：ChatRequest（防超长 / 防撑爆上下文）

本项目新增：
- run_langgraph_sse：用 LangGraph astream(subgraphs=True) 做节点级流式，
  把 Router 分派、子代理推理、工具调用逐帧推给前端（thinking 帧）。
- 主动工作记忆：Agent 通过 save_memory / search_memory / clear_memory 自行管理。

运行：python app_web.py  →  http://localhost:8001
"""

import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# ── 路径：agent / tools 在本目录；redis_client 在 rag_chat ──
_agent_dir = Path(__file__).resolve().parent
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))

_rag_chat_dir = _agent_dir.parent / "rag_chat"
if str(_rag_chat_dir) not in sys.path:
    sys.path.insert(0, str(_rag_chat_dir))

# ── 环境变量（优先 rag_chat/.env，与 agent.py 保持一致）──
_rag_chat_env = _rag_chat_dir / ".env"
if _rag_chat_env.exists():
    load_dotenv(_rag_chat_env)
else:
    load_dotenv()

from agent import build_agent, extract_final_answer  # noqa: E402
from tools import get_kb_sources, reset_kb_sources, set_session  # noqa: E402

# Redis（可选，用于服务端对话历史持久化）
try:
    from redis_client import get_redis_client, save_message, get_recent_messages
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

logger = logging.getLogger("langgraph_web")

# ═══════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="LangGraph 多 Agent 编排引擎", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = _rag_chat_dir / "static"

# 输入上限（照抄项目一，防超长撑爆 LLM 上下文）
MAX_MESSAGE_CHARS = 8000
MAX_HISTORY_TURNS = 20
MAX_HISTORY_MSG_CHARS = 2000


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[list[dict]] = None

    @field_validator("message")
    @classmethod
    def _check_message_length(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息不能为空")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"消息过长（上限 {MAX_MESSAGE_CHARS} 字符）")
        return v

    @field_validator("history")
    @classmethod
    def _check_history(cls, v: Optional[list[dict]]) -> Optional[list[dict]]:
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("history 必须是数组")
        if len(v) > MAX_HISTORY_TURNS * 2:
            raise ValueError(f"历史消息过多（上限 {MAX_HISTORY_TURNS} 轮）")
        for item in v:
            if not isinstance(item, dict):
                raise ValueError("历史消息必须是 {role, content} 对象")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("历史消息 content 缺失或为空")
            if len(content) > MAX_HISTORY_MSG_CHARS:
                raise ValueError(f"历史消息过长（单条上限 {MAX_HISTORY_MSG_CHARS} 字符）")
        return v


def _verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """服务端鉴权：X-API-Key 必须等于 rag_chat/.env 的 API_KEY（恒时比较）。"""
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="服务端未配置 API Key")
    provided = x_api_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return x_api_key


# ═══════════════════════════════════════════════════════════════
# Redis 对话历史（照抄项目一的降级模式：Redis 挂了不影响对话）
# ═══════════════════════════════════════════════════════════════

async def _save_to_redis(user_id: str, user_msg: str, assistant_msg: str) -> None:
    if not _REDIS_AVAILABLE or not user_id:
        return
    try:
        redis = await get_redis_client()
        await save_message(redis, user_id, {"role": "user", "content": user_msg})
        await save_message(redis, user_id, {"role": "assistant", "content": assistant_msg})
    except Exception:
        pass


async def _load_redis_history(user_id: str, count: int = 20) -> list[dict]:
    if not _REDIS_AVAILABLE or not user_id:
        return []
    try:
        redis = await get_redis_client()
        return await get_recent_messages(redis, user_id, count)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# SSE 生成器（节点级流式）：把 LangGraph 执行过程逐帧推给前端
#
# 用 agent.astream(stream_mode="updates", subgraphs=True) 逐节点拿更新：
#   - namespace 为空   → 根图节点（router / kb_agent / calc_agent / chat_agent）
#   - namespace 非空   → 子图内部节点（agent / tools）
# 把节点更新映射成 thinking 帧；最终答案用 extract_final_answer 切块流出。
# ═══════════════════════════════════════════════════════════════

_SUB_LABELS = {"kb_agent": "知识库", "calc_agent": "安全计算", "chat_agent": "闲聊"}
_SUB_AGENT_NODES = {"kb_agent", "calc_agent", "chat_agent"}
_ROUTE_LABELS = {"kb": "知识库", "calc": "安全计算", "chat": "闲聊"}


def _extract_tool_names(messages: list[dict]) -> list[str]:
    """从一条节点更新里提取 tool_use 块的工具名。"""
    names = []
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(block.get("name", "?"))
    return names


def _normalize_ns(ns0: str) -> str:
    """LangGraph 给编译子图节点名附加了 UUID（如 'calc_agent:a7cda846-...'），剥掉取前缀。"""
    return ns0.split(":")[0]


def _describe_step(namespace: tuple, node_name: str, update: dict) -> str | None:
    """把一次图节点更新映射成给用户看的 thinking 帧文本。"""
    if not namespace and node_name == "router":
        route = _ROUTE_LABELS.get(update.get("route", "?"), update.get("route", "?"))
        return f"🔀 Router 意图分类 → 分派给「{route}」子代理"
    if namespace and node_name == "agent":
        sub = _SUB_LABELS.get(_normalize_ns(namespace[0]), "")
        tool_names = _extract_tool_names(update.get("messages", []))
        if tool_names:
            return f"🔧 {sub}子代理调用工具: {', '.join(tool_names)}"
        return f"🧠 {sub}子代理生成回答…"
    if namespace and node_name == "tools":
        sub = _SUB_LABELS.get(_normalize_ns(namespace[0]), "")
        return f"📋 {sub}子代理工具执行完成"
    if not namespace and node_name in _SUB_AGENT_NODES:
        return f"✅ 「{_SUB_LABELS[node_name]}」子代理执行完毕"
    return None


async def run_langgraph_sse(message: str, history: list[dict], session_id: str):
    """SSE 帧生成器：节点级流式驱动 LangGraph 多 Agent。"""
    agent = build_agent()
    set_session(session_id)  # 主动工作记忆按会话隔离（contextvars 随请求传递）
    reset_kb_sources(session_id)  # 清空上次请求残留的 KB 来源

    initial_state = {
        "messages": list(history or []) + [{"role": "user", "content": message}],
        "round_count": 0,
        "route": "",
    }

    # 重建最终 messages：历史 + 用户消息 + 各节点产出的 messages 增量（operator.add 语义）
    accumulated_messages = list(initial_state["messages"])

    yield {"type": "thinking", "step": "🚀 多 Agent 引擎启动…"}

    try:
        async for namespace, updates in agent.astream(
            initial_state, stream_mode="updates", subgraphs=True
        ):
            for node_name, update in (updates or {}).items():
                if namespace and "messages" in update:
                    # 子图内部节点 → 累积 messages 增量
                    accumulated_messages.extend(update["messages"])
                step = _describe_step(namespace, node_name, update)
                if step:
                    yield {"type": "thinking", "step": step}

        answer = extract_final_answer({"messages": accumulated_messages})
        if not answer or answer == "（Agent 未生成回复）":
            yield {"type": "error", "message": "Agent 未生成有效回复，请重试。"}
            return

        # 答案切块流出，保留流式体验（前端逐块渲染）
        for i in range(0, len(answer), 30):
            yield {"type": "text", "content": answer[i:i + 30]}

        # KB 子代理命中过知识库 → 回传结构化来源（chat.html 据此渲染「📎 查看原文」链接）
        kb_docs = get_kb_sources(session_id)
        if kb_docs:
            yield {"type": "sources", "docs": kb_docs}

        yield {"type": "done", "thinking": [], "sources": kb_docs}

        # 持久化本轮对话到 Redis（多轮记忆）
        await _save_to_redis(session_id, message, answer)
    except Exception:
        logger.exception("LangGraph SSE 异常")
        yield {"type": "error", "message": "服务内部异常，请稍后重试。"}


# ═══════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    chat_html = _STATIC_DIR / "chat.html"
    if chat_html.exists():
        return FileResponse(chat_html)
    return HTMLResponse("<h1>chat.html not found</h1>", status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0", "engine": "langgraph-multi-agent"}


@app.post("/chat")
async def chat(req: ChatRequest, api_key: str = Depends(_verify_api_key)):
    """SSE 流式对话（支持多轮记忆：前端 history + Redis 服务端持久化）"""
    # 合并历史：Redis 服务端历史 + 前端会话历史（照抄项目一逻辑）
    redis_history = await _load_redis_history(req.user_id)
    frontend_history = req.history or []
    if redis_history and not frontend_history:
        merged_history = redis_history
    elif frontend_history:
        merged_history = frontend_history
    else:
        merged_history = []

    async def event_stream():
        async for event in run_langgraph_sse(req.message, merged_history, req.user_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# 静态文件（复用 rag_chat 的前端）
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEB_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
