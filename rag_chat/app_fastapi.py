"""
app_fastapi.py — FastAPI SSE 后端
================================
校园百事通：对接 chat.html 纯前端，提供 SSE 流式对话 + 爬虫 + 上传接口。
复用 app_backend.py 的全部核心逻辑（LLM 调用 / 工具执行 / 检索 / 爬虫）。
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app_backend import (
    _get_vector_store,
    get_kb_stats, ingest_files, run_scraper_pipeline, get_active_params,
    AgentLoop, ThinkingEvent, ToolCallEvent, ToolResultEvent,
    TextEvent, DoneEvent, ErrorEvent,
)
# Redis（可选，用于服务端对话历史持久化）
try:
    from redis_client import get_redis_client, save_message, get_recent_messages
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

load_dotenv()

logger = logging.getLogger("fastapi_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ═══════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════

app = FastAPI(title="校园百事通 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = _PROJECT_ROOT / "static"


# ═══════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[list[dict]] = None


# ═══════════════════════════════════════════════════════
# SSE Agent 生成器（核心）
# ═══════════════════════════════════════════════════════

async def run_agent_sse(message: str, history: list[dict], api_key: str, user_id: str = ""):
    """
    Agent 循环的 SSE 版本（事件驱动）。
    AgentLoop 产出事件对象，这里转换为前端 SSE 帧
    （{"type": "thinking"|"sources"|"text"|"done"|"error", ...}）。
    user_id 用于 Redis 持久化历史（可选）。
    """
    agent = AgentLoop(api_key=api_key)
    last_answer = ""

    try:
        async for event in agent.run_stream(message, history):
            if isinstance(event, ThinkingEvent):
                yield {"type": "thinking", "step": event.step}
            elif isinstance(event, ToolCallEvent):
                step = f"🔧 调用工具: `{event.tool}({json.dumps(event.args, ensure_ascii=False)})`"
                yield {"type": "thinking", "step": step}
            elif isinstance(event, ToolResultEvent):
                if event.sources:
                    yield {"type": "sources", "docs": event.sources}
                    step = f"📋 检索到 {len(event.sources)} 条相关结果"
                else:
                    step = f"📋 返回: {event.output[:120]}"
                yield {"type": "thinking", "step": step}
            elif isinstance(event, TextEvent):
                last_answer = event.content
                # 分块输出答案（模拟流式）
                chunk_size = 30
                for i in range(0, len(event.content), chunk_size):
                    chunk = event.content[i:i + chunk_size]
                    yield {"type": "text", "content": chunk}
                    await asyncio.sleep(0.015)
            elif isinstance(event, DoneEvent):
                yield {"type": "done", "thinking": event.thinking, "sources": event.sources}
                # ── 持久化：Redis + TXT ──
                await _save_to_redis(user_id, message, last_answer)
                _log_to_txt(user_id, message, last_answer)
            elif isinstance(event, ErrorEvent):
                yield {"type": "error", "message": event.message}
    except Exception as e:
        logger.exception("Agent SSE 异常")
        yield {"type": "error", "message": str(e)}


# ── Redis 辅助 ──

async def _save_to_redis(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """保存一轮对话到 Redis（静默失败，不影响主流程）"""
    if not _REDIS_AVAILABLE or not user_id:
        return
    try:
        redis = await get_redis_client()
        await save_message(redis, user_id, {"role": "user", "content": user_msg})
        await save_message(redis, user_id, {"role": "assistant", "content": assistant_msg})
    except Exception:
        pass  # Redis 不可用时静默跳过


async def _load_redis_history(user_id: str, count: int = 20) -> list[dict]:
    """从 Redis 加载历史消息"""
    if not _REDIS_AVAILABLE or not user_id:
        return []
    try:
        redis = await get_redis_client()
        return await get_recent_messages(redis, user_id, count)
    except Exception:
        return []


# ── TXT 对话日志 ──
# 按 .env 的 EXPERIMENT_TAG 分文件（消融实验：每套参数一份日志，方便对比）。
# 未设置标签时用默认 chat_logs.txt；每次记录自动附带当前运行时参数摘要。

_DEFAULT_LOG_FILE = _PROJECT_ROOT / "chat_logs.txt"
_EXPERIMENT_TAG = os.getenv("EXPERIMENT_TAG", "").strip()

def _log_to_txt(user_id: str, question: str, answer: str) -> None:
    """将一轮对话追加写入 TXT 文件（静默失败，不影响对话）"""
    if not user_id:
        return
    try:
        # 有实验标签 → 分文件；否则默认一个文件
        log_file = _DEFAULT_LOG_FILE
        if _EXPERIMENT_TAG:
            log_file = _PROJECT_ROOT / f"chat_logs_{_EXPERIMENT_TAG}.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 70
        params = get_active_params()
        entry = (
            sep + "\n"
            + f"📅 {now}  👤 {user_id}\n"
            + f"⚙️ 参数: {params}" + (f" | 实验组: {_EXPERIMENT_TAG}" if _EXPERIMENT_TAG else "") + "\n"
            + f"❓ 问：{question}\n"
            + f"🤖 答：{answer}\n"
            + sep + "\n\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass  # 写文件失败不影响对话


# ═══════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════

@app.get("/")
async def root():
    """首页 → chat.html"""
    chat_html = _STATIC_DIR / "chat.html"
    if chat_html.exists():
        return FileResponse(chat_html)
    return HTMLResponse("<h1>chat.html not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """健康检查"""
    try:
        vs = _get_vector_store()
        chunk_count = vs.count()
        return {
            "status": "ok",
            "version": "2.0",
            "kb_chunks": chunk_count,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: str = Header(None, alias="X-API-Key")):
    """SSE 流式对话（支持多轮记忆：前端 history + Redis 服务端持久化）"""
    api_key = x_api_key or ""
    if not api_key:
        api_key = os.getenv("API_KEY", "")

    # ── 合并历史：Redis 服务端历史 + 前端会话历史 ──
    redis_history = await _load_redis_history(req.user_id)
    frontend_history = req.history or []

    # 去重合并：前端历史更近，优先保留；Redis 历史补充早期消息
    if redis_history and not frontend_history:
        # 前端没传历史（页面刷新后首次对话），用 Redis 历史
        merged_history = redis_history
        logger.info("用户 %s：加载 Redis 历史 %d 条", req.user_id, len(redis_history))
    elif frontend_history:
        merged_history = frontend_history
    else:
        merged_history = []

    async def event_stream():
        async for event in run_agent_sse(req.message, merged_history, api_key, req.user_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """文档上传入库"""
    try:
        # 保存临时文件
        tmp_dir = _PROJECT_ROOT / "scraped_docs" / "_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)

        # 调用入库逻辑
        result = await ingest_files([str(tmp_path)])
        ok = result.get("files_processed", 0)
        errs = result.get("errors", [])
        chunks = result.get("total_chunks", 0)

        if errs:
            return JSONResponse({
                "filename": file.filename,
                "chunks": chunks,
                "status": "partial",
                "errors": errs,
            }, status_code=207)

        return {
            "filename": file.filename,
            "chunks": chunks,
            "status": "ok",
        }
    except Exception as e:
        logger.exception("上传失败")
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.post("/scrape")
async def scrape(pages: int = 2):
    """触发爬虫（后台运行，返回任务确认信息）"""
    return {
        "status": "confirmed",
        "message": (
            f"将从学校官网爬取最新通知公告，"
            f"包含通知公告、学校新闻、教务处、各学院等分类，"
            f"每类最多 {pages} 页。"
            f"过程较慢（每页间隔 3~8 分钟），请耐心等待。"
        ),
        "pages": pages,
    }


@app.post("/scrape/start")
async def scrape_start(pages: int = 2):
    """实际执行爬虫（同步等待完成）"""
    try:
        result = await run_scraper_pipeline(max_pages=pages)
        return {
            "status": "ok",
            "total_articles": result.get("total_articles", 0),
            "total_chunks": result.get("total_chunks", 0),
            "errors": result.get("errors", []),
        }
    except Exception as e:
        logger.exception("爬虫失败")
        return JSONResponse({
            "status": "error",
            "detail": str(e),
        }, status_code=500)


@app.get("/kb-stats")
async def kb_stats():
    """知识库统计"""
    try:
        stats = get_kb_stats()
        return {
            "total_articles": stats.get("total_articles", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "category_counts": stats.get("category_counts", {}),
        }
    except Exception as e:
        logger.exception("KB 统计失败")
        return JSONResponse({"detail": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════
# 静态文件（兜底）
# ═══════════════════════════════════════════════════════

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
