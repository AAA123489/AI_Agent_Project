"""
app_fastapi.py — FastAPI SSE 后端
================================
校园百事通：对接 chat.html 纯前端，提供 SSE 流式对话 + 爬虫 + 上传接口。
复用 app_backend.py 的全部核心逻辑（LLM 调用 / 工具执行 / 检索 / 爬虫）。
"""

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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

# ── request_id 贯穿：contextvar 存当前请求 ID，Filter 注入所有日志 record ──
# 子模块（app_backend / src.* 等）的 logger 默认传播到 root，加一次 Filter 全局生效，
# 排查问题时能按 request_id 串起「入口 → 检索 → LLM → 答案」整条链路。
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(request_id)s] %(levelname)s: %(message)s",
)
for _h in logging.getLogger().handlers:
    _h.addFilter(_RequestIdFilter())

# ═══════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════

# ── 启动预热：提前加载 embedding 模型，消掉首问的 ~20s 冷启动 ──
# 首问慢的元凶是 embedding 模型懒加载（paraphrase-multilingual-MiniLM-L12-v2），
# 首次检索时才在 Chroma 里初始化。服务启动时先在线程池建一次 VectorStore，
# 把模型加载挪到启动阶段，用户第一问就不用干等。
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(_get_vector_store)
        logger.info("✅ embedding 模型预热完成（首问提速）")
    except Exception:
        logger.warning("embedding 预热失败，首问可能仍较慢", exc_info=True)
    yield


app = FastAPI(title="校园百事通 API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # 鉴权走 X-API-Key header，不需要 credentials 模式；
    # "*" + allow_credentials=True 是非法组合，改 False 避免跨站带 cookie 风险
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = _PROJECT_ROOT / "static"


# ═══════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════

# 输入上限：单条消息最长字符数（超长直接 422，防撑爆 DeepSeek 上下文）
MAX_MESSAGE_CHARS = 8000
# 历史上下文上限：防客户端传超长 history 撑爆 LLM 上下文（最多 20 轮、单条 2000 字）
MAX_HISTORY_TURNS = 20
MAX_HISTORY_MSG_CHARS = 2000
# 上传白名单扩展名 + 大小上限（20MB）
_ALLOWED_UPLOAD_EXT = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024
# 查询级缓存：相同问题直接复用上次答案，省 DeepSeek 调用 + 检索。.env 可配 off / TTL
QUERY_CACHE = os.getenv("QUERY_CACHE", "on").strip().lower() == "on"
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "3600"))
_CACHE_PREFIX = "rag:answer:v2:"


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
        """校验历史消息：防超长 history 撑爆 LLM 上下文（数量 + 单条长度双重上限）。"""
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
    """服务端鉴权：受保护接口的 X-API-Key 必须等于 .env 的 API_KEY。

    朋友试用填的就是这把 key；防止 ngrok 暴露时被白嫖服务端 DeepSeek key。
    / 与 /health 保持公开（探活用）。
    """
    expected = os.getenv("API_KEY", "")
    if not expected:
        logger.warning("API_KEY 未配置，拒绝所有受保护接口")
        raise HTTPException(status_code=500, detail="服务端未配置 API Key")
    # 恒定时间比较：普通 != 在首字符不同时立即返回，可被时序攻击探测 key 长度/前缀
    provided = x_api_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return x_api_key


# ── 查询级缓存辅助 ──

def _cache_key(message: str) -> str:
    """规范化消息 → 缓存 key。忽略 history（校园问答问题独立，命中率优先）。"""
    norm = " ".join(message.split())
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return f"{_CACHE_PREFIX}{digest}"


async def _cache_get(key: str) -> dict | None:
    """读缓存；Redis 不可用或异常 → 返回 None（降级为直接检索）。"""
    if not _REDIS_AVAILABLE:
        return None
    try:
        redis = await get_redis_client()
        raw = await redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("缓存读取失败，跳过缓存: %s", key)
        return None


async def _cache_set(key: str, answer: str, sources: list, thinking: list) -> None:
    """写缓存；失败静默（不影响对话主流程）。"""
    if not _REDIS_AVAILABLE:
        return
    try:
        redis = await get_redis_client()
        await redis.setex(key, QUERY_CACHE_TTL, json.dumps(
            {"answer": answer, "sources": sources, "thinking": thinking},
            ensure_ascii=False,
        ))
    except Exception:
        pass


async def _flush_query_cache() -> int:
    """知识库内容变化后清空查询缓存（旧答案可能已过时）。

    上传新文档 / 爬虫入库成功后调用；扫描 `rag:answer:v2:*` 前缀逐一删除。
    返回删除的 key 数；Redis 不可用或异常 → 0（降级，不影响主流程）。
    """
    if not _REDIS_AVAILABLE:
        return 0
    try:
        redis = await get_redis_client()
        deleted = 0
        async for key in redis.scan_iter(match=f"{_CACHE_PREFIX}*", count=100):
            await redis.delete(key)
            deleted += 1
        if deleted:
            logger.info("知识库已更新，清空查询缓存 %d 个 key", deleted)
        return deleted
    except Exception:
        logger.warning("清空查询缓存失败，跳过", exc_info=True)
        return 0


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

    # ── 查询级缓存：相同问题直接回放，省 LLM 调用 + 检索 ──
    if QUERY_CACHE:
        cache_key = _cache_key(message)
        cached = await _cache_get(cache_key)
        if cached:
            answer = cached.get("answer", "")
            cached_sources = cached.get("sources") or []
            cached_thinking = cached.get("thinking") or []
            logger.info("缓存命中 key=%s 答案%d字 来源%d条", cache_key, len(answer), len(cached_sources))
            if cached_sources:
                yield {"type": "sources", "docs": cached_sources}
            yield {"type": "thinking", "step": "⚡ 命中缓存（重复问题，直接复用上次回答）"}
            # 分块回放，保留流式体验
            for i in range(0, len(answer), 30):
                yield {"type": "text", "content": answer[i:i + 30]}
            yield {"type": "done", "thinking": cached_thinking, "sources": cached_sources}
            # 命中也是一次真实提问，照常持久化到历史与日志
            await _save_to_redis(user_id, message, answer)
            _log_to_txt(user_id, message, answer)
            return

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
                # 真流式：后端已按 LLM 输出增量逐个发 TextEvent，这里直接转发
                last_answer += event.content
                yield {"type": "text", "content": event.content}
            elif isinstance(event, DoneEvent):
                yield {"type": "done", "thinking": event.thinking, "sources": event.sources}
                # ── 持久化：Redis + TXT ──
                await _save_to_redis(user_id, message, last_answer)
                _log_to_txt(user_id, message, last_answer)
                # ── 查询级缓存：写回完整答案 + 来源 + 思考过程 ──
                if QUERY_CACHE:
                    await _cache_set(_cache_key(message), last_answer, event.sources or [], event.thinking or [])
            elif isinstance(event, ErrorEvent):
                yield {"type": "error", "message": event.message}
    except Exception:
        logger.exception("Agent SSE 异常")
        # 不把内部异常原文透传给客户端（可能泄露路径/堆栈），统一友好文案
        yield {"type": "error", "message": "服务内部异常，请稍后重试。"}


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
async def chat(req: ChatRequest, api_key: str = Depends(_verify_api_key)):
    """SSE 流式对话（支持多轮记忆：前端 history + Redis 服务端持久化）"""
    _request_id_var.set(uuid.uuid4().hex[:12])  # request_id：贯穿本次请求全链路日志

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
async def upload(file: UploadFile = File(...), _: str = Depends(_verify_api_key)):
    """文档上传入库（受保护：需服务端 API Key）"""
    try:
        # 扩展名白名单 + 防路径穿越（只取文件名部分，忽略目录前缀）
        filename = Path(file.filename or "").name
        ext = Path(filename).suffix.lower()
        if not filename or ext not in _ALLOWED_UPLOAD_EXT:
            return JSONResponse(
                {"detail": f"不支持的文件类型「{ext or '未知'}」，仅支持 .txt/.md/.pdf/.docx"},
                status_code=400,
            )

        content = await file.read()
        if len(content) == 0:
            return JSONResponse({"detail": "文件内容为空"}, status_code=400)
        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                {"detail": f"文件过大（上限 {MAX_UPLOAD_SIZE // (1024 * 1024)}MB）"},
                status_code=413,
            )

        # 保存临时文件
        tmp_dir = _PROJECT_ROOT / "scraped_docs" / "_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(content)

        try:
            # 调用入库逻辑
            result = await ingest_files([str(tmp_path)])
            errs = result.get("errors", [])
            chunks = result.get("total_chunks", 0)

            # 知识库已变化（部分成功也变了），旧查询缓存可能过时，清空
            await _flush_query_cache()

            if errs:
                return JSONResponse({
                    "filename": filename,
                    "chunks": chunks,
                    "status": "partial",
                    "errors": errs,
                }, status_code=207)

            return {
                "filename": filename,
                "chunks": chunks,
                "status": "ok",
            }
        finally:
            # 临时文件入库后立即删除，防 scraped_docs/_uploads 堆积
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("清理临时文件失败: %s", tmp_path, exc_info=True)
    except Exception:
        logger.exception("上传失败")
        return JSONResponse({"detail": "上传处理失败，请检查文件格式后重试"}, status_code=500)


@app.post("/scrape")
async def scrape(pages: int = 2, _: str = Depends(_verify_api_key)):
    """触发爬虫（后台运行，返回任务确认信息；受保护）"""
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


# ── 爬虫并发控制 ──
# 原实现：请求里 await 整个爬取（全站 1~2 小时），连接一直挂着、客户端一断开爬虫就中断，
# 且无锁，两个请求可同时开爬。改后台任务 + 运行标记：请求立即返回，断开不影响爬取。
_scrape_lock = asyncio.Lock()
_scrape_running = False


async def _run_scrape_and_flush(pages: int) -> None:
    """后台执行爬虫 + 清空旧缓存；结束后复位运行标记。"""
    global _scrape_running
    try:
        async with _scrape_lock:
            result = await run_scraper_pipeline(max_pages=pages)
        # 爬虫入库后知识库已变化，旧查询缓存可能过时，清空
        await _flush_query_cache()
        logger.info(
            "爬虫后台任务完成: 文章 %d 篇, 分块 %d, 错误 %d 条",
            result.get("total_articles", 0),
            result.get("total_chunks", 0),
            len(result.get("errors", [])),
        )
    except Exception:
        logger.exception("后台爬虫任务失败")
    finally:
        _scrape_running = False


@app.post("/scrape/start")
async def scrape_start(background_tasks: BackgroundTasks, pages: int = 2, _: str = Depends(_verify_api_key)):
    """触发全站爬虫（后台运行、立即返回；防重复触发；受保护）"""
    global _scrape_running
    # 检查 + 置位之间无 await，单事件循环内原子，不会并发双开
    if _scrape_running:
        return JSONResponse(
            {"detail": "爬虫已在运行中，请勿重复触发"},
            status_code=409,
        )
    _scrape_running = True
    background_tasks.add_task(_run_scrape_and_flush, pages)
    return {
        "status": "started",
        "message": "爬虫已在后台启动，完成后会自动清空旧缓存。可稍后通过 /kb-stats 查看知识库变化。",
        "pages": pages,
    }


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
    except Exception:
        logger.exception("KB 统计失败")
        return JSONResponse({"detail": "知识库统计失败，请稍后重试"}, status_code=500)


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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
