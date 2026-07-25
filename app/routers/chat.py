"""聊天相关路由 —— SSE 流式对话 & 健康检查。"""
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.dependencies import _config as config
from app.dependencies import get_db, verify_api_key
from app.schemas.chat import ChatHistoryResponse, ChatRequest
from model import ChatHistory
from prompts import build_rag_prompt
from redis_client import get_recent_messages, save_message
from src.llm_client import call_llm_stream
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["chat"])


# ── 流式管道 ─────────────────────────────────────────────
async def generate_stream(
    prompt: str,
    api_key: str,
    api_url: str,
    model_name: str,
    user_id: str,
    db: Session,
    redis_client,
    context_docs,
):
    """把 LLM 的原始 SSE 块过滤为纯文本 data: 行，推给前端，并保存聊天历史。"""
    full_ai_reply = ""
    history = await get_recent_messages(redis_client, user_id, count=20)
    # 从检索结果中提取纯文本列表
    doc_texts = [doc["text"] for doc in context_docs] if context_docs else []
    context_prompt = build_rag_prompt(query=prompt, context_docs=doc_texts, history=history)

    # 先发送来源文档信息（前端展示用）
    if context_docs:
        sources_info = [{"text": doc["text"], "distance": doc["distance"]} for doc in context_docs]
        yield f"data: {json.dumps({'type': 'sources', 'docs': sources_info}, ensure_ascii=False)}\n\n"

    try:
        async for chunk in call_llm_stream(context_prompt, api_key, api_url, model_name):
            if not chunk.startswith("data:"):
                continue

            json_str = chunk[6:].strip()
            if not json_str or json_str == "[DONE]":
                continue

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("过滤出现问题: %s", chunk)
                continue

            if "error" in parsed:
                yield f"data: {json_str}\n\n"
                continue

            if (
                parsed.get("type") == "content_block_delta"
                and parsed.get("delta", {}).get("type") == "text_delta"
            ):
                text = parsed["delta"]["text"]
                full_ai_reply += text
                yield f"data: {text}\n\n"
    except asyncio.CancelledError:
        logger.info("客户端断开")
    except Exception:
        logger.exception("流式对话处理异常 | user_id=%s query=%s", user_id, prompt[:50])
        yield "data: [ERROR]服务异常\n\n"
    finally:
        # ── 保存到 SQLite（同步操作，扔到线程池）──
        if full_ai_reply:
            try:
                def _save_db():
                    new_record = ChatHistory(message=prompt, reply=full_ai_reply, user_id=user_id)
                    db.add(new_record)
                    db.commit()
                    db.refresh(new_record)
                    return new_record.id

                record_id = await asyncio.to_thread(_save_db)
                logger.info("聊天记录已保存，ID: %s", record_id)
            except Exception:
                try:
                    await asyncio.to_thread(db.rollback)
                except Exception:
                    pass
                logger.exception("保存聊天记录到 SQLite 失败 | user_id=%s", user_id)

        # ── 保存到 Redis ──
        if full_ai_reply:
            try:
                await save_message(redis_client, user_id, {"role": "user", "content": prompt})
                await save_message(redis_client, user_id, {"role": "assistant", "content": full_ai_reply})
            except Exception:
                logger.exception("保存聊天记录到 Redis 失败 | user_id=%s", user_id)

        logger.info("流结束了")


# ── 路由 ──────────────────────────────────────────────────
@router.get("/health")
def health_check():
    """健康检查端点，无需鉴权。"""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/history/{user_id}", response_model=list[ChatHistoryResponse])
def get_chat_history(
    user_id: str,
    db: Session = Depends(get_db),  # noqa: B008
    api_key: str = Depends(verify_api_key),
):
    """查询指定用户的对话历史，需要 API Key 鉴权。"""
    try:
        db_records = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
        return db_records
    except Exception:
        logger.exception("查询聊天历史失败 | user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="查询历史记录失败")


_vector_db: VectorStore | None = None
_vector_db_lock = asyncio.Lock()


async def get_vector_db() -> VectorStore:
    """延迟初始化向量库（异步安全），避免导入时就连接 Chroma。"""
    global _vector_db
    if _vector_db is None:
        async with _vector_db_lock:
            if _vector_db is None:
                _vector_db = await asyncio.to_thread(
                    VectorStore, collection_name="my_rag_collection"
                )
    return _vector_db


@router.post("/chat")
async def chat(
    data: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),  # noqa: B008
    api_key: str = Depends(verify_api_key),
):
    try:
        vector_db = await get_vector_db()
        # 多召回 + 来源多样性：确保不同文档的内容都有机会进入上下文
        raw_docs = await vector_db.search_similar_async(data.message, n_results=15)
        raw_docs = [doc for doc in raw_docs if doc["distance"] < vector_db.distance_threshold]

        # 按来源分组，每个来源最多取 3 条（避免单文档霸榜）
        source_groups: dict[str, list] = {}
        for doc in raw_docs:
            source = doc["metadata"].get("source", "unknown")
            source_groups.setdefault(source, []).append(doc)
        context_docs = []
        for source_docs in source_groups.values():
            context_docs.extend(source_docs[:3])
        # 按 distance 排序，取前 5 条作为最终上下文
        context_docs.sort(key=lambda d: d["distance"])
        context_docs = context_docs[:5]
    except Exception:
        logger.exception("向量检索失败 | user_id=%s query=%s", data.user_id, data.message[:50])
        raise HTTPException(status_code=500, detail="知识库检索失败，请稍后重试")

    redis_client = request.app.state.redis
    return StreamingResponse(
        generate_stream(
            data.message,
            config.api_key,
            config.api_url,
            config.model_name,
            data.user_id,
            db,
            redis_client,
            context_docs,
        ),
        media_type="text/event-stream",
    )
