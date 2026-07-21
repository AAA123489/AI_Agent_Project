"""聊天相关路由 —— SSE 流式对话 & 健康检查。"""
import asyncio
import json
import logging
from typing import List
from redis_client import save_message, get_recent_messages
from fastapi import APIRouter, Depends,Request
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from src.config import ConfigManager
from src.llm_client import call_llm_stream
from app.dependencies import get_db, verify_api_key
from app.schemas.chat import ChatHistoryResponse, ChatRequest
from model import ChatHistory


logger = logging.getLogger(__name__)

config = ConfigManager()

router = APIRouter(prefix="", tags=["chat"])



# ── 流式管道 ─────────────────────────────────────────────
async def generate_stream(prompt: str, api_key: str, api_url: str, model_name: str, user_id: str, db: Session,redis_client):
    """把 LLM 的原始 SSE 块过滤为纯文本 data: 行，推给前端，并保存聊天历史。"""
    full_ai_reply = ""
    history = await get_recent_messages(redis_client, user_id, count=20)
    context_prompt = prompt
    if history:
        context_lines = ["以下是之前的对话："]
        for msg in history:
            role_label = "用户" if msg.get("role") == "user" else "助手"
            context_lines.append(f"{role_label}: {msg.get('content', '')}")
        context_lines.append("---")
        context_lines.append(f"当前问题：{prompt}")
        context_prompt = "\n".join(context_lines)

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
    except Exception as exc:
        logger.exception("流式对话处理异常: %s", exc)
        yield "data: [ERROR]服务异常\n\n"
    finally:
        if full_ai_reply:
            new_record = ChatHistory(message=prompt, reply=full_ai_reply, user_id=user_id)
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            logger.info("聊天记录已保存，ID: %s", new_record.id)
            logger.info("流结束了")

        if full_ai_reply:
            await save_message(redis_client, user_id, {"role": "user", "content": prompt})
            await save_message(redis_client, user_id, {"role": "assistant", "content": full_ai_reply})


# ── 路由 ──────────────────────────────────────────────────
@router.get("/history/{user_id}", response_model=List[ChatHistoryResponse])
def get_chat_history(user_id: str, db: Session = Depends(get_db)):


    db_records = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
    
    return db_records


@router.post("/chat")
async def chat(
    data: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    redis_client = request.app.state.redis
    return StreamingResponse(
        generate_stream(data.message, config.api_key, config.api_url, config.model_name, data.user_id, db,redis_client),
        media_type="text/event-stream",
    )




