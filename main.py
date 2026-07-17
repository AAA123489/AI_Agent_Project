import src.logger
from fastapi.middleware.cors import CORSMiddleware
from src.config import ConfigManager
from src.llm_client import call_llm_client,call_llm_stream
from app.schemas import ChatRequest
from fastapi import FastAPI,Depends
import uvicorn
import fastapi_cdn_host
import logging
from dependencies import verify_api_key
import asyncio
from starlette.responses import StreamingResponse
import json


logger = logging.getLogger(__name__)

async def generate_stream(prompt, api_key, api_url):
    try:
        async for chunk in call_llm_stream(prompt, api_key, api_url):
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

            if (
                parsed.get("type") == "content_block_delta"
                and parsed.get("delta", {}).get("type") == "text_delta"
            ):
                text = parsed["delta"]["text"]
                yield f"data: {text}\n\n"
    except asyncio.CancelledError:
        logger.info("客户端断开")
    finally:
        logger.info("流结束了")
        
app = FastAPI(title = "AI-Chat-Api")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000","null"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

config = ConfigManager()

fastapi_cdn_host.patch_docs(app)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/chat")
async def chat(data:ChatRequest,api_key: str = Depends(verify_api_key)):
    return StreamingResponse(
        generate_stream(data.message,config.api_key,config.api_url),
        media_type="text/event-stream"
    )



if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)