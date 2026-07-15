import src.logger
from fastapi.middleware.cors import CORSMiddleware
from src.config import ConfigManager
from src.llm_client import call_llm_client
from app.schemas import ChatRequest
from fastapi import FastAPI,Depends
import uvicorn
import fastapi_cdn_host
import logging
from dependencies import verify_api_key
import asyncio
from starlette.responses import StreamingResponse


logger = logging.getLogger(__name__)

async def generate_stream(prompt):

    '''
    异步生成器，模拟大模型流式输出
    '''
    
    response_text = f"你好，我是AI助手！关于你提到的{prompt}我认为很有趣"
    try:
        await asyncio.sleep(0.1)
        for chat in  response_text:
            yield f"data: {chat}\n\n"
            await asyncio.sleep(0.1)
        yield "data: [DONE]\n\n"
        logger.info("✅ 流式生成完毕！")
    except asyncio.CancelledError:
        logger.info(f"客户端断开")
    finally:
        logger.info(f"流结束了")
        
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
        generate_stream(data.message),
        media_type="text/event-stream"
    )



if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)