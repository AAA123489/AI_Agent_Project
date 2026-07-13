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


logger = logging.getLogger(__name__)

app = FastAPI(title = "AI-Chat-Api")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000"],
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
    logger.info(f"收到请求: user_id={data.user_id}, message={data.message[:50]}...")
    api_url = config.api_url

    try:
        result = await call_llm_client(data.message, api_key, api_url)
        if result.get("error"):
            logger.error(f"LLM 调用失败: user_id={data.user_id}, error={result.get('message')}")
            return {"reply":"抱歉，服务器出错了"}
        reply_text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                reply_text += block.get("text", "")

        if not reply_text:
            logger.warning(f"LLM 未返回文本内容: user_id={data.user_id}")
            return {"reply": "抱歉，AI 未生成回复"}

        logger.info(f"请求成功: user_id={data.user_id}")
        return {"reply": reply_text}
    except Exception:
        logger.error(f"未预料的服务器错误: user_id={data.user_id}", exc_info=True)
        return {"reply": "服务器内部错误"}



if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)