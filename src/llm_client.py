import asyncio  # 引入 asyncio 库，用于编写异步协程
import aiohttp  # 引入 aiohttp 库，用于异步 HTTP 请求
import logging
from typing import AsyncGenerator
logger = logging.getLogger(__name__)


async def call_llm_stream(prompt: str, api_key: str, api_url: str) -> AsyncGenerator[str, None]:
    URL = api_url  # 使用传入的 api_url 参数
    headers = {
        "Authorization": f"Bearer {api_key}",  # 使用传入的 api_key 参数
        "Content-Type": "application/json"  # 设置请求的内容类型为 JSON
    }
    payload = {
        "max_tokens": 1000,  # 设置最大 token 数量为 100
        "model":"deepseek-v4-pro",  # 从环境变量中获取模型名称
        "messages":[{"role":"user","content":prompt}] , # 将 prompt 包装为消息格式
        "stream":True
    }
    logger.info(f"发送流式申请：{prompt}")
    async with aiohttp.ClientSession() as session:  # 创建一个异步 HTTP 会话
        try:
            async with session.post(URL, headers=headers, json=payload) as response:
                if response.status == 200:  # 如果响应状态码为 200，表示请求成功
                    buffer = ""
                    async for chunk in response.content.iter_any():
                         buffer += chunk.decode("utf-8",errors="ignore").replace('\r', '')
                         lines = buffer.split("\n")
                         buffer = lines.pop()
                         for line in lines:
                             if line.startswith("data:"):
                                 data_content = line[5:].strip()
                                 if data_content.strip() == "[DONE]":
                                     yield "data: [DONE]\n\n"
                                     return
                                 yield f"data: {data_content}\n\n"
                else:
                    error_text = await response.text()
                    yield f'{{"error": true, "status": {response.status}, "message": "{error_text}"}}'
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            yield f'{{"error": true, "message": "{str(e)}"}}'
        finally :
            logger.info("LLM 流式连接关闭")
