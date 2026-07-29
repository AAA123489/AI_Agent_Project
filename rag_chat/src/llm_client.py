import asyncio
import logging
from collections.abc import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)


async def call_llm_stream(prompt: str, api_key: str, api_url: str, model_name: str = "deepseek-v4-pro") -> AsyncGenerator[str, None]:
    """向 LLM 服务发送流式请求，返回标准化 SSE 行流（每条以 data: 开头）。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "max_tokens": 1000,
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    logger.info("发送流式请求：%s", prompt)
    async with aiohttp.ClientSession() as session:  # 创建一个异步 HTTP 会话
        try:
            async with session.post(api_url, headers=headers, json=payload) as response:
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
                    yield f'data: {{"error": true, "status": {response.status}, "message": "{error_text}"}}\n\n'
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            yield f'data: {{"error": true, "message": "{e!s}"}}\n\n'
        finally:
            logger.info("LLM 流式连接关闭")
