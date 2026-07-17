import asyncio  # 引入 asyncio 库，用于编写异步协程
import aiohttp  # 引入 aiohttp 库，用于异步 HTTP 请求
import logging
from typing import AsyncGenerator,Any
logger = logging.getLogger(__name__)

async def call_llm_client(prompt: str, api_key: str, api_url: str) -> dict[str, Any]: # 定义一个异步函数，用于向 LLM 客户端发送 prompt，并返回任意 JSON 兼容数据
    URL = api_url  # 使用传入的 api_url 参数
    headers = {
        "Authorization": f"Bearer {api_key}",  # 使用传入的 api_key 参数
        "Content-Type": "application/json"  # 设置请求的内容类型为 JSON
    }
    payload = {
        "max_tokens": 1000,  # 设置最大 token 数量为 100
        "model":"deepseek-v4-pro",  # 从环境变量中获取模型名称
        "messages":[{"role":"user","content":prompt}] , # 将 prompt 包装为消息格式
    }

    logger.info(f"发送请求的日志: {prompt}")  # 记录发送请求的日志

    async with aiohttp.ClientSession() as session:  # 创建一个异步 HTTP 会话
        try:
            async with session.post(URL, headers=headers, json=payload) as response:  # 发送 POST 请求
                if response.status == 200:  # 如果响应状态码为 200，表示请求成功
                    data = await response.json()
                    logger.info(f"请求成功")  # 记录接收到的响应数据
                    return data  # 返回数据
                else:
                    error_message = await response.text()  # 异步获取错误信息
                    logger.error(f"请求失败，状态码 {response.status}: {error_message}")  # 记录错误日志
                    return {"error": True, "status": response.status, "message": error_message}  # 返回错误信息
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:  # 捕获 aiohttp 客户端错误和超时错误
            logger.error(f"请求异常: {str(e)}")  # 记录异常日志
            return {"error": True, "message": str(e)}  # 返回异常信息

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
