import asyncio  # 引入 asyncio 库，用于编写异步协程
import aiohttp  # 引入 aiohttp 库，用于异步 HTTP 请求
import os  # 引入 os 模块，用于访问环境变量等操作
from dotenv import load_dotenv  # 从 dotenv 库中导入 load_dotenv，用于加载 .env 文件
from logger import logger  # 从 logger 模块中导入 logger 对象，用于日志记录
load_dotenv()  # 从当前工作目录加载 .env 文件中的环境变量

async def call_llm_client(prompt:str):  # 定义一个异步函数，用于向 LLM 客户端发送 prompt
    URL = os.getenv("API_URL")  # 从环境变量中获取 API_URL
    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}",  # 从环境变量中获取 API_KEY 并设置为 Bearer Token
        "Content-Type": "application/json"  # 设置请求的内容类型为 JSON
    }
    payload = {
        "max_tokens": 100,  # 设置最大 token 数量为 100
        "model":"deepseek-v4-pro",  # 从环境变量中获取模型名称
        "messages":[{"role":"user","content":prompt}]  # 将 prompt 包装为消息格式
    }

    logger.info(f"发送请求的日志: {prompt}")  # 记录发送请求的日志



    async with aiohttp.ClientSession() as session:  # 创建一个异步 HTTP 会话
        try:
            async with session.post(URL, headers=headers, json=payload) as response:  # 发送 POST 请求
                if response.status == 200:  # 如果响应状态码为 200，表示请求成功
                    data = await response.json()  # 异步获取响应的 JSON 数据
                    logger.info(f"请求成功")  # 记录接收到的响应数据
                    return data  # 返回数据
                else:
                    error_message = await response.text()  # 异步获取错误信息
                    logger.error(f"请求失败，状态码 {response.status}: {error_message}")  # 记录错误日志
                    return {"error": True, "status": response.status, "message": error_message}  # 返回错误信息
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:  # 捕获 aiohttp 客户端错误和超时错误
            logger.error(f"请求异常: {str(e)}")  # 记录异常日志
            return {"error": True, "message": str(e)}  # 返回异常信息
