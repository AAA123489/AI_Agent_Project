# main.py
# 导入日志模块
from src.logger import logger
import asyncio
from src import llm_client  # 导入 llm_client 模块，用于调用 LLM 客户端的函数  绝对导入


async def main() -> None:
    logger.info("开始发送请求到 LLM 客户端")  # 记录开始发送请求的日志

    prompts = [
        "1＋1等于多少？",
        "1＋2等于多少？",
        "1＋3等于多少？"
    ]
    results = await asyncio.gather(
        llm_client.call_llm_client(prompts[0]),
        llm_client.call_llm_client(prompts[1]),
        llm_client.call_llm_client(prompts[2])
    )
    for i, result in enumerate(results):
        if "error" in result:
            logger.error(f"请求 {i+1} 失败: {result['message']}")  # 记录请求失败的日志
        else:
            logger.info(f"请求 {i+1} 成功，响应数据: {result}")  # 记录请求成功的日志
    logger.info("所有请求已完成")  # 记录所有请求已完成的日志
if __name__ == "__main__":
    asyncio.run(main())