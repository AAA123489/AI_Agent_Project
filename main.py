"""
main 模块

演示如何并发调用 LLM 客户端并处理返回结果的简单示例脚本。
"""

# 导入项目级 logger，用于记录运行信息和错误
from src.logger import logger
import asyncio  # 用于异步并发执行任务
from src import llm_client  # 导入封装了与 LLM 服务交互的客户端函数


async def main() -> None:
    """
    主异步入口：并发发送若干 prompt 到 LLM 客户端并处理响应。

    行为说明：
    - 构造一个 prompt 列表
    - 使用 `asyncio.gather` 并发调用 `llm_client.call_llm_client`
    - 根据返回值中是否包含 `error` 字段来判断请求是否成功
    """
    logger.info("开始发送请求到 LLM 客户端")  # 记录开始发送请求的日志

    # 要发送给 LLM 的测试 prompts 列表
    prompts = [
        "1＋1等于多少？",
        "1＋2等于多少？",
        "1＋3等于多少？"
    ]

    # 并发执行多个 LLM 请求，所有任务会同时发出，直到全部完成
    # 返回值通常为 JSON 可序列化的数据结构（函数签名使用 Any），可能为字典或列表
    results = await asyncio.gather(
        llm_client.call_llm_client(prompts[0]),
        llm_client.call_llm_client(prompts[1]),
        llm_client.call_llm_client(prompts[2])
    )

    # 遍历处理每个请求的结果；根据 llm_client 的实现，失败时会返回包含 "error" 字段的 dict
    for i, result in enumerate(results):
        # 先做类型与错误字段检查，避免因非字典类型导致 KeyError
        if isinstance(result, dict) and "error" in result:
            logger.error(f"请求 {i+1} 失败: {result.get('message')}")  # 记录请求失败的日志
        else:
            logger.info(f"请求 {i+1} 成功，响应数据: {result}")  # 记录请求成功的日志

    logger.info("所有请求已完成")  # 记录所有请求已完成的日志


if __name__ == "__main__":
    # 使用 asyncio.run 运行顶层协程，适用于 Python 3.7+
    asyncio.run(main())