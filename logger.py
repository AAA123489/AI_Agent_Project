from loguru import logger
import sys

# 配置日志：同时输出控制台+写入文件
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("app.log", rotation="500 KB", encoding="utf-8")

if __name__ == "__main__":
    logger.info("日志模块测试")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.debug ("调试信息")