import logging
import sys
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)

# 配置日志：同时输出控制台+写入文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler("app.log",maxBytes=500*1024, backupCount=3, encoding="utf-8")
    ]
)
if __name__ == "__main__":
    logger.info("日志模块测试")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.debug ("调试信息")