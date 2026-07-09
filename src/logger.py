"""
logger 模块

提供项目统一的 `logger` 实例，已配置同时输出到控制台和滚动文件日志。
"""
import logging  # 标准库日志模块
import sys  # 用于获取标准错误输出流
from logging.handlers import RotatingFileHandler  # 支持按大小轮转的文件处理器


# 创建模块级 logger，使用模块名作为记录器名称，便于区分不同模块的日志来源
logger = logging.getLogger(__name__)

# 配置根日志行为：级别、格式和处理器
# - level=logging.INFO: 默认输出 INFO 及以上级别的日志
# - format: 包含时间戳、日志级别和消息，便于排查问题
# - handlers: 同时输出到 stderr（控制台）和文件（支持按大小滚动）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        # 控制台输出，错误信息写入 stderr 以便与 stdout 区分
        logging.StreamHandler(sys.stderr),
        # 文件输出，达到 maxBytes 时回滚到新文件，保留 backupCount 份旧日志
        RotatingFileHandler("app.log", maxBytes=500 * 1024, backupCount=3, encoding="utf-8")
    ]
)


if __name__ == "__main__":
    # 简单的自检输出，用于本模块单独运行时验证日志行为
    logger.info("日志模块测试")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.debug("调试信息")