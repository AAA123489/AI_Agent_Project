# main.py
# 导入日志模块
from logger import logger
from config import ConfigManager

if __name__ == "__main__":
    try:
        config = ConfigManager()
        logger.info("===== 程序开始运行 =====")
        config_get = config.get_config()
        logger.info(f"当前配置参数: {config_get}")
        logger.info("===== 程序运行结束 =====")
    except Exception as e:
        logger.error(f"程序运行时发生错误: {e}")