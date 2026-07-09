from dotenv import load_dotenv
import os
from src.logger import logger

# 尝试从 .env 文件中加载配置
load_dotenv()

class ConfigManager:
    """
    配置管理器类，用于读取和管理配置文件中的参数。
    """
    def __init__(self) -> None:
        # 从环境变量中获取配置参数
        self.api_key = os.getenv("API_KEY")
        if self.api_key is None:
            logger.warning("API_KEY 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("API_KEY 已成功加载。")
            logger.debug(f"API_KEY: {self.api_key}")

        self.api_url = os.getenv("API_URL")
        if self.api_url is None:
            logger.warning("API_URL 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("API_URL 已成功加载。")
            logger.debug(f"API_URL: {self.api_url}")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")  # 默认日志级别为 INFO
    def get_config(self) -> dict[str, str | None]:
        """
        返回当前的配置参数。
        """
        return {
            "API_KEY": self.api_key,
            "API_URL": self.api_url,
            "LOG_LEVEL": self.log_level
        }
