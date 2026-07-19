from dotenv import load_dotenv  # 用于从 .env 文件加载环境变量
import os  # 访问操作系统环境变量与路径
import logging

logger = logging.getLogger(__name__)

# 尝试从 .env 文件中加载配置到环境变量
load_dotenv()  # 在当前工作目录查找 .env 文件并将其中的键值对加载到 os.environ


class ConfigManager:
    def __init__(self) -> None:
        # 从环境变量中获取 API_KEY（若不存在则为 None）
        self.api_key = os.getenv("API_KEY")
        if self.api_key is None:
            logger.warning("API_KEY 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("API_KEY 已成功加载。")
            # 只打印后 4 位，避免敏感信息泄露
            logger.debug(f"API_KEY: ...{self.api_key[-4:]}")

        # 从环境变量中获取 API_URL（若不存在则为 None）
        self.api_url = os.getenv("API_URL")
        if self.api_url is None:
            logger.warning("API_URL 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("API_URL 已成功加载。")
            logger.debug(f"API_URL: {self.api_url}")

        # 从环境变量中获取模型名称（若不存在则使用默认值）
        self.model_name = os.getenv("MODEL_NAME", "deepseek-v4-pro")
        logger.info(f"MODEL_NAME: {self.model_name}")

        # 获取日志级别，若未设置则使用默认值 "INFO"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.getLogger().setLevel(self.log_level.upper())

    def get_config(self) -> dict[str, str | None]:
        """
        返回当前的配置参数字典。

        返回值格式示例:
        {
            "API_KEY": str | None,
            "API_URL": str | None,
            "MODEL_NAME": str,
            "LOG_LEVEL": str
        }
        """
        return {
            "API_KEY": self.api_key,
            "API_URL": self.api_url,
            "MODEL_NAME": self.model_name,
            "LOG_LEVEL": self.log_level
        }
