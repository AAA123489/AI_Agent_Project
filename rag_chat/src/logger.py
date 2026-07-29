import logging  # 标准库日志模块
import os  # 用于构建日志文件的绝对路径
import sys  # 用于获取标准错误输出流
from logging.handlers import RotatingFileHandler  # 支持按大小轮转的文件处理器

# 直接拿到根 logger，往它身上挂 handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 格式
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

# 控制台 handler
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# 文件 handler（路径问题后面单独说）
_logger_dir = os.path.dirname(os.path.abspath(__file__))   # src/ 目录
_project_root = os.path.dirname(_logger_dir)                # 项目根目录
_log_path = os.path.join(_project_root, "app.log")          # 拼接成绝对路径
file_handler = RotatingFileHandler(_log_path, maxBytes=500*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)