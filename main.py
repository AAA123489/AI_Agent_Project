# main.py
# 导入日志模块
from logger import logger
from file_handler import read_text_file

if __name__ == "__main__":
    # 测试读取文件
    file_path = r"E:\vscode-program\AI_Agent_Project\requirements.txt"
    content = read_text_file(file_path)
    if content:
        logger.info(f"文件内容:\n{content}")
    else:
        logger.warning("未能读取到文件内容")
    logger.info("===== 程序运行结束 =====")