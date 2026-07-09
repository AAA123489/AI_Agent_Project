# file_handler.py
from logger import logger

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"成功读取文件：{file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"读取文件发生未知错误：{str(e)}")
        return ""

if __name__ == "__main__":
    # 测试一个不存在的文件
    res = read_text_file(r"E:\vscode-program\AI_Agent_Project\requirements.txt")
    print(res)