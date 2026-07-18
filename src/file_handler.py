
import logging

logger = logging.getLogger(__name__)


def read_text_file(file_path: str) -> str:

    try:
        # 以 utf-8 编码打开并读取整个文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"成功读取文件：{file_path}")
        return content
    except FileNotFoundError:
        # 常见错误：文件路径错误或文件确实不存在
        logger.error(f"文件不存在: {file_path}")
        return ""
    except Exception as e:
        # 捕获并记录所有其他异常，避免抛出到上层
        logger.error(f"读取文件发生未知错误：{str(e)}")
        return ""


if __name__ == "__main__":
    # 当作为脚本直接运行时，执行简单的读取测试（示例目的）
    # 注意：生产代码不应在模块顶层执行 I/O 操作，这里仅用于本地快速验证
    res = read_text_file(r"E:\vscode-program\AI_Agent_Project\requirements.txt")
    print(res)