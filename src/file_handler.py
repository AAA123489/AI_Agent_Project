"""
file_handler 模块

包含用于读取文本文件的辅助函数。模块对外提供 `read_text_file`，
该函数以安全的方式读取文件内容并通过 `logger` 记录成功或错误信息。
"""
from src.logger import logger  # 项目内统一的日志实例，用于记录信息、警告和错误


def read_text_file(file_path: str) -> str:
    """
    以 UTF-8 编码读取给定路径的文本文件并返回其内容。

    参数:
        file_path: 要读取的文件路径（字符串）

    返回:
        文件内容的字符串；如果文件不存在或发生错误，则返回空字符串。

    日志行为:
        - 成功读取时记录 info 级别日志
        - 文件未找到时记录 error 级别日志
        - 其它异常也记录为 error，并返回空字符串
    """
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