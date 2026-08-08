import hashlib
import logging

from src.vector_store import VectorStore
from text_splitter import RecursiveTextSplitter, sanitize_privacy

logger = logging.getLogger(__name__)


def ingest_document(file_path: str, author: str = "unknown", year: str = "", vector_store: VectorStore | None = None):
    """
    文档入库流水线：读取 -> 切分 -> 生成ID -> 存入向量数据库
    """
    if vector_store is None:
        vector_store = VectorStore(collection_name="my_rag_collection")

    # 1. 读取文件内容
    logger.info("正在读取文件: %s", file_path)
    if file_path.lower().endswith(".pdf"):
        from document_parser import parse_pdf
        text, meta = parse_pdf(file_path)
        author = meta.get("author", author)
        year = meta.get("year", year)
    elif file_path.lower().endswith(".docx"):
        from document_parser import parse_docx
        text, meta = parse_docx(file_path)
        author = meta.get("author", author)
        year = meta.get("year", year)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # 2. 隐私脱敏（删除个人手机号/邮箱，再切分入库）
    text = sanitize_privacy(text)

    # 3. 调用切分器
    logger.info("正在切分文本...")
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    logger.info("文本切分完成，共生成 %d 个块。", len(chunks))

    # 3. 遍历 chunks，生成唯一 ID 并批量存入
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        # 使用 SHA-256 生成唯一 ID（去重、幂等）
        chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
        unique_id = f"{file_path}_{i}_{chunk_hash}"

        # 构建元数据（Metadata），方便后续过滤检索
        metadata = {
            "source": file_path,
            "chunk_index": i,
            "author": author,
            "year": year
        }

        ids.append(unique_id)
        documents.append(chunk)
        metadatas.append(metadata)

    # 4. 批量存入 Chroma（批量操作比单条循环存入快得多！）
    logger.info("正在将数据存入 ChromaDB...")
    vector_store.save_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    logger.info("文档入库流水线执行完毕！")


if __name__ == "__main__":
    # 使用方法: python rag_pipeline.py <你的文档路径>
    import sys
    if len(sys.argv) < 2:
        print("用法: python rag_pipeline.py <文档路径>")
        print("支持格式: .txt / .md / .pdf")
        sys.exit(1)
    file_path = sys.argv[1]
    vector_store = VectorStore(collection_name="my_rag_collection")

    # 入库前：按 source 覆盖旧数据（同文件幂等入库，不影响其他文档）
    total_before = vector_store.count()
    source_before = vector_store.count_by_source(file_path)
    logger.info("入库前 collection 文档总数: %d（其中 source='%s' 的有 %d 条）", total_before, file_path, source_before)
    if source_before > 0:
        vector_store.delete_by_source(file_path)
        logger.info("已删除旧数据 %d 条，准备重新入库", source_before)

    # 1. 将文档入库（复用同一个 VectorStore 实例）
    ingest_document(file_path, vector_store=vector_store)

    # 入库后：验证数量
    logger.info("入库后 collection 文档总数: %d", vector_store.count())

    # 2. 设定测试提问
    query = "FastAPI 有什么特点？"
    logger.info("正在检索问题: '%s'", query)

    # 3. 调用检索方法
    results = vector_store.search_similar(query=query, n_results=3)

    # 4. 格式化打印检索结果
    print("-" * 50)
    if not results:
        print("⚠️ 未检索到任何相关内容！请检查文档是否入库成功或 Embedding 模型是否正常。")
    else:
        for i, res in enumerate(results):
            chunk_text = res.get("text", res)
            metadata = res.get("metadata", {})

            print(f"[{i+1}] (来源: {metadata.get('source', '未知')})")
            print(f"内容预览: {chunk_text[:150]}...")
            print("-" * 50)
