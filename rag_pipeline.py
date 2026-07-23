import hashlib
from text_splitter import RecursiveTextSplitter
# 昨天写的 VectorStore，保存在 src/vector_store.py 中
from src.vector_store import VectorStore

def ingest_document(file_path: str):
    """
    文档入库流水线：读取 -> 切分 -> 生成ID -> 存入向量数据库
    """
    # 1. 读取文件内容
    print(f"📖 正在读取文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 调用切分器
    print("✂️ 正在切分文本...")
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"✅ 文本切分完成，共生成 {len(chunks)} 个块。")

    # 3. 初始化向量数据库
    vector_store = VectorStore(collection_name="my_rag_collection")

    # 4. 遍历 chunks，生成唯一 ID 并批量存入
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        # 核心：使用 MD5 Hash 生成唯一 ID
        # 好处：如果文档内容没变，Hash 值就不变，再次入库时可以实现"覆盖"或"跳过"（幂等性）
        chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        unique_id = f"{file_path}_{i}_{chunk_hash}"

        # 构建元数据（Metadata），方便后续过滤检索
        metadata = {
            "source": file_path,
            "chunk_index": i
        }

        ids.append(unique_id)
        documents.append(chunk)
        metadatas.append(metadata)

    # 5. 批量存入 Chroma（批量操作比单条循环存入快得多！）
    print("🚀 正在将数据存入 ChromaDB...")
    vector_store.save_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("🎉 文档入库流水线执行完毕！")


if __name__ == "__main__":
    from src.vector_store import VectorStore

    file_path = "测试使用.txt"
    vector_store = VectorStore(collection_name="my_rag_collection")

    # 入库前：清空旧数据，保证干净环境
    print(f"📊 入库前 collection 文档总数: {vector_store.count()}")
    vector_store.clear_all()

    # 1. 将文档入库
    ingest_document(file_path)

    # 入库后：验证数量
    print(f"📊 入库后 collection 文档总数: {vector_store.count()}")

    # 2. 设定测试提问
    query = "FastAPI 有什么特点？"
    print(f"\n🔍 正在检索问题: '{query}'")

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

            print(f"📌 匹配结果 {i+1} (来源: {metadata.get('source', '未知')})")
            print(f"内容预览: {chunk_text[:150]}...")
            print("-" * 50)
