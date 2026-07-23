# test_vector.py
from src.vector_store import VectorStore

if __name__ == "__main__":
    # 1. 实例化类（自动完成底层初始化）
    store = VectorStore()
    
    # 2. 存入测试数据
    store.save_document("FastAPI 是一个高性能的 Python 框架", {"category": "web"})
    store.save_document("Chroma 是一个开源的向量数据库", {"category": "database"})
    store.save_document("Python 适合做人工智能", {"category": "language"})
    
    print("\n🔍 开始检索：哪种语言适合做后端？")
    
    # 3. 执行检索
    search_results = store.search_similar("哪种语言适合做后端？", n_results=2)
    
    # 4. 打印结果
    for res in search_results:
        print(f"匹配文本: {res['text']}")
        print(f"元数据: {res['metadata']}")
        print(f"相似度距离: {res['distance']:.4f}")
        print("-" * 30)