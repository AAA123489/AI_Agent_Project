import asyncio
import logging
import os
import uuid

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# 国内访问 HuggingFace 需走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 中文友好的 Embedding 模型（默认 all-MiniLM-L6-v2 只支持英文）
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_fn = None


def _get_embedding_fn():
    """延迟加载 Embedding 函数（避免导入时下载模型）。"""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


class VectorStore:
    def __init__(self, db_path="./chroma_db", collection_name="ai_knowledge_base", distance_threshold: float = 0.85):
        self.distance_threshold = distance_threshold
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_get_embedding_fn(),
        )

    def save_document(self, text: str, metadata: dict | None = None):
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata] if metadata else None
        )
        logger.info("文档保存完成, ID: %s", doc_id)

    def save_documents(self, documents: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None):
        """批量保存文档到 Chroma"""
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        logger.info("批量保存完成，共 %d 条文档", len(documents))

    def count(self) -> int:
        """返回 collection 中的文档总数"""
        return self.collection.count()

    def count_by_source(self, source: str) -> int:
        """按 source 过滤，返回指定来源的文档数"""
        result = self.collection.get(where={"source": source})
        return len(result["ids"]) if result["ids"] else 0

    def count_by_category(self, category: str) -> int:
        """按 category 过滤，返回指定分类的文档数"""
        result = self.collection.get(where={"category": category})
        return len(result["ids"]) if result["ids"] else 0

    def delete_by_source(self, source: str):
        """按 source 删除文档（覆盖更新前先清理旧数据）"""
        self.collection.delete(where={"source": source})
        logger.info("已删除 source='%s' 的旧数据", source)

    def clear_all(self):
        """清空 collection 中的所有文档"""
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
            logger.info("已清空 collection，删除 %d 条文档", count)
        else:
            logger.info("collection 已为空，无需清空")

    def search_similar(self, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return formatted_results

    async def search_similar_async(self, query: str, n_results: int = 3):
        """异步包装：将同步查询扔到线程池，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.search_similar, query, n_results)