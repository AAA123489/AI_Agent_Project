import chromadb
import uuid


class VectorStore:
    def __init__(self,db_path="./chroma_db",collection_name="ai_knowledge_base"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection=self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space":"cosine"}
        )


    def save_document(self,text:str,metadata:dict = None):
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata] if metadata else None
        )
        print(f"文档保存完成,ID:{doc_id}")

    def save_documents(self, documents: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        """批量保存文档到 Chroma"""
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"批量保存完成，共 {len(documents)} 条文档")

    def count(self) -> int:
        """返回 collection 中的文档总数"""
        return self.collection.count()

    def count_by_source(self, source: str) -> int:
        """按 source 过滤，返回指定来源的文档数"""
        result = self.collection.get(where={"source": source})
        return len(result["ids"]) if result["ids"] else 0

    def delete_by_source(self, source: str):
        """按 source 删除文档（覆盖更新前先清理旧数据）"""
        self.collection.delete(where={"source": source})
        print(f"已删除 source='{source}' 的旧数据")

    def clear_all(self):
        """清空 collection 中的所有文档"""
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
            print(f"已清空 collection，删除 {count} 条文档")
        else:
            print("collection 已为空，无需清空")

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