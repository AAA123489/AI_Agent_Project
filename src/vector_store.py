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