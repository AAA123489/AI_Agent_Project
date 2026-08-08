"""
hybrid_retriever.py — 混合检索三件套（BM25 + RRF 融合 + Reranker）
================================================================
纯向量检索的短板是"关键词精确命中"：关键句（如 Q3 的「报名截止」）
埋在文档中段、与 query 语义重叠度低时，余弦相似度经常捞不到。

本模块补两路信号，让检索更工程化：
1. BM25 关键词召回 —— jieba 中文分词 + rank_bm25 OKAPI 打分，补精确命中
2. RRF 融合 —— Reciprocal Rank Fusion，把向量/BM25 两路排名的名次融合，
   无权重参数、稳健，避免手工调融合系数
3. Reranker —— 可选重排（CrossEncoder），开关开启才加载模型（~100MB），
   对候选 (query, 文本) 对做精排

所有组件均惰性加载 + 模块级单例缓存，聊天链路只加载用到的部分。

依赖：jieba、rank-bm25（轻量纯 Python 包）；sentence-transformers（重排才用）。
"""

import logging
import os

# 国内访问 HuggingFace 需走镜像（与 vector_store 一致；
# transformers 下载 CrossEncoder 模型时也读这个环境变量）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)

# 文档块级倒排索引的召回宽口径：全库打分后先取 30 条再按年过滤，防止年份过滤砍光
_BM25_PREFETCH = 30


class Bm25Retriever:
    """BM25 关键词召回器。

    - build()：全量拉取 collection 所有块（chunks），jieba 分词建 OKAPI 倒排索引
    - search()：query 分词 → 全库打分 → 取 prefetch 条 → 按 year 过滤 → 取 top_n
    - 返回结构与 vector_store.search_similar 对齐：{text, metadata, distance}
      distance 由归一化 BM25 分换算（best=1.0 → distance=0.0），
      保证下游「相似度: X%」展示格式一致。
    """

    def __init__(self, collection=None):
        self.collection = collection
        self._bm25 = None
        self._corpus: list[str] = []
        self._metadatas: list[dict] = []

    def build(self, collection) -> bool:
        """全量加载块 + 建索引。失败返回 False，调用方回退纯向量。"""
        try:
            import jieba
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            logger.warning("BM25 依赖未安装（jieba/rank_bm25），回退纯向量: %s", e)
            return False
        try:
            data = collection.get(include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            if not docs:
                logger.warning("collection 为空，BM25 索引跳过")
                return False
            tokenized = [jieba.lcut(d) for d in docs]
            self.collection = collection
            self._corpus = docs
            self._metadatas = [m or {} for m in metas]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 索引构建完成: %d 块", len(docs))
            return True
        except Exception as e:
            logger.error("BM25 索引构建失败，回退纯向量: %s", e)
            self._bm25 = None
            return False

    def search(self, query: str, top_n: int = 8, year: str | None = None) -> list[dict]:
        """BM25 打分 → 年份过滤 → 截断 top_n。"""
        if self._bm25 is None:
            return []
        import jieba

        q_tokens = [t for t in jieba.lcut(query) if t.strip()]
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # 全库按分排序，先取宽口径 prefetch，再做年份过滤
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:_BM25_PREFETCH]
        best = scores[order[0]] if order else 1.0
        best = best if best > 0 else 1.0
        out = []
        for i in order:
            meta = self._metadatas[i]
            # 年份过滤放宽为 ±1 年窗口（与向量侧口径一致）：query 年份常是「内容年份」，
            # 文档 year 是「发布年份」，两者常差一年（如 2025-12 发布 2026 挑战杯通知）。
            # 空年份放行；窗口外（≥2 年前）旧通知仍排除。
            if year:
                y = int(year)
                if str(meta.get("year", "")) not in (str(y - 1), year, str(y + 1), ""):
                    continue
            norm = max(0.0, scores[i] / best)  # 归一化到 [0,1]，best 块 = 1.0
            out.append({
                "text": self._corpus[i],
                "metadata": meta,
                "distance": 1.0 - norm,
            })
            if len(out) >= top_n:
                break
        return out


# 模块级单例缓存（懒加载）
_bm25_retriever: Bm25Retriever | None = None


def get_bm25_retriever(collection) -> Bm25Retriever | None:
    """获取全局 BM25 单例。首次调用建索引；失败返回 None（调用方回退纯向量）。"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = Bm25Retriever()
        if not _bm25_retriever.build(collection):
            _bm25_retriever = None
    return _bm25_retriever


def rrf_fuse(vector_results: list[dict], bm25_results: list[dict], k: int = 60, top_n: int = 8) -> list[dict]:
    """Reciprocal Rank Fusion：把两路排序的名次融合成单一排序。

    RRF 分数 = Σ 1/(k + rank)，k=60 是论文常用值。无需权重参数，稳健。
    两路都命中的文档自然获得双份名次分，天然上浮。

    - 按 text 去重识别同一块（块内容相同即同一块，兼容不同爬取目录的重复）
    - 重复出现时保留 similarity 更高（distance 更小）的一份，保证展示值不虚低
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    def _add(ranked: list[dict]) -> None:
        for rank, doc in enumerate(ranked, 1):
            text = doc.get("text", "")
            if not text:
                continue
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
            old = docs.get(text)
            if old is None or doc.get("distance", 1.0) < old.get("distance", 1.0):
                docs[text] = doc

    _add(vector_results)
    _add(bm25_results)
    ranked = sorted(docs.items(), key=lambda kv: -scores[kv[0]])
    return [d for _, d in ranked[:top_n]]


class Reranker:
    """可选重排器：CrossEncoder 对 (query, 文本) 打分精排。

    模型为中文精排标准选择 BGE-Reranker-base（~440MB），
    仅 RETRIEVAL_RERANK=on 时才加载（hf-mirror 可下载）。
    """

    # 中文精排模型（BAAI/bge-reranker-base），hf-mirror 可下载
    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is None:
            # 强制走镜像，避免 transformers 直连 huggingface.co 被墙
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from sentence_transformers import CrossEncoder
            logger.info("加载 Reranker 模型: %s", self.MODEL_NAME)
            self._model = CrossEncoder(self.MODEL_NAME)
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
        if not candidates:
            return []
        try:
            model = self.load()
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            return [c for c, _ in ranked[:top_n]]
        except Exception as e:
            # 重排失败：原样返回全部候选（不截断），等价于关掉重排，不降级质量
            logger.error("重排失败，保持原顺序: %s", e)
            return candidates


# 模块级单例缓存
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
