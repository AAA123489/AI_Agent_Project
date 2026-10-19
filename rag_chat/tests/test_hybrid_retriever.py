"""测试 RRF 融合（src/hybrid_retriever.py 的 rrf_fuse）——重点是 vector_distance 保留。

纯函数测试，零依赖（不 import jieba / rank_bm25 / chromadb）。

为什么单独测这个字段：融合后的 `distance` 会被 BM25 路的归一化分污染
（BM25 第一名恒为 0.0 → 展示成"相似度 100%"），**唯一可信的相似度信号**是向量路的
原始余弦距离，由 rrf_fuse 单独保留在 `vector_distance` 里。召回自检
（src/recall_guard.py）的快通过通道完全依赖它——这个字段一旦被静默丢掉，
自检会退化成"每问都调判官"（慢且更容易误判），而且不会报任何错。
"""
from src.hybrid_retriever import rrf_fuse


def _vec(text, distance, source="a.txt"):
    return {"text": text, "metadata": {"source": source}, "distance": distance}


def _bm25(text, distance, source="a.txt"):
    return {"text": text, "metadata": {"source": source}, "distance": distance}


def test_向量路命中的块带上vector_distance():
    out = rrf_fuse([_vec("甲", 0.20), _vec("乙", 0.45)], [], top_n=8)
    by_text = {d["text"]: d for d in out}

    assert by_text["甲"]["vector_distance"] == 0.20
    assert by_text["乙"]["vector_distance"] == 0.45


def test_只有BM25命中的块没有vector_distance():
    """判官/快通过必须能区分「这个块有没有可信距离」，缺字段本身就是信号。"""
    out = rrf_fuse([], [_bm25("甲", 0.0), _bm25("乙", 0.3)], top_n=8)

    assert all("vector_distance" not in d for d in out)


def test_BM25的第一名会污染distance但不污染vector_distance():
    """这就是必须另存字段的原因。

    BM25 第一名 distance 恒为 0.0（归一化分 best/best=1 → 1-1=0），
    两路都命中时它比向量距离小、于是被 rrf_fuse 选中当 `distance`。
    所以「筛 origin=='vector' 再读 distance」拿到的是空集——
    实测口径下，靠 distance 判相似度会把无关块看成"相似度 100%"。
    """
    out = rrf_fuse([_vec("甲", 0.62)], [_bm25("甲", 0.0)], top_n=8)
    doc = out[0]

    assert doc["distance"] == 0.0, "融合后的 distance 被 BM25 的 0.0 占据（展示成 100%）"
    assert doc["vector_distance"] == 0.62, "真实余弦距离必须保留在 vector_distance 里"


def test_两路都命中时RRF分数累加上浮():
    """两路都命中的块名次分翻倍，应排在只被一路命中的块前面。"""
    out = rrf_fuse([_vec("甲", 0.5), _vec("乙", 0.3)], [_bm25("甲", 0.4)], top_n=8)

    assert out[0]["text"] == "甲"


def test_按文本去重():
    """同一块被爬进两个分类目录 / 重叠窗口会产生内容相同的块，不能重复占位。"""
    out = rrf_fuse([_vec("甲", 0.2), _vec("甲", 0.5)], [_bm25("甲", 0.1)], top_n=8)

    assert len(out) == 1


def test_不改动入参dict():
    """带 vector_distance 的是副本——直接改入参 dict 会污染调用方的候选列表。"""
    vec = [_vec("甲", 0.2)]

    rrf_fuse(vec, [], top_n=8)

    assert "vector_distance" not in vec[0]


def test_空输入不炸():
    assert rrf_fuse([], [], top_n=8) == []


def test_空文本块被跳过():
    """text 为空无法去重识别（块内容相同即同一块），跳过而不是当成同一个空块合并。"""
    out = rrf_fuse([{"text": "", "metadata": {}, "distance": 0.1}], [_bm25("甲", 0.2)], top_n=8)

    assert [d["text"] for d in out] == ["甲"]


def test_top_n截断():
    ranked = [_vec(f"块{i}", 0.1 * i) for i in range(1, 11)]

    assert len(rrf_fuse(ranked, [], top_n=3)) == 3


# ── BM25 单例失效（定时增量爬取之后） ─────────────────────────────
# 定时爬取会往 Chroma 写新块，但 BM25 是**进程内一次性建好的倒排索引**。
# 不置空的话，新文章在服务重启前只有向量路能召回，RRF 的两路输入不一致。

class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def get(self, include=None):
        return {"documents": list(self._docs), "metadatas": [{} for _ in self._docs]}


def test_reset后下次get按当前collection重建(monkeypatch):
    """reset 必须真正丢弃旧索引，重建时读到爬取后的新 collection。"""
    import src.hybrid_retriever as hr

    built = []

    def fake_build(self, collection):
        built.append(list(collection.get()["documents"]))
        self._corpus = list(collection.get()["documents"])
        self._bm25 = object()  # 非 None 即视为可用，绕开 jieba/rank_bm25
        return True

    monkeypatch.setattr(hr.Bm25Retriever, "build", fake_build)
    monkeypatch.setattr(hr, "_bm25_retriever", None)

    old = hr.get_bm25_retriever(_FakeCollection(["旧文章块"]))
    hr.reset_bm25_retriever()
    new = hr.get_bm25_retriever(_FakeCollection(["旧文章块", "新文章块"]))

    assert new is not old, "reset 后必须新建实例，否则新块永远搜不到"
    assert built == [["旧文章块"], ["旧文章块", "新文章块"]]
    assert new._corpus == ["旧文章块", "新文章块"]


def test_reset是幂等的且能让构建失败后重试(monkeypatch):
    """构建失败时单例保持 None；reset 对 None 不抛异常，下次访问重新尝试。"""
    import src.hybrid_retriever as hr

    monkeypatch.setattr(hr.Bm25Retriever, "build", lambda self, c: False)
    monkeypatch.setattr(hr, "_bm25_retriever", None)

    assert hr.get_bm25_retriever(_FakeCollection(["块"])) is None
    hr.reset_bm25_retriever()
    hr.reset_bm25_retriever()


def test_爬虫入库后接线了BM25重置():
    """app_fastapi 的重依赖无法在单测里 import，这里钉住接线本身。

    条件必须是 total_chunks > 0：日常增量经常 0 篇，重建一次要几秒到十几秒。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app_fastapi.py").read_text(encoding="utf-8")

    assert 'if result.get("total_chunks", 0) > 0:' in src
    assert "reset_bm25_retriever()" in src
    idx_if = src.index('if result.get("total_chunks", 0) > 0:')
    idx_call = src.index("reset_bm25_retriever()")
    assert idx_if < idx_call, "重置必须在条件分支内"
