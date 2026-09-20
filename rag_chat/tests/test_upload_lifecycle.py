"""测试上传文档的生命周期 —— 入库后的可见性与可撤销性。

背景（2026-09-20 发现的真实问题）：
5 篇测试期间走 /upload 传进来的文档在 my_rag_collection 里躺了半个月，
共 378 块，其中一篇是**外校**（太原科技大学）的拟录取名单，含大量真人姓名。
所有评测成绩都是在 5919 块的语料上量的，线上却是 6297 块 ——
**线上系统和被测系统不是同一个**，而没人发现，因为：

  1. get_knowledge_base_stats 的 total_chunks 是「整个 collection」，
     但 category_counts 只覆盖 13 个爬虫分类，"用户上传"不在其中，
     两个数对不上却没有任何地方把这个差额说出来；
  2. 入库之后**没有任何 API 能撤销** —— 只能开 Python 手敲 col.delete()。

本文件覆盖这两个缺口的修复。用真实临时 Chroma（同 test_vector_store.py），
显式传入 vector_store，不碰线上库、不 import app_backend。
"""
import shutil
import tempfile

import pytest

from campus_scraper.pipeline import (
    UPLOAD_CATEGORY,
    get_knowledge_base_stats,
    remove_uploaded_file,
)
from src.vector_store import VectorStore


# ── fixtures / 辅助 ───────────────────────────────────────────


@pytest.fixture
def store():
    """独立临时 Chroma，避免污染线上库。"""
    tmpdir = tempfile.mkdtemp()
    try:
        yield VectorStore(db_path=tmpdir, collection_name="test_upload")
    finally:
        # Windows 上 SQLite 文件锁可能未立即释放，忽略清理失败
        shutil.rmtree(tmpdir, ignore_errors=True)


def _add_crawled(store, source="关于2026年端午节放假的通知", category="通知公告", n=3):
    """塞一篇"爬虫抓来的"文档。"""
    store.save_documents(
        documents=[f"官网正文第 {i} 块" for i in range(n)],
        ids=[f"scrape_{source}_{i}" for i in range(n)],
        metadatas=[
            {"source": source, "category": category, "publish_date": "2026-06-01"}
            for _ in range(n)
        ],
    )


def _add_upload(store, source="测试上传.pdf", n=2):
    """塞一篇"用户上传的"文档。"""
    store.save_documents(
        documents=[f"上传内容第 {i} 块" for i in range(n)],
        ids=[f"upload_{source}_{i}" for i in range(n)],
        metadatas=[
            {"source": source, "category": UPLOAD_CATEGORY, "publish_date": "2026-09-20"}
            for _ in range(n)
        ],
    )


# ── remove_uploaded_file：撤销上传 ────────────────────────────


def test_删除不存在的文件名不报错只是found为假(store):
    result = _run(remove_uploaded_file("根本没传过.pdf", store))
    assert result["found"] is False
    assert result["deleted_chunks"] == 0


def test_上传的文档能被删掉且块数正确(store):
    _add_upload(store, "测试上传.pdf", n=5)
    assert store.count() == 5

    result = _run(remove_uploaded_file("测试上传.pdf", store))

    assert result["found"] is True
    assert result["deleted_chunks"] == 5
    assert store.count() == 0


def test_删除只影响目标文档(store):
    _add_upload(store, "甲.pdf", n=2)
    _add_upload(store, "乙.pdf", n=3)

    _run(remove_uploaded_file("甲.pdf", store))

    assert store.count_by_source("甲.pdf") == 0
    assert store.count_by_source("乙.pdf") == 3


def test_爬虫语料拒删且数据分毫未动(store):
    """一个 DELETE 请求不该有能力抹掉学校官网的语料 —— chroma_db 没有备份。"""
    _add_crawled(store, "关于2026年端午节放假的通知", n=4)

    with pytest.raises(ValueError, match="不是用户上传"):
        _run(remove_uploaded_file("关于2026年端午节放假的通知", store))

    assert store.count_by_source("关于2026年端午节放假的通知") == 4


def test_同一source混了两种category也拒删(store):
    """半爬虫半上传的脏数据不许删 —— 宁可拒了让人来查，不可删一半留一半。"""
    store.save_documents(
        documents=["爬虫块", "上传块"],
        ids=["a", "b"],
        metadatas=[
            {"source": "混合.pdf", "category": "通知公告"},
            {"source": "混合.pdf", "category": UPLOAD_CATEGORY},
        ],
    )

    with pytest.raises(ValueError, match="不是用户上传"):
        _run(remove_uploaded_file("混合.pdf", store))

    assert store.count_by_source("混合.pdf") == 2


def test_用完整路径当source的文档不会被裸文件名删掉(store):
    """MCP 的 ingest_file 把**整条路径**写进 source（见 rag_pipeline.ingest_document）。

    所以拿裸文件名去删它必然 found=False。这不是 bug，是明确的边界：
    这条路径进来的文档属于「来源不明」，靠 unaccounted_chunks 报警，
    而不是靠这个函数删。此用例把这个边界钉住，免得日后误以为它覆盖了 MCP。
    """
    store.save_documents(
        documents=["MCP 进来的块"],
        ids=["mcp_1"],
        metadatas=[{"source": "E:/some/dir/无分类文档.txt", "chunk_index": 0}],
    )

    result = _run(remove_uploaded_file("无分类文档.txt", store))

    assert result["found"] is False
    assert store.count() == 1          # 原数据还在


# ── 统计：让污染看得见 ────────────────────────────────────────


def test_上传块数被单独数出来(store):
    _add_crawled(store, n=3)
    _add_upload(store, n=2)

    stats = get_knowledge_base_stats(store)

    assert stats.total_chunks == 5
    assert stats.uploaded_chunks == 2
    assert stats.unaccounted_chunks == 0


def test_无category的块落进差额并向上报(store):
    """MCP / 旧路径写进来的块没有 category 字段 —— 必须体现为差额，不许静默。"""
    _add_crawled(store, n=3)
    _add_upload(store, n=2)
    store.save_documents(
        documents=["来源不明的块"] * 4,
        ids=[f"ghost_{i}" for i in range(4)],
        metadatas=[{"chunk_index": i} for i in range(4)],   # 没有 category
    )

    stats = get_knowledge_base_stats(store)

    assert stats.total_chunks == 9
    assert stats.uploaded_chunks == 2
    assert stats.unaccounted_chunks == 4      # ← 这就是"库被污染了"的信号


def test_分类数加上传加差额恒等于总数(store):
    """这个恒等式是监控口径：三个数不闭合就说明有内容躲过了统计。"""
    _add_crawled(store, "通知A", "通知公告", n=3)
    _add_crawled(store, "新闻B", "学校新闻", n=2)
    _add_upload(store, n=4)
    store.save_documents(
        documents=["野块"],
        ids=["wild_1"],
        metadatas=[{"source": "野块.txt"}],
    )

    stats = get_knowledge_base_stats(store)

    assert (
        sum(stats.category_counts.values())
        + stats.uploaded_chunks
        + stats.unaccounted_chunks
        == stats.total_chunks
    )


def test_空库统计不炸(store):
    stats = get_knowledge_base_stats(store)

    assert stats.total_chunks == 0
    assert stats.uploaded_chunks == 0
    assert stats.unaccounted_chunks == 0
    assert stats.category_counts == {}


def test_用户上传不是爬虫分类(store):
    """钉住这个前提：UPLOAD_CATEGORY 一旦进了 CATEGORIES，上面的差额算法就错了
    （上传块会被 category_counts 数一遍、又被 uploaded_chunks 数一遍）。"""
    from campus_scraper.config import CATEGORIES

    assert UPLOAD_CATEGORY not in {c["name"] for c in CATEGORIES}


# ── rebuild_kb：上传的 .txt 不许混进消融语料 ──────────────────


def test_消融重建排除_uploads目录(tmp_path):
    """上传的 .txt 若被当成"官网文章"重建进库，消融实验的数字就全废了。"""
    from rebuild_kb import corpus_txt_files

    (tmp_path / "通知公告").mkdir()
    (tmp_path / "通知公告" / "端午.txt").write_text("正文", encoding="utf-8")
    (tmp_path / "_uploads").mkdir()
    (tmp_path / "_uploads" / "上传的.txt").write_text("上传正文", encoding="utf-8")

    found = corpus_txt_files(tmp_path)

    assert [p.name for p in found] == ["端午.txt"]


def test_消融重建在没语料时返回空(tmp_path):
    from rebuild_kb import corpus_txt_files

    assert corpus_txt_files(tmp_path) == []


# ── is_valid_source_name：删除目标的名字校验 ──────────────────
# 这个判断栽过一次：最早写的是 `Path(name).name != name`，把带斜杠的爬虫
# source 全判成"非法"→ 400，那道 403 安全闸门永远够不着、成了摆设。
# 线上 1194 篇语料的 source **全部**带斜杠（`分类/标题.txt`），
# 所以"有斜杠就拒"是错的。下面把边界钉死。


@pytest.mark.parametrize("name", [
    "测试上传.pdf",                      # 上传的裸文件名
    "团委新闻/2024年暑期社会实践活动.txt",   # 爬虫语料的真实形状，必须放行
    "通知公告/关于2026年端午节放假的通知.txt",
    "a/b/c.txt",
])
def test_合法source名放行(name):
    from app_backend import is_valid_source_name

    assert is_valid_source_name(name) is True


@pytest.mark.parametrize("name", [
    "",
    "/etc/passwd",              # 绝对路径
    "..",                       # 纯上跳
    "../secret.txt",
    "a/../../etc/passwd",       # 夹在中间的穿越
    "..\\windows\\system32",    # 反斜杠
    "C:\\Windows\\win.ini",
])
def test_危险source名拒绝(name):
    from app_backend import is_valid_source_name

    assert is_valid_source_name(name) is False


def test_真实语料里每一条source都能过校验():
    """拿线上库的真 source 跑一遍 —— 参数化列表是我猜的，这条才是事实。

    如果这条挂了，说明校验把真语料误判了，403 闸门又会变成够不着的摆设。
    """
    from app_backend import is_valid_source_name

    vs = _real_store_or_skip()
    got = vs.collection.get()
    sources = {(m or {}).get("source") for m in (got.get("metadatas") or []) if m}
    sources.discard(None)
    assert sources, "线上库没读到 source，测试前提不成立"

    bad = sorted(s for s in sources if not is_valid_source_name(s))
    assert bad == [], f"{len(bad)} 条真实 source 被误判为非法: {bad[:5]}"


def _real_store_or_skip():
    """读线上库只读校验用；库不存在就跳过（CI 上没有 chroma_db）。"""
    from pathlib import Path

    from src.vector_store import VectorStore

    db = Path(__file__).resolve().parent.parent / "chroma_db"
    if not db.exists():
        pytest.skip("没有线上 chroma_db，跳过")
    return VectorStore(db_path=str(db), collection_name="my_rag_collection")


# ── delete_uploaded_file：路由分流的 status 字段 ──────────────


def test_status_字段_找不到时是not_found(monkeypatch, store):
    from app_backend import delete_uploaded_file
    monkeypatch.setattr("app_backend._get_vector_store", lambda: store)

    assert _run(delete_uploaded_file("没有这个.pdf"))["status"] == "not_found"


def test_status_字段_删成功时是ok(monkeypatch, store):
    from app_backend import delete_uploaded_file
    monkeypatch.setattr("app_backend._get_vector_store", lambda: store)
    _add_upload(store, "删我.pdf", n=3)

    result = _run(delete_uploaded_file("删我.pdf"))

    assert result["status"] == "ok"
    assert result["deleted_chunks"] == 3


def test_status_字段_爬虫语料是rejected而非抛异常(monkeypatch, store):
    """路由靠 status 分流到 403。这里若抛出去，请求会变成 500 —— 观感完全不同。"""
    from app_backend import delete_uploaded_file
    monkeypatch.setattr("app_backend._get_vector_store", lambda: store)
    _add_crawled(store, "通知公告/端午.txt", "通知公告", n=2)

    result = _run(delete_uploaded_file("通知公告/端午.txt"))

    assert result["status"] == "rejected"
    assert "不是用户上传" in result["detail"]
    assert store.count_by_source("通知公告/端午.txt") == 2    # 数据还在


def test_status_字段_底层炸了是failed而非抛异常(monkeypatch, store):
    from app_backend import delete_uploaded_file

    class _Boom:
        def __getattr__(self, _):
            raise RuntimeError("chroma 挂了")

    monkeypatch.setattr("app_backend._get_vector_store", lambda: _Boom())

    result = _run(delete_uploaded_file("随便.pdf"))

    assert result["status"] == "failed"


# ── 异步入口包装 ──────────────────────────────────────────────
# remove_uploaded_file 是 async（要和 ingest_uploaded_files 对称），
# 但内部全是同步 Chroma 操作。用 asyncio.run 驱动，不引 pytest-asyncio
# （.venv 没装，照 tests/test_recall_guard.py 的做法）。


def _run(coro):
    import asyncio
    return asyncio.run(coro)
