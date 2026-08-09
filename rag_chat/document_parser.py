import logging

import pypdf
from docx import Document
from pypdf import PasswordType

logger = logging.getLogger(__name__)

# RapidOCR 引擎懒加载缓存：首次遇到图片型 PDF 时加载（约 1~3 秒），后续复用。
_ocr_engine = None


def _get_ocr_engine():
    """获取缓存的 RapidOCR 引擎；未安装时抛出带提示的 ImportError。"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "此 PDF 为图片型（无可提取文本），需要 OCR 能力：请先 pip install "
                "rapidocr-onnxruntime 后再试。"
            )
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_pdf_page(page, engine) -> str:
    """把 PDF 页渲染成图片并 OCR，返回识别文本；失败返回空串。"""
    try:
        import numpy as np
        pil_img = page.to_image(resolution=150).original
        result, _ = engine(np.array(pil_img.convert("RGB")))
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
    except Exception:
        logger.exception("PDF 页 OCR 失败")
        return ""


def parse_pdf(file_path: str) -> tuple[str, dict]:
    """解析 .pdf 文件，返回 (文本, 元数据)。

    文本用 pypdf 提取（保留原有加密处理与提取行为）；
    表格用 pdfplumber 结构化提取，每个表格按行展开为「单元格 | 单元格 | …」，
    用【表格N】标记插入正文。pdfplumber 对跨列合并的单元格会返回重复内容，
    因此逐行跳过与前一格相同的内容。
    """
    import pdfplumber

    reader = pypdf.PdfReader(file_path)
    if reader.is_encrypted:
        decrypt_status = reader.decrypt("")
        if decrypt_status == PasswordType.NOT_DECRYPTED:
            raise ValueError(f"PDF文件 '{file_path}' 已加密且无法解密，跳过处理。")

    parts: list[str] = []
    table_no = 0

    with pdfplumber.open(file_path, password="") as pdf:
        for page_idx, page in enumerate(pdf.pages):
            # 文本（用 pypdf 逐页提取，与旧行为一致）
            if page_idx < len(reader.pages):
                text = reader.pages[page_idx].extract_text()
                if text and text.strip():
                    parts.append(text.strip())
                else:
                    # 图片型页面（提取不到文本）：渲染 + OCR 兜底
                    engine = _get_ocr_engine()
                    ocr_text = _ocr_pdf_page(page, engine)
                    if ocr_text:
                        parts.append(f"【OCR识别·第{page_idx + 1}页】\n" + ocr_text)

            # 表格（pdfplumber 结构化提取）
            for table in page.extract_tables():
                table_no += 1
                rows: list[str] = []
                for row in table:
                    cells: list[str] = []
                    prev = None
                    for cell in row:
                        txt = (cell or "").strip()
                        if txt == prev:
                            continue
                        prev = txt
                        cells.append(txt)
                    row_text = " | ".join(c for c in cells if c)
                    if row_text:
                        rows.append(row_text)
                if rows:
                    parts.append(f"【表格{table_no}】\n" + "\n".join(rows))

    full_text = "\n\n".join(parts)

    if reader.metadata is not None:
        author = reader.metadata.get("/Author", "unknown")
        creation_date_str = reader.metadata.get("/CreationDate", "")
        year = creation_date_str[2:6] if len(creation_date_str) >= 6 else ""
    else:
        author = "unknown"
        year = ""
    return (full_text, {"author": author, "year": year})


def parse_docx(file_path: str) -> tuple[str, dict]:
    """解析 .docx 文件，返回 (文本, 元数据)。

    段落与表格都提取：每个表格按行展开为「单元格1 | 单元格2 | …」，
    用【表格N】标记块插入正文。合并单元格会让 row.cells 返回重复引用，
    因此逐行跳过与前一格相同的内容。
    """
    doc = Document(file_path)

    parts: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]

    for idx, table in enumerate(doc.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells: list[str] = []
            prev = None
            for cell in row.cells:
                txt = cell.text.strip()
                if txt == prev:
                    continue
                prev = txt
                cells.append(txt)
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                rows.append(row_text)
        if rows:
            parts.append(f"【表格{idx}】\n" + "\n".join(rows))

    full_text = "\n\n".join(parts)

    author = "unknown"
    year = ""
    if doc.core_properties:
        author = doc.core_properties.author or "unknown"
        if doc.core_properties.created:
            year = str(doc.core_properties.created.year)

    return (full_text, {"author": author, "year": year})


def parse_html(file_path: str) -> tuple[str, dict]:
    """解析 .html/.htm 文件，返回 (文本, 元数据)。
    委托给 campus_scraper.html_parser 实现 HTML 清洗。
    """
    from pathlib import Path
    from campus_scraper.html_parser import parse_html as _parse

    html = Path(file_path).read_text(encoding="utf-8", errors="replace")
    text, meta = _parse(html, url=f"file://{file_path}")
    return text, meta

