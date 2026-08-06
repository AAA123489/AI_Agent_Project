import pypdf
from docx import Document
from pypdf import PasswordType


def parse_pdf(file_path: str) -> tuple[str, dict]:
    reader = pypdf.PdfReader(file_path)
    if reader.is_encrypted:
        decrypt_status = reader.decrypt("")
        if decrypt_status == PasswordType.NOT_DECRYPTED:
            raise ValueError(f"PDF文件 '{file_path}' 已加密且无法解密，跳过处理。")
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text is None:
            text = ""
        full_text += text
    if reader.metadata is not None:
        author = reader.metadata.get("/Author", "unknown")
        creation_date_str = reader.metadata.get("/CreationDate", "")
        year = creation_date_str[2:6] if len(creation_date_str) >= 6 else ""
    else:
        author = "unknown"
        year = ""
    return (full_text, {"author": author, "year": year})


def parse_docx(file_path: str) -> tuple[str, dict]:
    """解析 .docx 文件，返回 (文本, 元数据)。"""
    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

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

