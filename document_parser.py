import pypdf
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

