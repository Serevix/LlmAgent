from io import BytesIO
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.rag.types import BookChunk, PageText


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_CHUNKS = 4000


def parse_book(filename: str, contents: bytes) -> list[PageText]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 PDF、Markdown 和 TXT 文件")
    if not contents or len(contents) > MAX_FILE_BYTES:
        raise ValueError("文件必须为非空且不超过 25 MB")
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(contents))
            pages = [PageText(page=index + 1, text=page.extract_text() or "")
                     for index, page in enumerate(reader.pages)]
        except Exception as exc:
            raise ValueError("PDF 无法解析或已加密") from exc
    else:
        try:
            pages = [PageText(page=0, text=contents.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("文本文件需要使用 UTF-8 编码") from exc
    if not any(page.text.strip() for page in pages):
        raise ValueError("没有提取到文字；扫描版 PDF 需要先进行 OCR")
    return pages


def split_book(book_id: str, user_id: int, title: str, pages: list[PageText]) -> list[BookChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for index, text in enumerate(splitter.split_text(page.text), start=1):
            if text.strip():
                chunks.append(BookChunk(id=uuid4().hex, book_id=book_id,
                                        user_id=user_id, title=title,
                                        page=page.page if page.page else -index,
                                        text=text.strip()))
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"书籍切分后超过 {MAX_CHUNKS} 段，请拆分文件后导入")
    return chunks
