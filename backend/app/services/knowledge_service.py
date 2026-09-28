import asyncio
import logging
from hashlib import sha256
from pathlib import Path, PureWindowsPath

from fastapi import UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.infrastructure.database.database import engine
from app.models.knowledge_book import KnowledgeBook
from app.rag.embeddings import EmbeddingService
from app.rag.milvus_store import MilvusBookStore
from app.rag.parser import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, parse_book, split_book
from app.rag.types import RetrievedChunk
from app.repositories.knowledge_book_repository import KnowledgeBookRepository


logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self):
        self.books = KnowledgeBookRepository()
        self.vectors = MilvusBookStore()

    async def upload(self, session: AsyncSession, user_id: int, file: UploadFile,
                     title: str | None = None) -> KnowledgeBook:
        filename = PureWindowsPath(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("仅支持 PDF、Markdown 和 TXT 文件")
        contents = await file.read(MAX_FILE_BYTES + 1)
        if not contents or len(contents) > MAX_FILE_BYTES:
            raise ValueError("文件必须为非空且不超过 25 MB")
        digest = sha256(contents).hexdigest()
        if await self.books.get_by_hash(session, user_id, digest):
            raise ValueError("这本书已经导入；失败记录可使用重试功能")
        book_title = (title or Path(filename).stem).strip()
        if not book_title or len(book_title) > 200:
            raise ValueError("书名长度应为 1–200 个字符")

        book = KnowledgeBook(user_id=user_id, title=book_title, filename=filename,
                             storage_path="", sha256=digest)
        storage = settings.BOOK_STORAGE_DIR / str(user_id) / f"{book.id}{suffix}"
        storage.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(storage.write_bytes, contents)
        book.storage_path = str(storage)
        try:
            return await self.books.create(session, book)
        except Exception:
            storage.unlink(missing_ok=True)
            raise

    async def process(self, book_id: str, user_id: int) -> None:
        async with AsyncSession(engine) as session:
            book = await self.books.get_owned(session, book_id, user_id)
            if book is None or book.status != "processing":
                return
            try:
                contents = await asyncio.to_thread(Path(book.storage_path).read_bytes)
                pages = await asyncio.to_thread(parse_book, book.filename, contents)
                chunks = await asyncio.to_thread(split_book, book.id, user_id, book.title, pages)
                if not chunks:
                    raise ValueError("书籍没有可索引的文字")
                vectors = await EmbeddingService().embed([chunk.text for chunk in chunks])
                await self.vectors.delete_book(user_id, book.id)
                await self.vectors.insert(chunks, vectors)
                book.status = "ready"
                book.chunk_count = len(chunks)
                book.error = None
            except Exception as exc:
                logger.exception("书籍导入失败 book_id=%s user_id=%s", book_id, user_id)
                try:
                    await self.vectors.delete_book(user_id, book.id)
                except Exception:
                    logger.exception("清理失败导入的向量失败 book_id=%s", book_id)
                book.status = "failed"
                book.error = str(exc)[:500]
            except asyncio.CancelledError:
                logger.warning("书籍导入被取消 book_id=%s user_id=%s", book_id, user_id)
                book.status = "failed"
                book.error = "导入任务被中断，请重试"
                await self.books.update(session, book)
                raise
            await self.books.update(session, book)

    async def retry(self, session: AsyncSession, book: KnowledgeBook) -> KnowledgeBook:
        if book.status == "processing":
            raise ValueError("书籍正在处理")
        book.status = "processing"
        book.error = None
        book.chunk_count = 0
        return await self.books.update(session, book)

    async def delete(self, session: AsyncSession, book: KnowledgeBook) -> None:
        if book.status == "processing":
            raise ValueError("书籍正在处理，完成后再删除")
        await self.vectors.delete_book(book.user_id, book.id)
        await self.books.delete(session, book)
        Path(book.storage_path).unlink(missing_ok=True)

    async def search(self, session: AsyncSession, user_id: int, question: str,
                     limit: int = 5) -> list[RetrievedChunk]:
        book_ids = await self.books.list_ready_ids(session, user_id)
        if not book_ids:
            return []
        query_vector = (await EmbeddingService().embed([question]))[0]
        hits = await self.vectors.search(user_id, query_vector, limit, book_ids)
        return [hit for hit in hits if hit.score >= settings.RAG_MIN_SCORE]


async def mark_interrupted_imports() -> None:
    repository = KnowledgeBookRepository()
    async with AsyncSession(engine) as session:
        for book in await repository.list_processing(session):
            book.status = "failed"
            book.error = "服务重启中断了导入，请重试"
            await repository.update(session, book)
