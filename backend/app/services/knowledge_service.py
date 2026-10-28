"""协调文件、关系表和向量库，管理导入状态、失败清理与检索策略。"""

import asyncio
import logging
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from dataclasses import replace

from fastapi import UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.infrastructure.database.database import engine
from app.models.knowledge_book import KnowledgeBook
from app.rag.embeddings import EmbeddingService
from app.rag.milvus_store import MilvusBookStore
from app.rag.parser import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, parse_book, split_book
from app.rag.query import asks_for_numbered_list
from app.rag.reranker import Reranker
from app.rag.types import RetrievedChunk
from app.repositories.knowledge_book_repository import KnowledgeBookRepository


logger = logging.getLogger(__name__)
MAX_EXPANDED_CHUNKS = 24
MIN_EXPANSION_SCORE = 0.70


class KnowledgeService:
    """协调上传、导入、清理、重试与书籍检索。"""
    def __init__(self, retrieval_mode: str | None = None, *, embedding_max_retries: int | None = None):
        """装配书籍关系仓储和向量存储；两者通过 user_id/book_id 关联。"""
        self.books = KnowledgeBookRepository()
        self.vectors = MilvusBookStore()
        self.reranker = Reranker()
        # 图内检索将 SDK 重试设为零，避免与节点重试相乘；导入沿用独立配置。
        self.embedding_max_retries = embedding_max_retries
        self.retrieval_mode = retrieval_mode or settings.RAG_RETRIEVAL_MODE
        if self.retrieval_mode not in ("dense", "hybrid", "hybrid_rerank"):
            raise ValueError("不支持的检索模式")

    async def upload(self, session: AsyncSession, user_id: int, file: UploadFile,
                     title: str | None = None) -> KnowledgeBook:
        """校验格式、大小和哈希，保存原件并创建 processing 记录；数据库失败时清理文件。"""
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
        """使用独立会话读取原件、解析切分、生成向量并写入，最后标记 ready 或 failed。"""
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
        """重置失败或已完成记录为 processing，后台任务随后重新导入。"""
        if book.status == "processing":
            raise ValueError("书籍正在处理")
        book.status = "processing"
        book.error = None
        book.chunk_count = 0
        return await self.books.update(session, book)

    async def delete(self, session: AsyncSession, book: KnowledgeBook) -> None:
        """禁止删除处理中书籍，依次删除向量、关系记录与文件。"""
        if book.status == "processing":
            raise ValueError("书籍正在处理，完成后再删除")
        await self.vectors.delete_book(book.user_id, book.id)
        await self.books.delete(session, book)
        Path(book.storage_path).unlink(missing_ok=True)

    async def search(self, session: AsyncSession, user_id: int, question: str,
                     limit: int = 5) -> list[RetrievedChunk]:
        """只查当前用户 ready 书籍；两路召回、RRF、可选重排，再选最终证据。

        dense 用余弦阈值作为基线；hybrid 只融合；hybrid_rerank 使用独立重排阈值。
        数量问题的短书补取先合并候选再重排；保留已命中短书的有界上下文，
        避免最后只取 top-k 再次丢失跨页条目。扩展结果可能超过 limit。
        """
        if not 1 <= limit <= 100:
            raise ValueError("检索返回数量应为 1–100")
        book_ids = await self.books.list_ready_ids(session, user_id)
        if not book_ids:
            return []
        embedding = (EmbeddingService() if self.embedding_max_retries is None
                     else EmbeddingService(max_retries=self.embedding_max_retries))
        query_vector = (await embedding.embed([question]))[0]
        if self.retrieval_mode != "dense":
            candidates = await self.vectors.hybrid_search(user_id, query_vector, question, book_ids)
            candidates = await self._expand_candidates(session, user_id, question, query_vector, candidates)
            if self.retrieval_mode == "hybrid_rerank":
                ranked = await self.reranker.rerank(question, candidates)
                qualified = [hit for hit in ranked if hit.score >= settings.RERANK_MIN_SCORE]
            else:
                ranked = candidates
                # RRF 的量纲不是余弦相似度，不使用旧 0.55 阈值。
                qualified = [hit for hit in ranked if hit.score_type == "rrf"]
            selected = qualified[:limit]
            selected_ids = {hit.chunk.id for hit in selected}
            selected_books = {hit.chunk.book_id for hit in selected}
            neighbors = [hit for hit in ranked if hit.is_expansion
                         and hit.chunk.book_id in selected_books and hit.chunk.id not in selected_ids]
            # 低分扩展片段是同书上下文，不冒充已经通过相关性阈值的独立命中。
            neighbors.sort(key=lambda hit: (hit.chunk.book_id, hit.chunk.page if hit.chunk.page > 0 else -hit.chunk.page))
            return [*selected, *neighbors]

        hits = await self.vectors.search(user_id, query_vector, limit, book_ids)
        relevant = [hit for hit in hits if hit.score >= settings.RAG_MIN_SCORE]
        if not relevant or not asks_for_numbered_list(question):
            return relevant

        # A short document may spread a numbered list across many pages. The
        # global top-k can omit later items even when its first hit is strong.
        top = relevant[0]
        if top.score < MIN_EXPANSION_SCORE:
            return relevant
        book = await self.books.get_owned(session, top.chunk.book_id, user_id)
        if (book is None or book.status != "ready"
                or not 1 < book.chunk_count <= MAX_EXPANDED_CHUNKS):
            return relevant
        expanded = await self.vectors.search(user_id, query_vector, book.chunk_count, [book.id])
        if len(expanded) != book.chunk_count:
            return relevant
        return sorted(expanded, key=lambda hit: (hit.chunk.page, -hit.score))

    async def _expand_candidates(self, session: AsyncSession, user_id: int, question: str,
                                  vector: list[float], hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """强语义命中的数量问题补取短书，按 ID 合并后整体交给重排。

        强命中使用单独的 dense_score，不能拿 RRF 小数与旧余弦阈值比较。
        数据不全时回到原候选；原候选与补取重叠时保留原融合分数并标记上下文。
        """
        if not hits or not asks_for_numbered_list(question):
            return hits
        top = max(hits, key=lambda hit: hit.dense_score if hit.dense_score is not None else -1)
        if top.dense_score is None or top.dense_score < MIN_EXPANSION_SCORE:
            return hits
        book = await self.books.get_owned(session, top.chunk.book_id, user_id)
        if book is None or book.status != "ready" or not 1 < book.chunk_count <= MAX_EXPANDED_CHUNKS:
            return hits
        expanded = await self.vectors.book_chunks(user_id, book.id, vector)
        if len(expanded) != book.chunk_count or len({hit.chunk.id for hit in expanded}) != book.chunk_count:
            return hits
        expanded_ids = {hit.chunk.id for hit in expanded}
        merged = {hit.chunk.id: replace(hit, is_expansion=hit.chunk.id in expanded_ids) for hit in hits}
        for hit in expanded:
            merged.setdefault(hit.chunk.id, replace(hit, is_expansion=True))
        return list(merged.values())


async def mark_interrupted_imports() -> None:
    """启动时将遗留的 processing 记录标记 failed，用户可明确重试。"""
    repository = KnowledgeBookRepository()
    async with AsyncSession(engine) as session:
        for book in await repository.list_processing(session):
            book.status = "failed"
            book.error = "服务重启中断了导入，请重试"
            await repository.update(session, book)
