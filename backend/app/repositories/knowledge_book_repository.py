"""书籍数据库操作；所有权和就绪状态为向量检索提供过滤范围。"""

from datetime import datetime, timezone

from sqlmodel import select,col
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.knowledge_book import KnowledgeBook


class KnowledgeBookRepository:
    """集中处理书籍状态及所有权查询；不直接存取向量。"""
    async def create(self, session: AsyncSession, book: KnowledgeBook) -> KnowledgeBook:
        """提交新书籍记录并刷新默认字段。"""
        session.add(book)
        await session.commit()
        await session.refresh(book)
        return book

    async def get_owned(self, session: AsyncSession, book_id: str, user_id: int) -> KnowledgeBook | None:
        """同时匹配书籍与用户编号，无记录或非所有者时返回 None。"""
        return (await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.id == book_id, KnowledgeBook.user_id == user_id))).first()

    async def get_by_hash(self, session: AsyncSession, user_id: int, sha256: str) -> KnowledgeBook | None:
        """按用户和文件哈希查重，同一文件允许不同用户各自导入。"""
        return (await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id, KnowledgeBook.sha256 == sha256))).first()

    async def list_by_user(self, session: AsyncSession, user_id: int) -> list[KnowledgeBook]:
        """返回当前用户的书籍，按创建时间倒序排列。"""
        return list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id).order_by(col(KnowledgeBook.created_at).desc()))).all())

    async def list_ready_ids(self, session: AsyncSession, user_id: int) -> list[str]:
        """只返回导入成功的书籍编号，作为 Milvus 检索范围。"""
        return list((await session.exec(select(KnowledgeBook.id).where(
            KnowledgeBook.user_id == user_id, KnowledgeBook.status == "ready"))).all())

    async def list_processing(self, session: AsyncSession) -> list[KnowledgeBook]:
        """查找仍处于处理中状态的导入，用于服务重启后的中断标记。"""
        return list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.status == "processing"))).all())

    async def update(self, session: AsyncSession, book: KnowledgeBook) -> KnowledgeBook:
        """更新时间并提交状态、错误或切片数的变化。"""
        book.updated_at = datetime.now(timezone.utc)
        session.add(book)
        await session.commit()
        await session.refresh(book)
        return book

    async def delete(self, session: AsyncSession, book: KnowledgeBook) -> None:
        """删除书籍关系记录并提交；向量和文件清理由业务层协调。"""
        await session.delete(book)
        await session.commit()
