from datetime import datetime, timezone

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.knowledge_book import KnowledgeBook


class KnowledgeBookRepository:
    async def create(self, session: AsyncSession, book: KnowledgeBook) -> KnowledgeBook:
        session.add(book)
        await session.commit()
        await session.refresh(book)
        return book

    async def get_owned(self, session: AsyncSession, book_id: str, user_id: int) -> KnowledgeBook | None:
        return (await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.id == book_id, KnowledgeBook.user_id == user_id))).first()

    async def get_by_hash(self, session: AsyncSession, user_id: int, sha256: str) -> KnowledgeBook | None:
        return (await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id, KnowledgeBook.sha256 == sha256))).first()

    async def list_by_user(self, session: AsyncSession, user_id: int) -> list[KnowledgeBook]:
        return list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id).order_by(KnowledgeBook.created_at.desc()))).all())

    async def list_ready_ids(self, session: AsyncSession, user_id: int) -> list[str]:
        return list((await session.exec(select(KnowledgeBook.id).where(
            KnowledgeBook.user_id == user_id, KnowledgeBook.status == "ready"))).all())

    async def list_processing(self, session: AsyncSession) -> list[KnowledgeBook]:
        return list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.status == "processing"))).all())

    async def update(self, session: AsyncSession, book: KnowledgeBook) -> KnowledgeBook:
        book.updated_at = datetime.now(timezone.utc)
        session.add(book)
        await session.commit()
        await session.refresh(book)
        return book

    async def delete(self, session: AsyncSession, book: KnowledgeBook) -> None:
        await session.delete(book)
        await session.commit()
