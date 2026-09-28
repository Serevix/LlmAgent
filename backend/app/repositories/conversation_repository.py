from sqlmodel import select,col
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import datetime,timezone

from app.models.conversation import Conversation


class ConversationRepository:
    async def create(self, session: AsyncSession, conversation: Conversation) -> Conversation:
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation

    async def get_by_id(self, session: AsyncSession, conversation_id: int) -> Conversation | None:
        return (await session.exec(select(Conversation).where(Conversation.id == conversation_id))).first()

    async def get_by_user(self, session: AsyncSession, user_id: int) -> list[Conversation]:
        statement = (select(Conversation)
                     .where(Conversation.user_id == user_id)
                     .order_by(col(Conversation.updated_at).desc())
        )
        return list((await session.exec(statement)).all())

    async def touch(self,session:AsyncSession,conversation:Conversation)->Conversation:
        conversation.updated_at=datetime.now(timezone.utc)
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation