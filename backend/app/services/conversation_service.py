from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.conversation import Conversation
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import ConversationCreate


class ConversationService:
    def __init__(self):
        self.repository = ConversationRepository()

    async def create(self, session: AsyncSession,user_id:int, data: ConversationCreate) -> Conversation:
        conversation = Conversation(user_id=user_id,title=data.title)
        return await self.repository.create(session, conversation)

    async def get(self, session: AsyncSession, conversation_id: int) -> Conversation:
        conversation = await self.repository.get_by_id(session, conversation_id)

        if conversation is None:
            raise ValueError("会话不存在")

        return conversation

    async def list_by_user(self, session: AsyncSession, user_id: int) -> list[Conversation]:
        return await self.repository.get_by_user(session, user_id)

    async def touch(self, session: AsyncSession, conversation: Conversation) -> Conversation:
        return await self.repository.touch(session, conversation)

    async def get_owned(self,session:AsyncSession,conversation_id:int,user_id:int)->Conversation:
        conversation=await self.get(session,conversation_id)
        if conversation.user_id!=user_id:
            raise ValueError("会话不存在")
        return conversation