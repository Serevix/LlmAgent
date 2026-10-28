"""会话业务：创建、查询、更新时间及检查资源是否属于当前用户。"""

from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.conversation import Conversation
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import ConversationCreate


class ConversationService:
    """协调会话仓储并校验当前用户对会话的所有权。"""
    def __init__(self):
        """创建会话仓储对象，复用调用者的数据库会话。"""
        self.repository = ConversationRepository()

    async def create(self, session: AsyncSession,user_id:int, data: ConversationCreate) -> Conversation:
        """创建属于当前用户的会话并交给仓储提交。"""
        conversation = Conversation(user_id=user_id,title=data.title)
        return await self.repository.create(session, conversation)

    async def get(self, session: AsyncSession, conversation_id: int) -> Conversation:
        """按编号查询，不存在时抛出业务异常。"""
        conversation = await self.repository.get_by_id(session, conversation_id)

        if conversation is None:
            raise ValueError("会话不存在")

        return conversation

    async def list_by_user(self, session: AsyncSession, user_id: int) -> list[Conversation]:
        """返回指定用户拥有的会话列表。"""
        return await self.repository.get_by_user(session, user_id)

    async def touch(self, session: AsyncSession, conversation: Conversation) -> Conversation:
        """通过仓储更新会话的最近使用时间。"""
        return await self.repository.touch(session, conversation)

    async def get_owned(self,session:AsyncSession,conversation_id:int,user_id:int)->Conversation:
        """在查询后核对用户归属；统一返回不存在，避免泄露他人会话。"""
        conversation=await self.get(session,conversation_id)
        if conversation.user_id!=user_id:
            raise ValueError("会话不存在")
        return conversation