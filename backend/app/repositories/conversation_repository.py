"""会话数据库操作；不在这一层处理模型调用或聊天历史。"""

from sqlmodel import select,col
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import datetime,timezone

from app.models.conversation import Conversation


class ConversationRepository:
    """集中处理会话关系表的查询和提交。"""
    async def create(self, session: AsyncSession, conversation: Conversation) -> Conversation:
        """保存会话，提交事务并刷新数据库生成的字段。"""
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation

    async def get_by_id(self, session: AsyncSession, conversation_id: int) -> Conversation | None:
        """按公开会话编号读取；所有权检查由业务服务负责。"""
        return (await session.exec(select(Conversation).where(Conversation.id == conversation_id))).first()

    async def get_by_user(self, session: AsyncSession, user_id: int) -> list[Conversation]:
        """按用户筛选并按最近更新时间排序，返回会话列表。"""
        statement = (select(Conversation)
                     .where(Conversation.user_id == user_id)
                     .order_by(col(Conversation.updated_at).desc())
        )
        return list((await session.exec(statement)).all())

    async def touch(self,session:AsyncSession,conversation:Conversation)->Conversation:
        """更新会话时间并提交，使最近使用的会话排在前面。"""
        conversation.updated_at=datetime.now(timezone.utc)
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation