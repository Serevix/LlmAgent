"""按用户隔离模型配置，只向业务层提供记录，不暴露加密凭据。"""

from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.model_profile import ModelProfile


class ModelProfileRepository:
    """自定义模型的持久化入口，所有读取及删除都同时过滤 user_id。"""

    async def list_owned(self, session: AsyncSession, user_id: int) -> list[ModelProfile]:
        """按创建时间返回当前用户的配置列表。"""
        result = await session.exec(select(ModelProfile).where(ModelProfile.user_id == user_id)
                                    .order_by(col(ModelProfile.created_at).asc()))
        return list(result.all())

    async def get_owned(self, session: AsyncSession, profile_id: str, user_id: int) -> ModelProfile | None:
        """未找到或属于其他用户时统一返回 None。"""
        return (await session.exec(select(ModelProfile).where(
            ModelProfile.id == profile_id, ModelProfile.user_id == user_id))).first()

    async def create(self, session: AsyncSession, profile: ModelProfile) -> ModelProfile:
        """提交加密后的配置并刷新数据库字段。"""
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        return profile

    async def delete(self, session: AsyncSession, profile: ModelProfile) -> None:
        """仅删除配置；不会删除会话或改变历史消息。"""
        await session.delete(profile)
        await session.commit()
