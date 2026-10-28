"""用户数据库操作：按唯一账号字段查询，提交并刷新新用户。"""

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.user import User


class UserRepository:

    """集中处理用户查询和写入，不承担密码校验逻辑。"""
    async def get_by_username(self,session: AsyncSession,username: str,) -> User | None:

        """按用户名查询用户，用于查重或登录。"""
        statement = select(User).where(User.username == username)

        result = await session.exec(statement)

        return result.first()

    async def get_by_email(
        self,
        session: AsyncSession,
        email: str,
    ) -> User | None:

        """按邮箱查询用户，用于查重或邮箱登录。"""
        statement = select(User).where(
            User.email == email
        )

        result = await session.exec(statement)

        return result.first()

    async def create(self,session: AsyncSession,user: User,) -> User:

        """提交新用户并刷新，取得数据库生成的用户主键。"""
        session.add(user)

        await session.commit()

        await session.refresh(user)

        return user

    async def get_by_id(self,session: AsyncSession,user_id: int,) -> User | None:

        """按用户主键查询，供 token 解析后的身份验证使用。"""
        return await session.get(User,user_id)
