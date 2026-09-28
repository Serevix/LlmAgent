from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.user import User


class UserRepository:

    async def get_by_username(self,session: AsyncSession,username: str,) -> User | None:

        statement = select(User).where(User.username == username)

        result = await session.exec(statement)

        return result.first()

    async def get_by_email(
        self,
        session: AsyncSession,
        email: str,
    ) -> User | None:

        statement = select(User).where(
            User.email == email
        )

        result = await session.exec(statement)

        return result.first()

    async def create(self,session: AsyncSession,user: User,) -> User:

        session.add(user)

        await session.commit()

        await session.refresh(user)

        return user

    async def get_by_id(self,session: AsyncSession,user_id: int,) -> User | None:

        return await session.get(User,user_id)
