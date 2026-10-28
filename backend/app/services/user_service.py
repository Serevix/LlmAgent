"""注册业务：检查重复账号，将密码哈希后保存用户。"""

from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import hash_password
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate


class UserService:

    """协调用户查重、密码哈希和新用户持久化。"""
    def __init__(self):
        """创建用户仓储，注册流程通过它访问关系数据库。"""
        self.repository = UserRepository()

    async def create_user(
        self,
        session: AsyncSession,
        data: UserCreate,
    ) -> User:

        """检查用户名和邮箱重复，哈希密码后保存用户，返回持久化对象。"""
        existing_user = await self.repository.get_by_username(session,data.username,)

        if existing_user:
            raise ValueError("用户名已经存在")

        existing_email = await self.repository.get_by_email(
            session,
            str(data.email),
        )

        if existing_email:
            raise ValueError("邮箱已经存在")

        user = User(
            username=data.username,
            hashed_password=hash_password(data.password),
            email=str(data.email),
            avatar=data.avatar,
        )

        return await self.repository.create(
            session,
            user,
        )
