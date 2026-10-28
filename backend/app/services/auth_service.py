"""登录业务：按用户名或邮箱查询用户，再验证密码哈希。"""

from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import verify_password
from app.models.user import User
from app.repositories.user_repository import UserRepository

class AuthService:
    """验证登录凭据，成功后返回用户对象。"""
    def __init__(self):
        """创建用户仓储对象；数据库会话由调用者传入。"""
        self.respository = UserRepository()

    async def authenticate(self,session:AsyncSession,username:str,password:str)->User:
        """根据账号格式选择用户名或邮箱查询，再校验密码，返回已验证用户。"""
        user=(await self.respository.get_by_email(session,username)
              if "@" in username else await self.respository.get_by_username(session,username))
        if user is None or not verify_password(password,user.hashed_password):
            raise ValueError("用户名或密码错误")

        return user
