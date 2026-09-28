from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import verify_password
from app.models.user import User
from app.repositories.user_repository import UserRepository

class AuthService:
    def __init__(self):
        self.respository = UserRepository()

    async def authenticate(self,session:AsyncSession,username:str,password:str)->User:
        user=(await self.respository.get_by_email(session,username)
              if "@" in username else await self.respository.get_by_username(session,username))
        if user is None or not verify_password(password,user.hashed_password):
            raise ValueError("用户名或密码错误")

        return user
