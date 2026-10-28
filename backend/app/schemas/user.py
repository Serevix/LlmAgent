"""注册输入及用户输出格式；输入验证与数据库表定义分开。"""

from pydantic import EmailStr, Field  #校验邮箱格式
from sqlmodel import SQLModel


class UserCreate(SQLModel):
    """注册输入，校验账号、邮箱和密码字段。"""
    username: str = Field(min_length=2, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    email: EmailStr
    avatar: str | None = None


class UserRead(SQLModel):
    """可公开的用户信息，排除密码哈希。"""
    id: int
    username: str
    email: EmailStr
    vip_level: int
    avatar: str | None
