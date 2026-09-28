from pydantic import EmailStr, Field  #校验邮箱格式
from sqlmodel import SQLModel


class UserCreate(SQLModel):
    username: str = Field(min_length=2, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    email: EmailStr
    avatar: str | None = None


class UserRead(SQLModel):
    id: int
    username: str
    email: EmailStr
    vip_level: int
    avatar: str | None
