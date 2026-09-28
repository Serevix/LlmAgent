from pwdlib import PasswordHash #负责密码 Hash 和密码验证。
from datetime import datetime,timedelta,timezone

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings


password_hash = PasswordHash.recommended() #使用 pwdlib 当前推荐的密码 Hash 配置。


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str,hashed_password: str,) -> bool:
    return password_hash.verify(
        plain_password,
        hashed_password,
    )

def create_access_token(user_id:int)->str:
    expire=datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload={"sub":str(user_id),"exp":expire}
    return jwt.encode(payload,settings.JWT_SECRET_KEY,algorithm=settings.JWT_ALGORITHM)

def decode_access_token(token:str)->int:
    try:
        payload=jwt.decode(token,settings.JWT_SECRET_KEY,algorithms=[settings.JWT_ALGORITHM])
        return int(payload.get("sub"))
    except (InvalidTokenError,KeyError,ValueError) as exc:
        raise ValueError("无效或已过期的Token") from exc