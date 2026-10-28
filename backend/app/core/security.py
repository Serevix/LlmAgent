"""密码哈希和 JWT 签名验证；数据库只保存密码哈希。"""

from pwdlib import PasswordHash #负责密码 Hash 和密码验证。
from datetime import datetime,timedelta,timezone

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings


password_hash = PasswordHash.recommended() #使用 pwdlib 当前推荐的密码 Hash 配置。


def hash_password(password: str) -> str:
    """将明文密码转换为不可逆哈希，写库时只使用返回值。"""
    return password_hash.hash(password)


def verify_password(plain_password: str,hashed_password: str,) -> bool:
    """用密码库验证输入密码是否匹配数据库中的哈希。"""
    return password_hash.verify(
        plain_password,
        hashed_password,
    )

def create_access_token(user_id:int)->str:
    """将用户编号与过期时间签入 JWT，供前端后续请求使用。"""
    expire=datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload={"sub":str(user_id),"exp":expire}
    return jwt.encode(payload,settings.JWT_SECRET_KEY,algorithm=settings.JWT_ALGORITHM)

def decode_access_token(token:str)->int:
    """验证签名、过期时间并提取用户编号；无效 token 转为业务异常。"""
    try:
        payload=jwt.decode(token,settings.JWT_SECRET_KEY,algorithms=[settings.JWT_ALGORITHM])
        return int(payload.get("sub"))
    except (InvalidTokenError,KeyError,ValueError) as exc:
        raise ValueError("无效或已过期的Token") from exc