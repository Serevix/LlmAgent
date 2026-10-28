"""认证依赖：从请求的 Bearer token 中解析并查询当前用户。"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import decode_access_token
from app.infrastructure.database.database import get_session
from app.models.user import User
from app.repositories.user_repository import UserRepository


bearer_scheme = HTTPBearer()  #从 HTTP 请求的 Authorization 头中解析 Bearer Token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    """验证 Bearer token，解析用户编号并查询用户；失败时拒绝请求。"""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录状态无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError:
        raise credentials_error

    user = await UserRepository().get_by_id(session, user_id)

    if user is None:
        raise credentials_error

    return user