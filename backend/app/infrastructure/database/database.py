"""创建共享连接池；每次请求单独使用异步数据库会话。"""

from collections.abc import AsyncGenerator

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

engine:AsyncEngine =create_async_engine(
    settings.DATABASE_URL
)
async def get_session()->AsyncGenerator[AsyncSession, None]:
    """每次请求创建独立 AsyncSession，yield 给依赖使用，结束时自动关闭。"""
    async with AsyncSession(engine) as session:
        yield session
