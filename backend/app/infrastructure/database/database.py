from collections.abc import AsyncGenerator

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

engine:AsyncEngine =create_async_engine(
    settings.DATABASE_URL
)
async def get_session()->AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine) as session:
        yield session