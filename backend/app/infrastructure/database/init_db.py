import asyncio

from sqlalchemy import text
from sqlmodel import SQLModel

from app.infrastructure.database.database import engine
from app.models.user import User
from app.models.conversation import Conversation
from app.models.knowledge_book import KnowledgeBook


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        # create_all does not update columns in an existing table. Earlier
        # databases made avatar NOT NULL although registration treats it as optional.
        if conn.dialect.name == "postgresql":
            await conn.execute(text("ALTER TABLE users ALTER COLUMN avatar DROP NOT NULL"))


if __name__ == "__main__":
    asyncio.run(init_db())
