from datetime import datetime,timezone
from uuid import uuid4

from sqlmodel import Field,SQLModel


class Conversation(SQLModel,table=True):
    __tablename__ = "conversations"
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    thread_id: str = Field(default_factory=lambda: uuid4().hex, unique=True, index=True, nullable=False)
    title: str = Field(default="新对话", max_length=200, nullable=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)