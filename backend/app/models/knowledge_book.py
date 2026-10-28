"""书籍关系表：保存文件位置、所有者、哈希及向量导入状态。"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel


class KnowledgeBook(SQLModel, table=True):
    """记录书籍归属、原文件路径和 processing/ready/failed 状态。"""
    __tablename__ = "knowledge_books"

    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    title: str = Field(max_length=200, nullable=False)
    filename: str = Field(max_length=255, nullable=False)
    storage_path: str = Field(nullable=False)
    sha256: str = Field(max_length=64, index=True, nullable=False)
    status: str = Field(default="processing", max_length=20, nullable=False)
    chunk_count: int = Field(default=0, nullable=False)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
