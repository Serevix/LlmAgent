"""生成文件记录：文件归属、会话关联和服务器生成的存储位置。"""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel
from app.models.user import User  # noqa: F401：独立调用生成工具时也注册外键引用的用户表。


class GeneratedArtifact(SQLModel, table=True):
    """保存图片或 Word 的元数据；实际文件存放在生成文件目录。"""
    __tablename__ = "generated_artifacts"

    id: str = Field(primary_key=True, max_length=32)
    user_id: int = Field(foreign_key="users.id", index=True)
    thread_id: str = Field(index=True, max_length=100)
    request_key: str = Field(unique=True, max_length=64)
    kind: str = Field(max_length=16)
    title: str = Field(max_length=120)
    filename: str = Field(max_length=180)
    storage_path: str
    media_type: str = Field(max_length=100)
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
