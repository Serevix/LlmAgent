"""用户的模型服务配置；API Key 加密保存，不进入会话历史或接口响应。"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel


class ModelProfile(SQLModel, table=True):
    """只保存接入参数，不上传或执行模型权重文件。"""

    __tablename__ = "model_profiles"
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, nullable=False)
    name: str = Field(max_length=100)
    model: str = Field(max_length=200)
    base_url: str = Field(max_length=800)
    encrypted_api_key: str
    context_window: int = 32768
    max_output_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    timeout: int = 60
    structured_output_method: str = "function_calling"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
