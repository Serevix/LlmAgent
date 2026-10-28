"""聊天请求、最终答案和历史消息的接口数据格式。"""

from pydantic import BaseModel,Field
from app.schemas.knowledge import SourceRead
from app.schemas.artifact import ArtifactRead

class ChatRequest(BaseModel):
    """当前问题和公开会话编号；用户身份从认证依赖获取。"""
    conversation_id: int
    message: str = Field(min_length=1, max_length=10000)
    model_id: str | None = Field(default=None, max_length=100, description="模型配置编号；为空时使用服务端默认模型。")

class ChatResponse(BaseModel):
    """非流式接口返回的最终答案与已确认来源。"""
    conversation_id: int
    answer: str
    sources: list[SourceRead] = []
    artifacts: list[ArtifactRead] = Field(default_factory=list)

class ChatMessageRead(BaseModel):
    """历史中的一条消息，保留角色、文字及最终来源。"""
    role: str
    content: str
    sources: list[SourceRead] = []
    artifacts: list[ArtifactRead] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """公开会话编号与按顺序排列的历史消息。"""
    conversation_id: int
    messages: list[ChatMessageRead]
