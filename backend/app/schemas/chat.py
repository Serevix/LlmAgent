from pydantic import BaseModel,Field
from app.schemas.knowledge import SourceRead

class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=10000)

class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[SourceRead] = []

class ChatMessageRead(BaseModel):
    role: str
    content: str
    sources: list[SourceRead] = []


class ChatHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[ChatMessageRead]
