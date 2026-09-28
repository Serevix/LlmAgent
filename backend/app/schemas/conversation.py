from datetime import datetime

from pydantic import BaseModel,Field

class ConversationCreate(BaseModel):
    title:str=Field(default="新对话",max_length=200)

class ConversationRead(BaseModel):
    id: int
    user_id: int
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime