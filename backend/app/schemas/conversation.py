"""会话创建请求与列表响应的数据格式。"""

from datetime import datetime

from pydantic import BaseModel,Field

class ConversationCreate(BaseModel):
    """新建会话的标题输入。"""
    title:str=Field(default="新对话",max_length=200)

class ConversationRead(BaseModel):
    """会话列表信息，不含 checkpoint 中的全部聊天内容。"""
    id: int
    user_id: int
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime