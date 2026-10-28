"""长期记忆创建与读取的接口数据格式。"""

from pydantic import BaseModel,Field

class MemoryCreate(BaseModel):
    """手动创建长期记忆的输入；接口还会核对用户身份。"""
    user_id:int
    content:str=Field(min_length=1,max_length=1000)


class MemoryRead(BaseModel):
    """长期记忆读取结果，包含记录键与内容。"""
    key:str
    content:str
