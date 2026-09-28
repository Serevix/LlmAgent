from pydantic import BaseModel,Field

class MemoryCreate(BaseModel):
    user_id:int
    content:str=Field(min_length=1,max_length=1000)


class MemoryRead(BaseModel):
    key:str
    content:str
