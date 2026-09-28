

from sqlmodel import Field,SQLModel

class User(SQLModel,table=True):
    __tablename__ = "users"
    id:int|None=Field(default=None,primary_key=True)
    username:str=Field(min_length=2,max_length=50,unique=True,index=True,nullable=False)
    hashed_password:str=Field(min_length=8,nullable=False)
    vip_level:int=Field(default=0,nullable=False)
    avatar:str|None=Field(default=None,nullable=True)
    email:str=Field(nullable=False,unique=True)