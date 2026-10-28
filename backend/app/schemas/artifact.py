"""生成文件的公开信息；不向浏览器或模型暴露服务器路径。"""

from typing import Literal

from pydantic import BaseModel


class ArtifactRead(BaseModel):
    """聊天消息携带的附件描述，内容接口仍需登录并校验所有者。"""
    id: str
    kind: Literal["image", "word"]
    title: str
    filename: str
    media_type: str
    size_bytes: int
    download_url: str
