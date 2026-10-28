"""长期记忆的结构化决策：是否保存、记忆分类和提炼后的内容。"""

from typing import Literal

from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    """模型返回的长期记忆决策，约束分类与是否保存。"""
    should_save: bool = Field(description="当前用户消息是否包含值得跨会话长期保存的信息")
    #profile→ 用户相对稳定的信息 goal→ 学习目标 preference→ 学习方式、回答方式偏好 mastery→ 已经掌握的知识 weakness→ 当前薄弱知识
    category: Literal["profile", "goal", "preference", "mastery", "weakness"] | None = None
    content: str | None = Field(default=None, description="提炼后的长期记忆")