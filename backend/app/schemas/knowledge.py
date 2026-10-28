"""书籍状态与引用来源的接口格式，不直接返回整本书内容。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BookRead(BaseModel):
    """书籍列表与导入状态响应，隐藏本地存储路径。"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    status: str
    chunk_count: int
    error: str | None
    created_at: datetime


class SourceRead(BaseModel):
    """前端引用详情所需的书名、页码、切片编号、分数和摘录。"""
    book_id: str
    title: str
    page: int
    chunk_id: str
    score: float
    excerpt: str
    score_type: str = "dense"
    dense_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    is_expansion: bool = False
