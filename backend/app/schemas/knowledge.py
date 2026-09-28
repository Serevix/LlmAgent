from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BookRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    status: str
    chunk_count: int
    error: str | None
    created_at: datetime


class SourceRead(BaseModel):
    book_id: str
    title: str
    page: int
    chunk_id: str
    score: float
    excerpt: str
