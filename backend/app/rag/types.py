from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class BookChunk:
    id: str
    book_id: str
    user_id: int
    title: str
    page: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: BookChunk
    score: float
