"""书籍处理的数据类型：页面文本、切片及带相似度的检索结果。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """解析后的页面文本；PDF 使用真实页码，纯文本使用零作为无页码标记。"""
    page: int
    text: str


@dataclass(frozen=True)
class BookChunk:
    """可索引文本块，携带所属用户、书籍、位置及完整原文。"""
    id: str
    book_id: str
    user_id: int
    title: str
    page: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """片段与分阶段分数；score 是当前排序分数，由 score_type 解释其含义。"""
    chunk: BookChunk
    score: float
    score_type: str = "dense"
    dense_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    is_expansion: bool = False
