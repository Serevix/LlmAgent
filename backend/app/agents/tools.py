"""供学习助手工作流调用的受控工具。"""

from langchain_core.tools import tool
from sqlmodel.ext.asyncio.session import AsyncSession

from app.infrastructure.database.database import engine
from app.services.knowledge_service import KnowledgeService


@tool
async def search_book_chunks(question: str, user_id: int) -> list[dict]:
    """检索当前用户已成功导入的大模型工程书籍，返回相关原文片段及来源信息，供回答时引用。

    参数说明：
        question：需要从书籍中查找依据的问题。
        user_id：由工作流提供的当前登录用户编号。

    返回片段原文、来源和分阶段分数；score_type 区分余弦、RRF 与重排，不能混用阈值。
    """
    # 在本次工具调用中创建数据库会话，用于查询书籍归属和导入状态。
    async with AsyncSession(engine) as session:
        # 本工具由 retrieve 节点调用，重试交给该节点，避免 Embedding SDK 再次叠加重试。
        hits = await KnowledgeService(embedding_max_retries=0).search(session, user_id, question)
    # 将检索结果对象转换成字典；保留完整原文供回答使用，同时提供简短原文摘录供展示。
    return [
        {"book_id": hit.chunk.book_id, "title": hit.chunk.title,
         "page": hit.chunk.page, "chunk_id": hit.chunk.id,
         "score": round(hit.score, 4), "excerpt": hit.chunk.text[:240],
         "score_type": hit.score_type, "dense_score": hit.dense_score,
         "fusion_score": hit.fusion_score, "rerank_score": hit.rerank_score,
         "is_expansion": hit.is_expansion,
         "text": hit.chunk.text}
        for hit in hits
    ]
