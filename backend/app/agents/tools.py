"""Tools callable from the constrained learning-assistant graph."""

from langchain_core.tools import tool
from sqlmodel.ext.asyncio.session import AsyncSession

from app.infrastructure.database.database import engine
from app.services.knowledge_service import KnowledgeService


@tool
async def search_book_chunks(question: str, user_id: int) -> list[dict]:
    """Search the authenticated user's indexed LLM engineering books for evidence."""
    async with AsyncSession(engine) as session:
        hits = await KnowledgeService().search(session, user_id, question)
    return [
        {"book_id": hit.chunk.book_id, "title": hit.chunk.title,
         "page": hit.chunk.page, "chunk_id": hit.chunk.id,
         "score": round(hit.score, 4), "excerpt": hit.chunk.text[:240],
         "text": hit.chunk.text}
        for hit in hits
    ]
