from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:
    def __init__(self):
        if not settings.GuiJiLiuDong_API_KEY:
            raise ValueError("请先配置 GuiJiLiuDong_API_KEY 才能使用书籍知识库")
        self.client = AsyncOpenAI(
            api_key=settings.GuiJiLiuDong_API_KEY,
            base_url=settings.GuiJiLiuDong_BASE_URL or "https://api.siliconflow.cn/v1",
            timeout=45.0,
            max_retries=2,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            response = await self.client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if len(vectors) != len(batch) or any(len(vector) != settings.EMBEDDING_DIMENSION for vector in vectors):
                raise ValueError("Embedding 维度或返回数量与配置不一致")
            result.extend(vectors)
        return result
