"""调用兼容 OpenAI 的 Embedding 接口，将文本批量转换成同序向量。"""

from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:
    """为书籍切片和问题使用同一 Embedding 模型，保持向量空间一致。"""
    def __init__(self, *, max_retries: int = 2):
        """导入默认最多重试两次；图内检索传零，由 LangGraph 控制重试次数。"""
        if not settings.GuiJiLiuDong_API_KEY:
            raise ValueError("请先配置 GuiJiLiuDong_API_KEY 才能使用书籍知识库")
        self.client = AsyncOpenAI(
            api_key=settings.GuiJiLiuDong_API_KEY,
            base_url=settings.GuiJiLiuDong_BASE_URL or "https://api.siliconflow.cn/v1",
            timeout=45.0,
            max_retries=max_retries,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """每批最多十六段；按返回 index 对齐向量并检查数量和维度。"""
        result: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            response = await self.client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if len(vectors) != len(batch) or any(len(vector) != settings.EMBEDDING_DIMENSION for vector in vectors):
                raise ValueError("Embedding 维度或返回数量与配置不一致")
            result.extend(vectors)
        return result
