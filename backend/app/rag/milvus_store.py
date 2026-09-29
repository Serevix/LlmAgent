import asyncio

from pymilvus import DataType, MilvusClient

from app.core.config import settings
from app.rag.types import BookChunk, RetrievedChunk


class MilvusBookStore:
    """Vector storage only; PostgreSQL remains the source for book ownership/status."""

    def _client(self) -> MilvusClient:
        return MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN or "",
            timeout=10,
        )

    def _ping(self) -> None:
        client = self._client()
        try:
            client.list_collections(timeout=10)
        finally:
            client.close()

    async def ping(self) -> None:
        await asyncio.to_thread(self._ping)

    def _ensure_collection(self, client: MilvusClient) -> None:
        name = settings.MILVUS_COLLECTION
        if client.has_collection(name):
            description = client.describe_collection(name)
            vector = next((field for field in description["fields"] if field["name"] == "vector"), None)
            dimension = int((vector or {}).get("params", {}).get("dim", 0))
            if dimension and dimension != settings.EMBEDDING_DIMENSION:
                raise ValueError("Milvus 集合维度与 EMBEDDING_DIMENSION 不一致，请更换集合名或模型")
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
        schema.add_field("user_id", DataType.INT64)
        schema.add_field("book_id", DataType.VARCHAR, max_length=32)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(name, schema=schema, index_params=indexes)

    def _insert(self, chunks: list[BookChunk], vectors: list[list[float]]) -> None:
        client = self._client()
        try:
            self._ensure_collection(client)
            for start in range(0, len(chunks), 100):
                rows = [
                    {"id": chunk.id, "vector": vector, "user_id": chunk.user_id,
                     "book_id": chunk.book_id, "title": chunk.title, "page": chunk.page,
                     "text": chunk.text}
                    for chunk, vector in zip(chunks[start:start + 100], vectors[start:start + 100])
                ]
                client.insert(settings.MILVUS_COLLECTION, data=rows)
        finally:
            client.close()

    async def insert(self, chunks: list[BookChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("文本段和向量数量不一致")
        await asyncio.to_thread(self._insert, chunks, vectors)

    def _search(self, user_id: int, vector: list[float], limit: int,
                book_ids: list[str]) -> list[RetrievedChunk]:
        if not book_ids:
            return []
        client = self._client()
        try:
            if not client.has_collection(settings.MILVUS_COLLECTION):
                return []
            book_filter = ", ".join(f'"{book_id}"' for book_id in book_ids)
            hits = client.search(
                settings.MILVUS_COLLECTION, data=[vector],
                filter=f"user_id == {int(user_id)} and book_id in [{book_filter}]",
                limit=limit, output_fields=["user_id", "book_id", "title", "page", "text"],
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong",
            )[0]
            return [
                RetrievedChunk(
                    chunk=BookChunk(id=str(hit["id"]), book_id=hit["entity"]["book_id"],
                                    user_id=int(hit["entity"]["user_id"]),
                                    title=hit["entity"]["title"], page=int(hit["entity"]["page"]),
                                    text=hit["entity"]["text"]),
                    score=float(hit["distance"]),
                )
                for hit in hits
            ]
        finally:
            client.close()

    async def search(self, user_id: int, vector: list[float], limit: int = 5,
                     book_ids: list[str] | None = None) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._search, user_id, vector, limit, book_ids or [])

    def _delete_book(self, user_id: int, book_id: str) -> None:
        client = self._client()
        try:
            if client.has_collection(settings.MILVUS_COLLECTION):
                client.delete(settings.MILVUS_COLLECTION,
                              filter=f'user_id == {int(user_id)} and book_id == "{book_id}"')
        finally:
            client.close()

    async def delete_book(self, user_id: int, book_id: str) -> None:
        await asyncio.to_thread(self._delete_book, user_id, book_id)
