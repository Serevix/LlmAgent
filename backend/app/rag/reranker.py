"""硅基流动重排：用问题与候选原文重新计算相关性，不执行回答生成。"""

import math
from dataclasses import replace

import httpx

from app.core.config import settings
from app.rag.types import RetrievedChunk


class RerankError(RuntimeError):
    """重排失败时返回可读错误，避免把第三方响应体或 Key 写入日志。"""

    def __init__(self, message: str, *, retryable: bool = False):
        """仅保存安全提示与可重试标志，不保存第三方响应体和密钥。"""
        super().__init__(message)
        self.retryable = retryable


class Reranker:
    """配置只在调用时读取；聊天模型切换不会替换独立的重排模型。"""

    async def rerank(self, question: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """按返回 index 对齐原片段，验证全量排名，再按分数降序返回。

        请求所有候选的分数，方便在重排后单独筛选和保留短书上下文。
        超时、鉴权失败、缺项或重复 index 都明确报错，不悄悄跳过重排。
        """
        if not hits:
            return []
        if len(hits) > settings.RAG_RERANK_MAX_CANDIDATES:
            raise RerankError("重排候选超过配置上限，请调整检索数量。")
        secret = settings.RERANK_API_KEY or settings.GuiJiLiuDong_API_KEY
        if not secret:
            raise RerankError("请配置 RERANK_API_KEY 或 GuiJiLiuDong_API_KEY 后使用重排。")
        base = settings.RERANK_BASE_URL or settings.GuiJiLiuDong_BASE_URL or "https://api.siliconflow.cn/v1"
        try:
            async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT, follow_redirects=False) as client:
                response = await client.post(base.rstrip("/") + "/rerank",
                    headers={"Authorization": f"Bearer {secret}"},
                    json={"model": settings.RERANK_MODEL, "query": question,
                          "documents": [hit.chunk.text for hit in hits],
                          "top_n": len(hits), "return_documents": False})
                response.raise_for_status()
                results = response.json()["results"]
            if not isinstance(results, list) or len(results) != len(hits):
                raise ValueError("重排结果数量错误")
            ranked, seen = [], set()
            for result in results:
                index, score = result["index"], result["relevance_score"]
                if (type(index) is not int or not 0 <= index < len(hits) or index in seen
                        or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("重排结果编号或分数无效")
                seen.add(index)
                ranked.append(replace(hits[index], score=float(score), score_type="rerank", rerank_score=float(score)))
            return sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.id))
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code in (408, 409, 429, 500, 502, 503, 504)
            raise RerankError("重排服务调用失败，请检查接口、模型和密钥，或稍后重试。",
                              retryable=retryable) from None
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            raise RerankError("重排服务暂时不可用，请稍后重试。", retryable=True) from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError):
            # 不包含 exc 的文本，服务商错误可能回显请求内容和凭据。
            raise RerankError("重排服务调用失败或结果无效，请检查接口、模型和密钥。") from None
