"""检索节点的有限重试策略：仅重试可恢复的读操作故障。"""

import grpc
import httpx
from langgraph.types import RetryPolicy
from openai import APIConnectionError, APIStatusError
from pymilvus.exceptions import ErrorCode, MilvusException
from sqlalchemy.exc import DBAPIError, TimeoutError as DatabaseTimeoutError

from app.rag.reranker import RerankError


TRANSIENT_HTTP_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
TRANSIENT_RPC_STATUS = frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED,
                                grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.ABORTED})


class RetryableRetrievalError(RuntimeError):
    """临时检索故障的安全包装；LangGraph 重试日志和接口都只接触中文提示。"""


def is_transient_retrieval_error(exc: Exception) -> bool:
    """识别网络、限流和临时服务故障；鉴权、参数、解析与程序错误直接失败。"""
    if isinstance(exc, RerankError):
        return exc.retryable
    if isinstance(exc, APIStatusError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, (APIConnectionError, httpx.TimeoutException, httpx.NetworkError,
                        httpx.RemoteProtocolError, TimeoutError, ConnectionError, DatabaseTimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, DBAPIError):
        return exc.connection_invalidated
    if isinstance(exc, grpc.RpcError):
        return callable(getattr(exc, "code", None)) and exc.code() in TRANSIENT_RPC_STATUS
    if isinstance(exc, MilvusException):
        # Milvus 可能把底层 gRPC 异常包成 MilvusException，保留按错误码判断的能力。
        if exc.code == ErrorCode.RATE_LIMIT or exc.code in TRANSIENT_RPC_STATUS:
            return True
        return isinstance(exc.__cause__, grpc.RpcError) and is_transient_retrieval_error(exc.__cause__)
    return False


RETRIEVAL_RETRY_POLICY = RetryPolicy(
    max_attempts=2,          # 包含第一次执行，所以最多只重试一次。
    initial_interval=0.5,   # 短暂等待，避免立即再次冲击故障服务。
    backoff_factor=1.0,
    max_interval=0.5,
    jitter=False,
    retry_on=RetryableRetrievalError,
)
