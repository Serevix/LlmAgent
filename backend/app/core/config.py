"""统一加载环境配置，并校验模型上下文预算与数据库连接配置。"""

from functools import lru_cache  #用于为函数添加缓存功能，以提升性能。
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
#BaseSettings：这是一个基类，用于定义应用的配置模型。继承它的类可以自动从环境变量、.env文件等来源加载配置，并利用Pydantic进行类型验证和转换。
#SettingsConfigDict：这是一个配置字典，用于定义配置模型的配置选项。它可以指定配置项的默认值、环境变量前缀、是否必需等。
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR=Path(__file__).resolve().parents[2]
env_path=BASE_DIR / ".env"



class Settings(BaseSettings):
    """应用配置模型；从环境变量和 .env 读取并验证配置。"""
    PROJECT_NAME: str = "LLM Learning Assistant"
    VERSION:str="0.1.0"

    API_V1_PREFIX:str="/api/v1"

    #JWT
    JWT_SECRET_KEY:str  #签名密钥
    JWT_ALGORITHM:str="HS256"#签名算法
    ACCESS_TOKEN_EXPIRE_MINUTES:int=120# 表示登录Token两个小时后过期


    DEBUG:bool=True
    DATABASE_URL:str
    FRONTEND_ORIGINS:str="http://localhost:5173,http://127.0.0.1:5173"
    MILVUS_URI:str="http://127.0.0.1:19530"
    MILVUS_TOKEN:str|None=None
    MILVUS_COLLECTION:str="llm_book_chunks_bge_m3"
    # 原集合保留为迁移来源；应用的新读写使用混合索引集合或同名别名。
    MILVUS_HYBRID_COLLECTION: str = "llm_book_chunks_bge_m3_hybrid"
    EMBEDDING_MODEL:str="BAAI/bge-m3"
    EMBEDDING_DIMENSION:int=1024
    RAG_MIN_SCORE:float=0.55
    RAG_RETRIEVAL_MODE: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    RAG_RECALL_K: int = Field(default=20, ge=5, le=100)
    RAG_FUSION_K: int = Field(default=40, ge=5, le=200)
    RAG_RRF_K: int = Field(default=60, ge=1, le=1000)
    RAG_RERANK_MAX_CANDIDATES: int = Field(default=64, ge=5, le=256)
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_BASE_URL: str | None = None
    RERANK_API_KEY: str | None = Field(default=None, repr=False)
    RERANK_TIMEOUT: float = Field(default=25, ge=1, le=120)
    # 初始工作阈值，不是概率保证，需通过实际语料评测校准。
    RERANK_MIN_SCORE: float = Field(default=0.30, ge=0, le=1)
    CHAT_MODEL:str="openai:deepseek-ai/deepseek-v4-flash"
    # 可选的服务端模型名称，逗号分隔，使用同一个 SiliconFlow 服务端凭据。
    CHAT_MODEL_OPTIONS: str = ""
    # auto 仅为含指代/省略的追问调用改写模型；always 保留原来的每轮改写策略。
    CHAT_QUERY_REWRITE_MODE: Literal["auto", "always"] = "auto"
    # selective 仅提炼学习目标/偏好等候选信息；always 每轮判断；off 关闭自动提炼。
    CHAT_MEMORY_MODE: Literal["selective", "always", "off"] = "selective"
    # 自定义模型 API Key 的加密密钥；不配置时由 JWT 签名密钥派生。
    MODEL_CREDENTIAL_SECRET: str | None = None
    # 这是部署预算，不是自动探测的模型规格；更换模型时应按服务商的实际窗口调整。
    CHAT_CONTEXT_WINDOW_TOKENS: int = Field(default=32768, ge=4096)
    CHAT_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=128)
    CONTEXT_SAFETY_MARGIN_TOKENS: int = Field(default=1024, ge=256)
    # 下面的输入预算使用 UTF-8 字节保守估算，中文通常比真实 token 计数更早触发摘要。
    CONTEXT_HISTORY_BUDGET_TOKENS: int = Field(default=6000, ge=512)
    CONTEXT_SUMMARY_TRIGGER_TOKENS: int = Field(default=6000, ge=512)
    CONTEXT_RECENT_MESSAGES: int = Field(default=8, ge=1)
    CONTEXT_SUMMARY_MAX_TOKENS: int = Field(default=1800, ge=128)
    CONTEXT_SUMMARY_BATCH_TOKENS: int = Field(default=6000, ge=512)
    CONTEXT_AUX_MAX_OUTPUT_TOKENS: int = Field(default=512, ge=128)
    RAG_CONTEXT_BUDGET_TOKENS: int = Field(default=12000, ge=512)
    BOOK_STORAGE_DIR:Path=BASE_DIR / "data" / "books"
    ARTIFACT_STORAGE_DIR: Path = BASE_DIR / "data" / "generated"
    # 图片接口独立于聊天模型；为空时沿用硅基流动服务端凭据。
    IMAGE_MODEL: str = "Kwai-Kolors/Kolors"
    IMAGE_API_KEY: str | None = Field(default=None, repr=False)
    IMAGE_BASE_URL: str | None = None
    IMAGE_TIMEOUT: float = Field(default=120, ge=10, le=300)
    IMAGE_MAX_BYTES: int = Field(default=10 * 1024 * 1024, ge=1024, le=30 * 1024 * 1024)
    # 只下载图片服务返回的可信存储地址；不允许模型提供任意下载 URL。
    IMAGE_DOWNLOAD_HOSTS: str = "*.siliconflow.cn,*.siliconflow.com,sc-maas.oss-cn-shanghai.aliyuncs.com"
    # DeepSeek
    DeepSeeK_API_KEY: str | None = None
    DeepSeek_BASE_URL: str | None = None
    # 官方平台的模型标识，与硅基流动平台的模型名称分别配置。
    DEEPSEEK_CHAT_MODEL: str = "deepseek-flash"

    # 硅基流动
    GuiJiLiuDong_API_KEY: str | None = None
    GuiJiLiuDong_BASE_URL: str | None = None


    model_config=SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore" #.env 中出现了 Settings 不认识的字段，不报错，直接忽略。
    )

    @property
    def PSYCOPG_DATABASE_URL(self)->str:
        """将 SQLAlchemy 的 asyncpg 地址转换为 checkpoint/Store 使用的 psycopg 地址。"""
        return self.DATABASE_URL.replace("postgresql+asyncpg://","postgresql://",1)

    @model_validator(mode="after")
    def validate_context_budget(self):
        """启动时检查分配的预算，避免配置本身已经超过模型窗口。"""
        if self.MILVUS_COLLECTION == self.MILVUS_HYBRID_COLLECTION:
            raise ValueError("原向量集合与混合索引集合必须使用不同名称")
        if self.RAG_RERANK_MAX_CANDIDATES < self.RAG_FUSION_K + 24:
            raise ValueError("重排候选上限至少为 RAG_FUSION_K + 24，为短书补取预留空间")
        allocated = (self.CONTEXT_HISTORY_BUDGET_TOKENS + self.CONTEXT_SUMMARY_MAX_TOKENS
                     + self.RAG_CONTEXT_BUDGET_TOKENS + self.CHAT_MAX_OUTPUT_TOKENS
                     + self.CONTEXT_SAFETY_MARGIN_TOKENS + 1024)
        if allocated > self.CHAT_CONTEXT_WINDOW_TOKENS:
            raise ValueError("历史、摘要、书籍证据和输出预算之和超过 CHAT_CONTEXT_WINDOW_TOKENS")
        if self.CONTEXT_SUMMARY_BATCH_TOKENS + self.CONTEXT_SUMMARY_MAX_TOKENS * 2 + 2048 > self.CHAT_CONTEXT_WINDOW_TOKENS:
            raise ValueError("摘要批次预算过大，请减小 CONTEXT_SUMMARY_BATCH_TOKENS")
        return self


@lru_cache()
def get_settings() -> Settings:
    """创建并缓存配置，避免每次请求重新读取环境变量。"""
    return Settings()

settings=get_settings()
