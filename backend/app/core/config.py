from functools import lru_cache  #用于为函数添加缓存功能，以提升性能。
from pathlib import Path
#BaseSettings：这是一个基类，用于定义应用的配置模型。继承它的类可以自动从环境变量、.env文件等来源加载配置，并利用Pydantic进行类型验证和转换。
#SettingsConfigDict：这是一个配置字典，用于定义配置模型的配置选项。它可以指定配置项的默认值、环境变量前缀、是否必需等。
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR=Path(__file__).resolve().parents[2]
env_path=BASE_DIR / ".env"



class Settings(BaseSettings):
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
    EMBEDDING_MODEL:str="BAAI/bge-m3"
    EMBEDDING_DIMENSION:int=1024
    RAG_MIN_SCORE:float=0.55
    CHAT_MODEL:str="openai:deepseek-ai/deepseek-v4-flash"
    BOOK_STORAGE_DIR:Path=BASE_DIR / "data" / "books"
    # DeepSeek
    DeepSeeK_API_KEY: str | None = None
    DeepSeek_BASE_URL: str | None = None

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
        return self.DATABASE_URL.replace("postgresql+asyncpg://","postgresql://",1)


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings=get_settings()
