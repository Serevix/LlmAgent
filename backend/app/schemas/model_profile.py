"""自定义模型参数验证与公开响应；密钥只接受写入，不返回明文或密文。"""

import ipaddress
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from app.agents.context import ContextLimits


class ModelProfileCreate(BaseModel):
    """手填及 JSON 导入使用同一数据格式，拒绝未知参数，避免悄悄忽略拼写错误。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100, description="下拉框显示的名称。")
    model: str = Field(min_length=1, max_length=200, description="模型服务要求的模型标识。")
    base_url: str = Field(min_length=1, max_length=800, description="兼容 OpenAI 的 API 根地址。")
    api_key: SecretStr = Field(default=SecretStr(""), max_length=4096, description="无需鉴权的本地服务可留空。")
    context_window: int = Field(default=32768, ge=4096, le=2000000)
    max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    timeout: int = Field(default=60, ge=10, le=120)
    structured_output_method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling"

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """支持本地模型，但禁止 URL 中嵌入密钥、用户密码及非 HTTP 协议。"""
        try:
            parsed = urlsplit(value)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError()
            _ = parsed.port
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError:
                address = None
            if address and (address.is_link_local or address.is_multicast or address.is_unspecified):
                raise ValueError()
        except ValueError:
            raise ValueError("接口地址须为 HTTP/HTTPS 根地址，不能包含账号、密钥、查询参数或片段。") from None
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_budget(self):
        """模型自带窗口限制也约束摘要、检索证据和辅助调用。"""
        ContextLimits.from_settings(self.context_window, self.max_output_tokens)
        return self


class ModelProfileRead(BaseModel):
    """用于模型选择与管理；不包含 API Key，前端无需永久保存密钥。"""

    id: str
    name: str
    model: str
    kind: Literal["builtin", "custom"]
    base_url: str | None = None
    context_window: int
    max_output_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    timeout: int = 60
    structured_output_method: str = "function_calling"
    has_api_key: bool = False
