"""模型配置与请求级工作流：用户密钥加密入库，切换模型仍使用原 thread_id。"""

import base64
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from cryptography.fernet import Fernet, InvalidToken
from langchain_openai import ChatOpenAI
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.chat_graph import build_chat_graph
from app.agents.context import ContextLimits
from app.core.config import settings
from app.models.model_profile import ModelProfile
from app.repositories.model_profile_repository import ModelProfileRepository
from app.schemas.model_profile import ModelProfileCreate, ModelProfileRead


@dataclass(frozen=True, repr=False)
class ModelSpec:
    """仅在本次调用内持有解密凭据，不作为图状态或日志数据。"""

    model: str
    base_url: str
    api_key: str
    context_window: int
    max_output_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    timeout: int = 60
    structured_output_method: str | None = None


def credential_cipher() -> Fernet:
    """以服务端秘密派生 Fernet 密钥；更换秘密后旧密文需重新配置。"""
    secret = settings.MODEL_CREDENTIAL_SECRET or settings.JWT_SECRET_KEY
    key = hashlib.sha256(("llm-model-credentials:" + secret).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


class ModelService:
    """按用户管理配置；服务端预设只引用服务端凭据，列表中永不包含凭据。"""

    def __init__(self):
        """创建模型配置仓储；只保存依赖，不在初始化时连接数据库或调用模型。"""
        self.repository = ModelProfileRepository()

    def builtin_models(self) -> list[ModelProfileRead]:
        """读取环境配置，生成前端可以选择的服务端预设。

        默认模型始终存在；额外硅基流动预设按模型名去重；官方 Key 非空时
        加入 builtin:deepseek。返回公开的编号、名称和预算，不返回任何密钥。
        list_models 用它拼接用户配置，resolve 用它识别可信的内置编号。
        """
        result = [ModelProfileRead(id="default", name=f"硅基流动（默认） · {settings.CHAT_MODEL.removeprefix('openai:')}",
                                  model=settings.CHAT_MODEL, kind="builtin",
                                  context_window=settings.CHAT_CONTEXT_WINDOW_TOKENS,
                                  max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS)]
        for model in dict.fromkeys(item.strip() for item in settings.CHAT_MODEL_OPTIONS.split(",") if item.strip()):
            if model != settings.CHAT_MODEL:
                identity = "builtin:" + hashlib.sha256(model.encode()).hexdigest()[:16]
                result.append(ModelProfileRead(id=identity, name=f"硅基流动 · {model.removeprefix('openai:')}",
                                              model=model, kind="builtin",
                                              context_window=settings.CHAT_CONTEXT_WINDOW_TOKENS,
                                              max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS))
        if settings.DeepSeeK_API_KEY and settings.DeepSeeK_API_KEY.strip():
            result.append(ModelProfileRead(
                id="builtin:deepseek", name=f"DeepSeek 官方 · {settings.DEEPSEEK_CHAT_MODEL}",
                model=settings.DEEPSEEK_CHAT_MODEL, kind="builtin",
                context_window=settings.CHAT_CONTEXT_WINDOW_TOKENS,
                max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
            ))
        return result

    @staticmethod
    def public_profile(profile: ModelProfile) -> ModelProfileRead:
        """显式挑选允许返回的字段，排除明文及密文密钥。"""
        return ModelProfileRead(id=profile.id, name=profile.name, model=profile.model, kind="custom",
                                base_url=profile.base_url, context_window=profile.context_window,
                                max_output_tokens=profile.max_output_tokens, temperature=profile.temperature,
                                top_p=profile.top_p, timeout=profile.timeout,
                                structured_output_method=profile.structured_output_method,
                                has_api_key=bool(profile.encrypted_api_key))

    async def list_models(self, session: AsyncSession, user_id: int) -> list[ModelProfileRead]:
        """返回服务端预设与当前用户自定义配置。"""
        owned = await self.repository.list_owned(session, user_id)
        return [*self.builtin_models(), *(self.public_profile(profile) for profile in owned)]

    async def create(self, session: AsyncSession, user_id: int, data: ModelProfileCreate) -> ModelProfileRead:
        """把已验证的用户输入转换为数据库记录，并返回公开配置。

        session 用于关系库读写，user_id 来自登录身份，data 来自请求 schema。
        限制每个用户最多 20 条，API Key 经 Fernet 加密后才交给 repository。
        保存配置不会调用远程模型；超过限制时抛 ValueError，由接口转为 400。
        """
        owned = await self.repository.list_owned(session, user_id)
        if len(owned) >= 20:
            raise ValueError("最多保存 20 个自定义模型，请先删除不再使用的配置。")
        values = data.model_dump(exclude={"api_key"})
        secret = data.api_key.get_secret_value().strip()
        encrypted = credential_cipher().encrypt(secret.encode()).decode() if secret else ""
        profile = ModelProfile(user_id=user_id, encrypted_api_key=encrypted, **values)
        return self.public_profile(await self.repository.create(session, profile))

    async def resolve(self, session: AsyncSession, user_id: int, model_id: str | None) -> ModelSpec | None:
        """将前端 model_id 转换为本次请求的 ModelSpec，默认选择返回 None。

        None/default 表示让 runtime_graph 复用已装配的默认图；内置编号读取
        对应平台环境配置；自定义编号按 user_id 查归属，再解密 API Key。
        配置不存在抛 LookupError，解密失败抛 ValueError；chat.resolve_model
        在流开始前转换为 HTTP 错误。ModelSpec 含凭据，不能写入消息或响应。
        """
        if not model_id or model_id == "default":
            return None
        for builtin in self.builtin_models()[1:]:
            if builtin.id == model_id:
                # 官方预设必须使用官方的地址和 Key，不能落入硅基流动凭据分支。
                if builtin.id == "builtin:deepseek":
                    return ModelSpec(builtin.model, settings.DeepSeek_BASE_URL or "https://api.deepseek.com/v1",
                                     settings.DeepSeeK_API_KEY or "", builtin.context_window,
                                     builtin.max_output_tokens, structured_output_method="function_calling")
                return ModelSpec(builtin.model, settings.GuiJiLiuDong_BASE_URL or "https://api.siliconflow.cn/v1",
                                 settings.GuiJiLiuDong_API_KEY or "", builtin.context_window,
                                 builtin.max_output_tokens)
        profile = await self.repository.get_owned(session, model_id, user_id)
        if profile is None:
            raise LookupError("模型配置不存在，请重新选择模型。")
        try:
            secret = (credential_cipher().decrypt(profile.encrypted_api_key.encode()).decode()
                      if profile.encrypted_api_key else "")
        except InvalidToken:
            raise ValueError("模型密钥无法解密，请删除该配置并重新添加。") from None
        return ModelSpec(profile.model, profile.base_url, secret, profile.context_window,
                         profile.max_output_tokens, profile.temperature, profile.top_p,
                         profile.timeout, profile.structured_output_method)

    @asynccontextmanager
    async def runtime_model(self, spec: ModelSpec):
        """根据 ModelSpec 创建并 yield 本次请求使用的 ChatOpenAI。

        调用者用 async with 获取模型；请求结束、异常或取消时退出上下文，
        自动关闭同步及异步 HTTP 客户端。接口地址、Key、采样参数和超时
        来自 spec；不跟随重定向，不将用户密钥缓存到共享图。
        """
        # 不跟随重定向，避免请求中的 API Key 被转发到另一个地址。
        async with httpx.AsyncClient(follow_redirects=False) as async_client:
            with httpx.Client(follow_redirects=False) as sync_client:
                parameters = {key: value for key, value in {
                    "temperature": spec.temperature, "top_p": spec.top_p,
                }.items() if value is not None}
                model = ChatOpenAI(model=spec.model.removeprefix("openai:"), base_url=spec.base_url,
                                   api_key=spec.api_key or "local-no-key", timeout=spec.timeout,
                                   max_retries=1, stream_usage=False, use_responses_api=False,
                                   http_async_client=async_client, http_client=sync_client, **parameters)
                yield model

    @asynccontextmanager
    async def runtime_graph(self, app, spec: ModelSpec | None):
        """为聊天接口提供可执行图，保持模型切换前后的会话连续性。

        app 提供启动时创建的图、checkpointer 和 Store；spec=None 直接 yield
        默认图，否则创建独立模型和 ContextLimits，再编译同一节点结构的图。
        复用持久化组件，thread_id 仍由聊天接口传入；不修改全局配置。
        调用者必须在该上下文中消费完整流，避免提前关闭模型客户端。
        """
        if spec is None:
            yield app.state.chat_graph
            return
        async with self.runtime_model(spec) as model:
            limits = ContextLimits.from_settings(spec.context_window, spec.max_output_tokens)
            yield build_chat_graph(model, app.state.checkpointer, app.state.store,
                                   limits=limits, structured_output_method=spec.structured_output_method)
