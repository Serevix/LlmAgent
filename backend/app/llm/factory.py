from typing import Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

def create_chat_model(platform:str,model:str,**model_kwargs:Any)->BaseChatModel:
    model_kwargs.setdefault("timeout", 60)
    model_kwargs.setdefault("max_retries", 1)
    if platform == "siliconflow":
        if (settings.GuiJiLiuDong_API_KEY is None or settings.GuiJiLiuDong_BASE_URL == ""):
            raise ValueError("SiliconFlow 的配置不存在")
        return init_chat_model(
                model=model,
                api_key=settings.GuiJiLiuDong_API_KEY,
                base_url=settings.GuiJiLiuDong_BASE_URL,
                **model_kwargs
        )



    if platform == "deepseek":
        if (settings.DeepSeeK_API_KEY is None or settings.DeepSeeK_API_KEY == ""):
            raise ValueError("DeepSeek 的配置不存在")
        return init_chat_model(
                model=model,
                api_key=settings.DeepSeeK_API_KEY,
                base_url=settings.DeepSeek_BASE_URL,
                **model_kwargs
           )
    raise ValueError("不支持的模型平台")
