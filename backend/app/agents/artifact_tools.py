"""供模型选择的生成工具；身份、会话、幂等键和可信正文由后端注入。"""

from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.services.artifact_service import ArtifactService


@tool
async def generate_image(
    prompt: Annotated[str, Field(min_length=2, max_length=3000, description="完整的中文或英文画面描述，说明主体、构图和风格。")],
    title: Annotated[str, Field(min_length=1, max_length=120, description="图片的简短中文名称。")],
    user_id: Annotated[int, InjectedToolArg],
    thread_id: Annotated[str, InjectedToolArg],
    request_key: Annotated[str, InjectedToolArg],
    aspect: Literal["square", "landscape", "portrait"] = "square",
) -> dict:
    """根据用户的明确绘图需求生成一张图片并返回附件信息。aspect 分别代表方图、横图、竖图。

    仅在用户要求画图或生成图片时使用；用户询问绘图原理、代码或工具用法时不要调用。
    图片是生成式创作，不作为知识库原文或事实证明。
    """
    return await ArtifactService().create(user_id=user_id, thread_id=thread_id, request_key=request_key,
                                         kind="image", title=title, prompt=prompt, aspect=aspect)


@tool
async def generate_word(
    title: Annotated[str, Field(min_length=1, max_length=120, description="Word 文档的中文标题。")],
    user_id: Annotated[int, InjectedToolArg],
    thread_id: Annotated[str, InjectedToolArg],
    request_key: Annotated[str, InjectedToolArg],
    content: Annotated[str, InjectedToolArg],
    sources: Annotated[list[dict], InjectedToolArg],
    content_source: Literal["latest_answer", "knowledge"] = "latest_answer",
    question: Annotated[str, Field(max_length=3000, description="knowledge 模式下需要检索的独立知识问题，保留用户的主题与要求。") ] = "",
) -> dict:
    """生成可下载的 Word 文档，保留标题、列表、代码、表格与书籍引用。

    用户要求导出刚才的回答时选 latest_answer；要求根据知识库生成新主题文档时选 knowledge，
    并填写要检索的 question。正文和引用由后端从已核对回答中注入，无依据时不会生成文档。
    仅咨询 Word 使用方法时不要调用。
    """
    return await ArtifactService().create(user_id=user_id, thread_id=thread_id, request_key=request_key,
                                         kind="word", title=title, content=content, sources=sources)


ARTIFACT_TOOLS = {item.name: item for item in (generate_image, generate_word)}
