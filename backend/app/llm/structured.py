"""JSON 模式需要显式告诉模型输出结构；工具调用和 JSON Schema 模式由协议携带结构。"""

import json

from pydantic import BaseModel


def output_instructions(schema: type[BaseModel], method: str | None) -> str:
    """只为 JSON 模式追加完整格式说明，并让后续输入计数包含这些文字。"""
    if method != "json_mode":
        return ""
    return "\n请严格输出 JSON 对象，不要 Markdown。字段及约束如下：\n" + json.dumps(
        schema.model_json_schema(), ensure_ascii=False,
    )
