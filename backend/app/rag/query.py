"""识别问题中的数量列举表达，为短文档的补充检索提供信号。"""

import re


ENUMERATION_PATTERN = re.compile(r"(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*(?:种|个|项|类|条|大)")


def asks_for_numbered_list(question: str) -> bool:
    """检查问题是否包含数量加单位的表达；这是扩展检索信号，不是输入合法性验证。"""
    return bool(ENUMERATION_PATTERN.search(question))
