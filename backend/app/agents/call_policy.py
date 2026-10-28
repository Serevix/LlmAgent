"""用本地规则筛选辅助任务，避免每轮问答都额外调用改写和长期记忆模型。"""

import re


# 这里只判断问题是否可能依赖前文，不校验问题合法性，也不决定能否回答。
CONTEXT_REFERENCE = re.compile(
    r"它|他们|它们|她们|这个|那个|这些|那些|这种|那种|这两|那两|二者|该(?:功能|模型|方法)|上述|下述|前面|上面|刚才|之前|上次|"
    r"上文|前文|前者|后者|继续|接着|再(?:说|讲|解释|展开|举)|换(?:个|一个)(?:例子|说法)|"
    r"呢[？?。\s]*$|\b(?:it|its|they|them|this|that|these|those|above|previous|earlier|continue)\b",
    re.IGNORECASE,
)
ELLIPTICAL_FOLLOWUP = re.compile(
    r"^(?:那|那么|还有|然后|具体(?:呢|一点|一些)|详细(?:一点|一些)|举个例子|为什么[？?]?$|"
    r"(?:怎么|如何)(?:用|使用|配置|部署|实现|操作)[呢吗啊？?！!。\s]*$|"
    r"(?:有|是|能|可|支持|区别|优点|缺点|优缺点|适用场景|实际应用))|"
    r"^(?:why\??|how\??|how (?:does it work|to use it)\??|what else\??|give (?:an? )?example\??)$",
    re.IGNORECASE,
)
MEMORY_CANDIDATE = re.compile(
    r"(?:请)?(?:记住|记一下|记下来|记着|保存为长期记忆)|"
    r"(?:以后|今后|往后).{0,30}(?:请|回答|解释|用中文|用英文|举例)|"
    r"我(?:更|比较|最)?(?:喜欢|偏好|习惯|希望|打算|计划)|"
    r"我(?:是一名|是一个|从事|就读于|毕业于)|我想(?:学习|准备|转行|找工作|从事|成为)|"
    r"我的(?:学习目标|目标|职业|工作|专业|背景)|"
    r"我(?:正在|目前在|最近在|主要在)(?:学习|准备|做)|"
    r"我(?:已经|还没|尚未)?(?:掌握|学会|熟悉|不懂|不理解|不熟悉)|"
    r"\b(?:remember (?:that|my|this)|i (?:prefer|plan|am learning)|my (?:goal|preference))\b",
    re.IGNORECASE,
)


def needs_query_rewrite(question: str, has_history: bool, mode: str = "auto") -> bool:
    """有前文且存在指代或省略时才改写；always 可恢复所有多轮问题都改写的策略。"""
    if not has_history:
        return False
    if mode == "always":
        return True
    # 规则偏保守：有明显指代就允许模型补全，专业词和数字不能作为跳过依据。
    return bool(CONTEXT_REFERENCE.search(question) or ELLIPTICAL_FOLLOWUP.search(question.strip()))


def needs_memory_extraction(message: str, mode: str = "selective") -> bool:
    """只筛选可能包含稳定用户信息的消息；是否保存仍由模型判断，不直接写入原文。"""
    if mode == "off":
        return False
    return mode == "always" or bool(MEMORY_CANDIDATE.search(message))
