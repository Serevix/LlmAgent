"""生成工具对外使用的安全中文错误。"""


class ArtifactError(RuntimeError):
    """内容或参数不适合生成，错误提示可以直接展示给用户。"""


class ArtifactProviderError(ArtifactError):
    """图片服务或工具规划模型调用失败；不携带第三方响应体和密钥。"""
