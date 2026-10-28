"""调用硅基流动生图接口，立即下载并验证图片，避免临时 URL 失效。"""

import asyncio
import io
import ipaddress
import socket
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from app.artifacts.errors import ArtifactProviderError
from app.core.config import settings


async def validate_image_url(url: str) -> None:
    """只允许 HTTPS 可信图片存储域名，并拒绝内网、凭据 URL 和重定向。"""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    allowed = [item.strip().lower() for item in settings.IMAGE_DOWNLOAD_HOSTS.split(",") if item.strip()]
    trusted = any(host == rule or (rule.startswith("*.") and host.endswith(rule[1:])) for rule in allowed)
    if (parsed.scheme != "https" or not trusted or parsed.username or parsed.password
            or parsed.port not in (None, 443)):
        raise ArtifactProviderError("图片服务返回了未允许的下载地址，请检查 IMAGE_DOWNLOAD_HOSTS 配置。")
    addresses = await asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ArtifactProviderError("图片下载地址不可用。")


def validate_image(contents: bytes) -> tuple[str, str]:
    """验证真实文件格式和像素上限，保存原始图片以保留服务商水印。"""
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("图片尺寸过大")
            formats = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"),
                       "WEBP": ("webp", "image/webp")}
            result = formats.get(image.format or "")
            if result is None:
                raise ValueError("不支持的图片格式")
            image.verify()
            return result
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ArtifactProviderError("图片服务返回的图片文件无效。") from None


async def generate_image_bytes(prompt: str, aspect: str) -> tuple[bytes, str, str]:
    """每次只生成一张；不自动重试收费的生成请求，下载请求不携带 API Key。"""
    key = settings.IMAGE_API_KEY or settings.GuiJiLiuDong_API_KEY
    if not key:
        raise ArtifactProviderError("请先配置图片服务的硅基流动 API Key。")
    is_qwen = "qwen-image" in settings.IMAGE_MODEL.lower()
    sizes = ({"square": "1328x1328", "landscape": "1664x928", "portrait": "928x1664"}
             if is_qwen else {"square": "1024x1024", "landscape": "1280x720", "portrait": "720x1280"})
    body = {"model": settings.IMAGE_MODEL, "prompt": prompt, "image_size": sizes[aspect],
            "num_inference_steps": 20}
    base = settings.IMAGE_BASE_URL or settings.GuiJiLiuDong_BASE_URL or "https://api.siliconflow.cn/v1"
    try:
        async with httpx.AsyncClient(timeout=settings.IMAGE_TIMEOUT, follow_redirects=False) as client:
            response = await client.post(base.rstrip("/") + "/images/generations",
                                         headers={"Authorization": f"Bearer {key}"}, json=body)
            response.raise_for_status()
            url = response.json()["images"][0]["url"]
            if not isinstance(url, str) or len(url) > 8192:
                raise ValueError("下载地址无效")
            await validate_image_url(url)
            # 单独的 GET 请求不复用 Authorization；禁止跟随到未校验的地址。
            async with client.stream("GET", url) as download:
                download.raise_for_status()
                data = bytearray()
                async for part in download.aiter_bytes():
                    data.extend(part)
                    if len(data) > settings.IMAGE_MAX_BYTES:
                        raise ArtifactProviderError("生成图片超过文件大小上限。")
        contents = bytes(data)
        extension, media_type = await asyncio.to_thread(validate_image, contents)
        return contents, extension, media_type
    except ArtifactProviderError:
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, OSError):
        raise ArtifactProviderError("图片生成或下载失败，请检查图片模型、账号额度和网络后重试。") from None
