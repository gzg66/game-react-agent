"""Vision LLM client using the google-genai SDK for image analysis."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from game_agent.config import VisionConfig

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class VisionLLMClient:
    """Sends screenshots + prompts to Gemini for structured page analysis."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._client: Any = None
        if HAS_GENAI and config.api_key:
            self._client = genai.Client(
                vertexai=True,
                api_key=config.api_key,
                http_options=types.HttpOptions(api_version="v1"),
            )
            logger.info("视觉 LLM 客户端已初始化，模型=%s", config.model_name)
        elif not HAS_GENAI:
            logger.warning("未安装 google-genai，视觉分析不可用")
        else:
            logger.warning("未提供 API Key，视觉分析不可用")

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def _parse_response_text(raw_text: str) -> dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("无法将视觉模型响应解析为 JSON")
            return {"raw_text": raw_text}

    def chat(self, prompt: str, image_base64: str) -> str:
        """Send an image (base64) + text prompt to Gemini. Returns raw text."""
        if self._client is None:
            logger.warning("视觉 LLM 客户端不可用，返回空结果")
            return "{}"

        try:
            image_bytes = base64.b64decode(image_base64)
            response = self._client.models.generate_content(
                model=self._config.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return response.text or "{}"
        except Exception as exc:
            logger.error("视觉 LLM 请求失败：%s", exc)
            return "{}"

    def analyze_page(self, screenshot_path: str, prompt: str) -> dict[str, Any]:
        """Load a screenshot file and send it with a prompt. Returns parsed JSON."""
        path = Path(screenshot_path)
        if not path.exists():
            logger.warning("未找到截图文件：%s", screenshot_path)
            return {}

        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        raw_text = self.chat(prompt, image_b64)
        return self._parse_response_text(raw_text)

    def analyze_page_with_raw(
        self, screenshot_path: str, prompt: str
    ) -> tuple[dict[str, Any], str]:
        """Load a screenshot file and return both parsed JSON and raw text."""
        path = Path(screenshot_path)
        if not path.exists():
            logger.warning("未找到截图文件：%s", screenshot_path)
            return {}, "{}"

        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        raw_text = self.chat(prompt, image_b64)
        return self._parse_response_text(raw_text), raw_text

    def analyze_page_bytes(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """Send raw image bytes with a prompt. Returns parsed JSON."""
        if self._client is None:
            return {}

        try:
            response = self._client.models.generate_content(
                model=self._config.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text or "{}"
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"raw_text": response.text}
        except Exception as exc:
            logger.error("视觉 LLM 请求失败：%s", exc)
            return {}
