"""Thin wrapper over the Google Generative AI SDK for Gemini."""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from game_agent.config import GeminiConfig
from game_agent.exceptions import GeminiError, GeminiRateLimitError

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


@dataclass
class FunctionCall:
    """A parsed function call from the LLM response."""

    name: str
    args: dict[str, Any]


@dataclass
class GeminiResponse:
    """Parsed response from Gemini."""

    text: str | None = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    raw_response: Any = None


class GeminiClient:
    """Gemini API client with retry logic and function calling support."""

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        self._client: Any = None
        self._system_prompt = ""
        self._tools: list[dict] | None = None
        self._init_model()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "429" in text
            or "rate limit" in text
            or "resource_exhausted" in text
            or "quota" in text
            or "too many requests" in text
        )

    def _init_model(self) -> None:
        try:
            if not HAS_GENAI:
                raise ImportError
            if not self._config.api_key:
                logger.warning("未提供 Gemini API Key，文本模型调用不可用")
                return

            self._client = genai.Client(
                vertexai=True,
                api_key=self._config.api_key,
                http_options=types.HttpOptions(api_version="v1"),
            )
        except ImportError:
            logger.warning("未安装 google-genai，Gemini 调用将不可用")
        except Exception as exc:
            logger.warning("初始化 Gemini 模型失败：%s", exc)

    @staticmethod
    def _extract_function_call_from_text(text: str | None) -> FunctionCall | None:
        if not text:
            return None

        candidates = [text.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            action = (
                payload.get("action")
                or payload.get("tool")
                or payload.get("name")
                or payload.get("动作")
            )
            params = (
                payload.get("params")
                or payload.get("parameters")
                or payload.get("arguments")
                or payload.get("args")
                or payload.get("参数")
                or {}
            )
            if isinstance(action, str) and isinstance(params, dict):
                return FunctionCall(name=action, args=params)
            if isinstance(action, str):
                inline_call = GeminiClient._parse_inline_action_call(action)
                if inline_call is not None:
                    return inline_call

        # Fallback: scan for inline action call pattern like "动作：poco_click(...)"
        inline_match = re.search(
            r"(?:动作|action)\s*[:：]\s*([a-zA-Z_]\w*\(.*?\))", text, flags=re.S
        )
        if inline_match:
            inline_call = GeminiClient._parse_inline_action_call(inline_match.group(1))
            if inline_call is not None:
                return inline_call

        return None

    @staticmethod
    def _parse_inline_action_call(action_text: str) -> FunctionCall | None:
        match = re.match(r"^\s*([a-zA-Z_][\w]*)\((.*)\)\s*$", action_text, flags=re.S)
        if not match:
            return None

        fn_name = match.group(1)
        args_text = match.group(2).strip()
        if not args_text:
            return FunctionCall(name=fn_name, args={})

        try:
            expr = ast.parse(f"_f({args_text})", mode="eval")
        except SyntaxError:
            return None

        call = expr.body
        if not isinstance(call, ast.Call):
            return None

        parsed_args: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                return None
            try:
                parsed_args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                return None
        return FunctionCall(name=fn_name, args=parsed_args)

    def start_chat(self, system_prompt: str, tools: list[dict] | None = None) -> None:
        if self._client is None:
            raise GeminiError("Gemini 模型尚未初始化")
        self._system_prompt = system_prompt
        self._tools = tools

    def _build_generation_config(
        self,
        *,
        tools: list[dict] | None = None,
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
        enable_thinking: bool = False,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if tools:
            config_kwargs["tools"] = tools
        if enable_thinking and HAS_GENAI:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                )
            except Exception:
                pass
        return types.GenerateContentConfig(**config_kwargs)

    def _generate_content(
        self,
        contents: Any,
        *,
        system_prompt: str | None,
        tools: list[dict] | None,
        response_mime_type: str | None = None,
        enable_thinking: bool = False,
    ) -> Any:
        if self._client is None:
            raise GeminiError("Gemini 模型尚未初始化")
        return self._client.models.generate_content(
            model=self._config.model_name,
            contents=contents,
            config=self._build_generation_config(
                tools=tools,
                system_instruction=system_prompt,
                response_mime_type=response_mime_type,
                enable_thinking=enable_thinking,
            ),
        )

    def send_message(self, content: str, retry_count: int = 3) -> GeminiResponse:
        if not self._system_prompt:
            raise GeminiError("对话尚未启动，请先调用 start_chat()")

        for attempt in range(retry_count):
            try:
                response = self._generate_content(
                    content,
                    system_prompt=self._system_prompt,
                    tools=self._tools,
                    enable_thinking=True,
                )
                return self._parse_response(response)
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    wait = 2 ** (attempt + 1)
                    logger.warning("触发限流，等待 %d 秒后重试...", wait)
                    time.sleep(wait)
                    if attempt == retry_count - 1:
                        raise GeminiRateLimitError(str(exc)) from exc
                else:
                    raise GeminiError(f"Gemini API 调用失败：{exc}") from exc
        raise GeminiError("重试次数已耗尽")

    def send_multimodal(
        self, text: str, image_b64: str, retry_count: int = 3
    ) -> GeminiResponse:
        if not self._system_prompt:
            raise GeminiError("对话尚未启动，请先调用 start_chat()")

        image_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        for attempt in range(retry_count):
            try:
                response = self._generate_content(
                    [text, image_part],
                    system_prompt=self._system_prompt,
                    tools=self._tools,
                    enable_thinking=True,
                )
                return self._parse_response(response)
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    wait = 2 ** (attempt + 1)
                    logger.warning("触发限流，等待 %d 秒后重试...", wait)
                    time.sleep(wait)
                    if attempt == retry_count - 1:
                        raise GeminiRateLimitError(str(exc)) from exc
                else:
                    raise GeminiError(f"Gemini API 调用失败：{exc}") from exc
        raise GeminiError("重试次数已耗尽")

    def single_prompt(self, prompt: str, retry_count: int = 3) -> GeminiResponse:
        """One-shot prompt without chat context (for annotation, decomposition)."""
        if self._client is None:
            raise GeminiError("Gemini 模型尚未初始化")

        for attempt in range(retry_count):
            try:
                response = self._generate_content(
                    prompt,
                    system_prompt=None,
                    tools=None,
                    response_mime_type=self._config.response_mime_type,
                )
                return self._parse_response(response)
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    wait = 2 ** (attempt + 1)
                    logger.warning("触发限流，等待 %d 秒后重试...", wait)
                    time.sleep(wait)
                    if attempt == retry_count - 1:
                        raise GeminiRateLimitError(str(exc)) from exc
                else:
                    raise GeminiError(f"Gemini API 调用失败：{exc}") from exc
        raise GeminiError("重试次数已耗尽")

    def _parse_response(self, response: Any) -> GeminiResponse:
        text_parts: list[str] = []
        thought_parts: list[str] = []
        function_calls: list[FunctionCall] = []

        try:
            if hasattr(response, "function_calls") and response.function_calls:
                for fc in response.function_calls:
                    function_calls.append(
                        FunctionCall(name=fc.name, args=dict(fc.args or {}))
                    )

            if getattr(response, "candidates", None):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought and hasattr(part, "text") and part.text:
                        thought_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call is not None and part.function_call.name:
                        fc = part.function_call
                        if FunctionCall(name=fc.name, args=dict(fc.args or {})) not in function_calls:
                            function_calls.append(
                                FunctionCall(name=fc.name, args=dict(fc.args or {}))
                            )
                    elif hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
            elif getattr(response, "text", None):
                text_parts.append(response.text)
        except (IndexError, AttributeError) as exc:
            logger.warning("解析 Gemini 响应失败：%s", exc)

        if thought_parts and not text_parts:
            text_parts = thought_parts

        text = "\n".join(text_parts) if text_parts else None
        if not function_calls:
            fallback_call = self._extract_function_call_from_text(text)
            if fallback_call is not None:
                logger.info("从文本 JSON 回退解析到工具调用：%s", fallback_call.name)
                function_calls.append(fallback_call)

        return GeminiResponse(
            text=text,
            function_calls=function_calls,
            raw_response=response,
        )
