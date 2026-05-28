"""OpenAI-compatible provider for runtime use."""

from __future__ import annotations

import json
import os
from typing import Any

from agentic_mm_rag.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OpenAIChatProvider(LLMProvider):
    """Thin async OpenAI Chat Completions adapter.

    The import is lazy so local retrieval does not require OpenAI packages.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        timeout_env = os.environ.get("AGENTIC_RAG_LLM_TIMEOUT_S")
        self.timeout_s = timeout_s or (float(timeout_env) if timeout_env else 90.0)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "openai package is required for OpenAIChatProvider; install openai"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.timeout_s:
                kwargs["timeout"] = self.timeout_s
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not model:
            raise ValueError("model is required")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._get_client().chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCallRequest] = []
        for idx, call in enumerate(message.tool_calls or []):
            arguments: dict[str, Any]
            try:
                parsed = json.loads(call.function.arguments or "{}")
                arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCallRequest(
                    id=call.id or f"call_{idx}",
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
