"""Shared lifecycle hook primitives for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_mm_rag.providers.base import LLMResponse, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to runner hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """Minimal lifecycle surface for shared runner customization."""

    async def before_iteration(self, context: AgentHookContext) -> None:
        return None

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        return None

    async def after_iteration(self, context: AgentHookContext) -> None:
        return None

    def finalize_content(
        self,
        context: AgentHookContext,
        content: str | None,
    ) -> str | None:
        return content


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks."""

    def __init__(self, hooks: list[AgentHook]) -> None:
        self._hooks = list(hooks)

    async def before_iteration(self, context: AgentHookContext) -> None:
        for hook in self._hooks:
            await hook.before_iteration(context)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for hook in self._hooks:
            await hook.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        for hook in self._hooks:
            await hook.after_iteration(context)

    def finalize_content(
        self,
        context: AgentHookContext,
        content: str | None,
    ) -> str | None:
        for hook in self._hooks:
            content = hook.finalize_content(context, content)
        return content
