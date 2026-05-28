"""Runtime context contracts for tool construction and per-request injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RequestContext:
    """Per-request context that can be injected into context-aware tools."""

    session_key: str | None = None
    query_text: str | None = None
    corpus: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextAware(Protocol):
    """Protocol for tools that need request-scoped metadata."""

    def set_context(self, ctx: RequestContext) -> None:
        ...


@dataclass
class ToolContext:
    """Construction-time context for registry and tool loading."""

    workspace: str | None = None
    config: Any | None = None
    runtime_backend: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
