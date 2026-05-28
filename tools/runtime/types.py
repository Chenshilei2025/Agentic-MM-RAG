"""Shared request and result types for multimodal retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_mm_rag.schemas import EvidenceCard, ToolResponse


@dataclass(slots=True)
class QueryRequest:
    """Unified query payload routed through the tools runtime."""

    query_text: str
    top_k: int = 12
    corpora: tuple[str, ...] = ("doc", "video")
    modalities: tuple[str, ...] = ("text",)
    doc_root: str | None = None
    video_root: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryBundle:
    """Normalized retrieval result bundle for orchestration layers."""

    evidence: list[EvidenceCard] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_tool_response(self, tool: str, *, ok: bool = True) -> ToolResponse:
        return ToolResponse(
            ok=ok,
            tool=tool,
            evidence=self.evidence,
            data=self.data,
            warnings=self.warnings,
        )
