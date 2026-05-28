"""Shared schemas returned by retrieval and inspection tools."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


SourceType = Literal["doc", "video"]
Modality = Literal[
    "text",
    "image",
    "chart",
    "table",
    "equation",
    "page_footnote",
    "aside_text",
    "code",
    "video_segment",
    "entity",
    "relation",
    "unknown",
]


@dataclass(slots=True)
class Locator:
    """Human- and machine-readable source location."""

    file_path: str | None = None
    doc_id: str | None = None
    page_idx: int | None = None
    video_id: str | None = None
    segment_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    raw_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(slots=True)
class ScoreParts:
    """Explainable score components for fused evidence."""

    text: float = 0.0
    graph: float = 0.0
    visual: float = 0.0
    source_filter: float = 0.0
    rerank: float = 0.0

    def total(self) -> float:
        return self.text + self.graph + self.visual + self.source_filter + self.rerank

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceCard:
    """A normalized retrieval result for agentic reasoning."""

    id: str
    source_type: SourceType
    modality: Modality
    source_id: str
    locator: Locator
    content: str
    score: float = 0.0
    score_parts: ScoreParts = field(default_factory=ScoreParts)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, content_chars: int | None = None) -> dict[str, Any]:
        content = self.content
        if content_chars is not None and len(content) > content_chars:
            content = content[:content_chars].rstrip() + "..."
        return {
            "id": self.id,
            "source_type": self.source_type,
            "modality": self.modality,
            "source_id": self.source_id,
            "locator": self.locator.to_dict(),
            "content": content,
            "score": self.score,
            "score_parts": self.score_parts.to_dict(),
            "provenance": self.provenance,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ToolResponse:
    """Common response envelope for tools."""

    ok: bool
    tool: str
    evidence: list[EvidenceCard] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self, *, content_chars: int | None = 1200) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "evidence": [
                item.to_dict(content_chars=content_chars) for item in self.evidence
            ],
            "data": self.data,
            "warnings": self.warnings,
            "error": self.error,
        }
