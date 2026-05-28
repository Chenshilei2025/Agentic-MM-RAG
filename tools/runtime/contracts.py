"""Request and backend contracts for final seek tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_mm_rag.schemas import ToolResponse


@dataclass(slots=True)
class SeekRequest:
    query_vector: list[float]
    query_text: str | None = None
    top_k: int = 12
    min_score: float | None = None
    visual_query_vector: list[float] | None = None
    include_multimodal: bool = True
    include_mapped_segment_details: bool = False
    exact_detail_lexical: bool = False
    graph_top_k_entities: int | None = None
    graph_top_k_chunks: int | None = None
    edge_type_filter: list[str] | None = None
    doc_ids: list[str] | None = None
    evidence_pages: list[int] | None = None
    page_bias_pages: list[int] | None = None
    visual_anchors: list[str] | None = None
    visual_block_ids: list[str] | None = None
    segment_ids: list[str] | None = None
    graph_mode: str = "hybrid"


class RetrievalBackend(Protocol):
    """Corpus backend protocol used by final seek tools."""

    corpus_type: str

    async def doc_text_seek(self, request: SeekRequest) -> ToolResponse:
        ...

    async def doc_visual_seek(self, request: SeekRequest) -> ToolResponse:
        ...

    async def doc_graph_seek(self, request: SeekRequest) -> ToolResponse:
        ...

    async def video_text_seek(self, request: SeekRequest) -> ToolResponse:
        ...

    async def video_visual_seek(self, request: SeekRequest) -> ToolResponse:
        ...

    async def video_graph_seek(self, request: SeekRequest) -> ToolResponse:
        ...
