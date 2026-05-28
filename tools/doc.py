"""Document seek tools exposed to document subagents."""

from __future__ import annotations

from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.runtime.contracts import SeekRequest
from agentic_mm_rag.tools.base import FunctionTool
from agentic_mm_rag.tools.common import (
    DOC_IDS_SCHEMA,
    DOC_ROOT_SCHEMA,
    GRAPH_EDGE_FILTER_SCHEMA,
    MIN_SCORE_SCHEMA,
    MODALITY_FILTER_SCHEMA,
    PAGE_NUMBERS_SCHEMA,
    QUERY_VECTOR_SCHEMA,
    TOP_K_SCHEMA,
    doc_backend,
)
from agentic_mm_rag.tools.metadata import seek_metadata
from agentic_mm_rag.tools.registry import ToolRegistry
from agentic_mm_rag.tools.schema import ArraySchema, BooleanSchema, StringSchema, tool_parameters_schema


async def doc_text_seek(
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int = 10,
    min_score: float | None = None,
    doc_ids: list[str] | None = None,
    evidence_pages: list[int] | None = None,
    page_bias_pages: list[int] | None = None,
    include_multimodal: bool = True,
    doc_root: str | None = None,
) -> ToolResponse:
    response = await doc_backend(doc_root).doc_text_seek(
        SeekRequest(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            include_multimodal=include_multimodal,
            doc_ids=doc_ids,
            evidence_pages=evidence_pages,
            page_bias_pages=page_bias_pages,
        )
    )
    response.tool = "doc_text_seek"
    return response


async def doc_visual_seek(
    query_vector: list[float],
    query_text: str | None = None,
    modalities: list[str] | None = None,
    top_k: int = 10,
    min_score: float | None = None,
    doc_ids: list[str] | None = None,
    evidence_pages: list[int] | None = None,
    page_bias_pages: list[int] | None = None,
    visual_block_ids: list[str] | None = None,
    doc_root: str | None = None,
) -> ToolResponse:
    response = await doc_backend(doc_root).doc_visual_seek(
        SeekRequest(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            visual_anchors=modalities,
            visual_block_ids=visual_block_ids,
            doc_ids=doc_ids,
            evidence_pages=evidence_pages,
            page_bias_pages=page_bias_pages,
        )
    )
    response.tool = "doc_visual_seek"
    if visual_block_ids:
        response.data.setdefault("requested_visual_block_ids", list(visual_block_ids))
    return response


async def doc_graph_seek(
    query_vector: list[float],
    query_text: str | None = None,
    top_k_entities: int = 10,
    top_k_chunks: int = 10,
    min_score: float | None = None,
    graph_strategy: str = "hybrid",
    edge_type_filter: list[str] | None = None,
    doc_ids: list[str] | None = None,
    evidence_pages: list[int] | None = None,
    page_bias_pages: list[int] | None = None,
    doc_root: str | None = None,
) -> ToolResponse:
    response = await doc_backend(doc_root).doc_graph_seek(
        SeekRequest(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k_chunks,
            min_score=min_score,
            graph_mode=graph_strategy,
            graph_top_k_entities=top_k_entities,
            graph_top_k_chunks=top_k_chunks,
            edge_type_filter=edge_type_filter,
            doc_ids=doc_ids,
            evidence_pages=evidence_pages,
            page_bias_pages=page_bias_pages,
        )
    )
    response.tool = "doc_graph_seek"
    return response


def register_doc_tools(registry: ToolRegistry) -> None:
    registry.register(
        FunctionTool(
            "doc_text_seek",
            "Seek document text chunks by text vector similarity.",
            doc_text_seek,
            tool_parameters_schema(
                query_vector=QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Original or rewritten query text for lexical score fusion.", nullable=True),
                top_k=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                doc_ids=DOC_IDS_SCHEMA,
                include_multimodal=BooleanSchema(
                    description="Whether linked multimodal blocks may be surfaced.",
                    default=True,
                ),
                evidence_pages=PAGE_NUMBERS_SCHEMA,
                page_bias_pages=PAGE_NUMBERS_SCHEMA,
                doc_root=DOC_ROOT_SCHEMA,
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("doc", "text"),
        )
    )
    registry.register(
        FunctionTool(
            "doc_visual_seek",
            "Seek document visual blocks and source image paths by visual query or linked block id.",
            doc_visual_seek,
            tool_parameters_schema(
                query_vector=QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Original or rewritten query text for visual/text score fusion.", nullable=True),
                modalities=MODALITY_FILTER_SCHEMA,
                top_k=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                doc_ids=DOC_IDS_SCHEMA,
                evidence_pages=PAGE_NUMBERS_SCHEMA,
                page_bias_pages=PAGE_NUMBERS_SCHEMA,
                visual_block_ids=ArraySchema(
                    StringSchema("Document visual block id."),
                    description="Optional known visual block ids to trace.",
                    nullable=True,
                ),
                doc_root=DOC_ROOT_SCHEMA,
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("doc", "visual"),
        )
    )
    registry.register(
        FunctionTool(
            "doc_graph_seek",
            "Seek document graph evidence with explicit entity paths and edge semantics.",
            doc_graph_seek,
            tool_parameters_schema(
                query_vector=QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Graph-oriented query text.", nullable=True),
                top_k_entities=TOP_K_SCHEMA,
                top_k_chunks=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                graph_strategy=StringSchema("Graph seek strategy.", enum=["hybrid", "light_graph"]),
                edge_type_filter=GRAPH_EDGE_FILTER_SCHEMA,
                doc_ids=DOC_IDS_SCHEMA,
                evidence_pages=PAGE_NUMBERS_SCHEMA,
                page_bias_pages=PAGE_NUMBERS_SCHEMA,
                doc_root=DOC_ROOT_SCHEMA,
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("doc", "graph"),
        )
    )
