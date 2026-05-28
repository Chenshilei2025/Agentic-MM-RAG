"""Video seek tools exposed to video subagents."""

from __future__ import annotations

from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.runtime.contracts import SeekRequest
from agentic_mm_rag.tools.base import FunctionTool
from agentic_mm_rag.tools.common import (
    MIN_SCORE_SCHEMA,
    QUERY_VECTOR_SCHEMA,
    TOP_K_SCHEMA,
    VIDEO_ROOT_SCHEMA,
    VISUAL_QUERY_VECTOR_SCHEMA,
    video_backend,
)
from agentic_mm_rag.tools.metadata import seek_metadata
from agentic_mm_rag.tools.registry import ToolRegistry
from agentic_mm_rag.tools.schema import ArraySchema, BooleanSchema, StringSchema, tool_parameters_schema


async def video_text_seek(
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int = 10,
    min_score: float | None = None,
    video_root: str | None = None,
    include_mapped_segment_details: bool = False,
    exact_detail_lexical: bool = False,
) -> ToolResponse:
    response = await video_backend(video_root).video_text_seek(
        SeekRequest(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            include_mapped_segment_details=include_mapped_segment_details,
            exact_detail_lexical=exact_detail_lexical,
        )
    )
    response.tool = "video_text_seek"
    return response


async def video_visual_seek(
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int = 8,
    min_score: float | None = None,
    video_root: str | None = None,
    segment_ids: list[str] | None = None,
    visual_query_vector: list[float] | None = None,
) -> ToolResponse:
    response = await video_backend(video_root).video_visual_seek(
        SeekRequest(
            query_vector=query_vector,
            visual_query_vector=visual_query_vector,
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            segment_ids=segment_ids,
        )
    )
    response.tool = "video_visual_seek"
    if segment_ids:
        response.data.setdefault("requested_segment_ids", list(segment_ids))
    return response


async def video_graph_seek(
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int = 10,
    top_k_entities: int = 10,
    top_k_chunks: int = 2,
    min_score: float | None = None,
    graph_strategy: str = "hybrid",
    video_root: str | None = None,
) -> ToolResponse:
    response = await video_backend(video_root).video_graph_seek(
        SeekRequest(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            graph_mode=graph_strategy,
            graph_top_k_entities=top_k_entities,
            graph_top_k_chunks=top_k_chunks,
        )
    )
    response.tool = "video_graph_seek"
    return response


def register_video_tools(registry: ToolRegistry) -> None:
    registry.register(
        FunctionTool(
            "video_text_seek",
            "Seek video ASR, subtitle, caption, or OCR text chunks by vector similarity.",
            video_text_seek,
            tool_parameters_schema(
                query_vector=QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Original or rewritten query text for lexical score fusion.", nullable=True),
                top_k=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                video_root=VIDEO_ROOT_SCHEMA,
                include_mapped_segment_details=BooleanSchema(
                    description="Append mapped segment transcript/caption details to chunk evidence.",
                    default=False,
                ),
                exact_detail_lexical=BooleanSchema(
                    description="Boost quoted phrases, names, and hyphenated terms during lexical candidate scoring.",
                    default=False,
                ),
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("video", "text"),
        )
    )
    registry.register(
        FunctionTool(
            "video_visual_seek",
            "Seek video keyframes and candidate segments by visual vector similarity.",
            video_visual_seek,
            tool_parameters_schema(
                query_vector=VISUAL_QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Original or rewritten visual query text for lexical score fusion.", nullable=True),
                top_k=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                video_root=VIDEO_ROOT_SCHEMA,
                visual_query_vector=VISUAL_QUERY_VECTOR_SCHEMA,
                segment_ids=ArraySchema(
                    StringSchema("Video segment id."),
                    description="Optional segment ids to trace to nearby frames.",
                    nullable=True,
                ),
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("video", "visual"),
        )
    )
    registry.register(
        FunctionTool(
            "video_graph_seek",
            "Seek video perception-graph evidence with entity, event, and relation paths.",
            video_graph_seek,
            tool_parameters_schema(
                query_vector=QUERY_VECTOR_SCHEMA,
                query_text=StringSchema("Graph-oriented query text.", nullable=True),
                top_k=TOP_K_SCHEMA,
                top_k_entities=TOP_K_SCHEMA,
                top_k_chunks=TOP_K_SCHEMA,
                min_score=MIN_SCORE_SCHEMA,
                graph_strategy=StringSchema("Graph seek strategy.", enum=["hybrid", "light_graph"]),
                video_root=VIDEO_ROOT_SCHEMA,
                required=["query_vector"],
            ),
            _manifest_metadata=seek_metadata("video", "graph"),
        )
    )
