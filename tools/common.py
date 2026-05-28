"""Shared constants and helpers for retrieval tool modules."""

from __future__ import annotations

from agentic_mm_rag.tools.runtime.backends import build_doc_backend, build_video_backend
from agentic_mm_rag.tools.schema import (
    ArraySchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
)


QUERY_VECTOR_SCHEMA = ArraySchema(
    NumberSchema(description="One embedding dimension."),
    description="Query embedding vector. It must match the target vector store dimension.",
    min_items=1,
)
TOP_K_SCHEMA = IntegerSchema(
    description=(
        "Maximum number of results to return. Subagents should set this explicitly "
        "from their retrieval_budget for each seek call."
    ),
    minimum=1,
)
MIN_SCORE_SCHEMA = NumberSchema(
    description="Optional minimum cosine similarity score.",
    nullable=True,
)
DOC_ROOT_SCHEMA = StringSchema(
    "Optional processed RAGAnything storage root override.",
    nullable=True,
)
VIDEO_ROOT_SCHEMA = StringSchema(
    "Optional processed VideoRAG storage root override.",
    nullable=True,
)
DOC_IDS_SCHEMA = ArraySchema(
    StringSchema("Document id."),
    description="Optional list of document ids to restrict retrieval.",
    nullable=True,
)
PAGE_NUMBERS_SCHEMA = ArraySchema(
    IntegerSchema("One-indexed document page number."),
    description="Optional document page numbers to restrict or bias retrieval.",
    nullable=True,
)
SEGMENT_IDS_SCHEMA = ArraySchema(
    StringSchema("Video segment id, for example video_3."),
    description="Seed segment ids to expand.",
    min_items=1,
)
OPTIONAL_CHUNK_IDS_SCHEMA = ArraySchema(
    StringSchema("Document chunk id."),
    description="Candidate document chunk ids to inspect locally.",
    min_items=1,
    nullable=True,
)
OPTIONAL_SEGMENT_IDS_SCHEMA = ArraySchema(
    StringSchema("Video segment id, for example video_3."),
    description="Candidate video segment ids to inspect locally.",
    min_items=1,
    nullable=True,
)
EVIDENCE_ITEMS_SCHEMA = ArraySchema(
    ObjectSchema(additional_properties=True),
    description="Serialized EvidenceCard dictionaries from seek/expand tools.",
    min_items=1,
)
MODALITY_FILTER_SCHEMA = ArraySchema(
    StringSchema(
        "Allowed modality.",
        enum=["image", "chart", "table", "equation", "page_footnote", "aside_text", "code", "unknown"],
    ),
    description="Optional modality filter.",
    nullable=True,
)
GRAPH_EDGE_FILTER_SCHEMA = ArraySchema(
    StringSchema("Allowed graph edge type.", enum=["semantic", "temporal", "table", "visual"]),
    description="Optional graph edge type filter.",
    nullable=True,
)
VISUAL_QUERY_VECTOR_SCHEMA = ArraySchema(
    NumberSchema(description="One visual embedding dimension."),
    description="Optional visual/text-to-video query vector matching the visual segment store dimension.",
    min_items=1,
    nullable=True,
)
REFLECTION_HINT_SCHEMA = ObjectSchema(
    additional_properties=True,
    description="Planner reflection hint passed into inspection for answer or gap-focused candidate scoring.",
    nullable=True,
)
INSPECTION_MODE_SCHEMA = StringSchema(
    "Inspection mode.",
    enum=["answer_mode", "gap_mode"],
    nullable=True,
)


def doc_backend(doc_root: str | None = None):
    return build_doc_backend(root=doc_root)


def video_backend(video_root: str | None = None):
    return build_video_backend(root=video_root)
