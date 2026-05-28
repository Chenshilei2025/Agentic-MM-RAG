"""Registry assembly for the final seek/evidence tool surface."""

from __future__ import annotations

from agentic_mm_rag.tools.context import ToolContext
from agentic_mm_rag.tools.doc import (
    doc_graph_seek,
    doc_text_seek,
    doc_visual_seek,
    register_doc_tools,
)
from agentic_mm_rag.tools.loader import ToolLoader
from agentic_mm_rag.tools.registry import ToolRegistry
from agentic_mm_rag.tools.video import (
    register_video_tools,
    video_graph_seek,
    video_text_seek,
    video_visual_seek,
)


def register_tools(registry: ToolRegistry, ctx: ToolContext | None = None) -> None:
    """Register the agent-visible doc/video seek tool surface."""

    register_doc_tools(registry)
    register_video_tools(registry)


def build_default_registry(context: ToolContext | None = None, *, scope: str = "core") -> ToolRegistry:
    registry = ToolRegistry()
    ToolLoader().load(registry, ctx=context, scope=scope)
    return registry


__all__ = [
    "build_default_registry",
    "doc_graph_seek",
    "doc_text_seek",
    "doc_visual_seek",
    "register_tools",
    "video_graph_seek",
    "video_text_seek",
    "video_visual_seek",
]
