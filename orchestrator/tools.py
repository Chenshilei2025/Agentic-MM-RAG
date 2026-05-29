"""Builders for final agentic multimodal RAG registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agentic_mm_rag.tools.names import PUBLIC_TOOL_NAMES
from agentic_mm_rag.tools.registry import ToolRegistry


ToolRegistryProfile = Literal[
    "default",
    "decision_agent",
    "doc_text_subagent",
    "doc_visual_subagent",
    "doc_graph_subagent",
    "video_text_subagent",
    "video_visual_subagent",
    "video_graph_subagent",
]

DOC_TEXT_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_DOC_TEXT = ("doc_text_seek", "write_evidence")
DOC_VISUAL_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_DOC_VISUAL = ("doc_visual_seek", "write_evidence")
DOC_GRAPH_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_DOC_GRAPH = ("doc_graph_seek", "write_evidence")
VIDEO_TEXT_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_VIDEO_TEXT = ("video_text_seek", "write_evidence")
VIDEO_VISUAL_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_VIDEO_VISUAL = ("video_visual_seek", "write_evidence")
VIDEO_GRAPH_SUBAGENT_TOOL_NAMES = PROFILE_TOOL_NAMES_VIDEO_GRAPH = ("video_graph_seek", "write_evidence")
DECISION_AGENT_TOOL_NAMES = ("read_evidence",)

PROFILE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "default": PUBLIC_TOOL_NAMES,
    "decision_agent": DECISION_AGENT_TOOL_NAMES,
    "doc_text_subagent": DOC_TEXT_SUBAGENT_TOOL_NAMES,
    "doc_visual_subagent": DOC_VISUAL_SUBAGENT_TOOL_NAMES,
    "doc_graph_subagent": DOC_GRAPH_SUBAGENT_TOOL_NAMES,
    "video_text_subagent": VIDEO_TEXT_SUBAGENT_TOOL_NAMES,
    "video_visual_subagent": VIDEO_VISUAL_SUBAGENT_TOOL_NAMES,
    "video_graph_subagent": VIDEO_GRAPH_SUBAGENT_TOOL_NAMES,
}


@dataclass(slots=True)
class RegistryBundle:
    """Named registry bundle used by orchestration and public APIs."""

    internal_tools: ToolRegistry
    public_tools: ToolRegistry
    profile_tools: dict[str, ToolRegistry]

    @property
    def tools(self) -> ToolRegistry:
        return self.internal_tools

    @property
    def default_profile(self) -> ToolRegistry:
        return self.public_tools


def _subset_registry(source: ToolRegistry, names: tuple[str, ...]) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        tool = source.get(name)
        if tool is not None:
            registry.register(tool)
    return registry


def public_tool_names(profile: ToolRegistryProfile = "default") -> tuple[str, ...]:
    try:
        return PROFILE_TOOL_NAMES[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported tool registry profile: {profile}") from exc


def build_public_tool_registry(
    source: ToolRegistry,
    *,
    profile: ToolRegistryProfile = "default",
) -> ToolRegistry:
    """Build a curated registry for one agent contract profile."""

    return _subset_registry(source, public_tool_names(profile))


def build_registry_bundle(
    *,
    evidence_board: Any | None = None,
    evidence_writer: Any | None = None,
) -> RegistryBundle:
    """Construct the final shared tool registry bundle."""

    from agentic_mm_rag.orchestrator.evidence.board import EvidenceBoard
    from agentic_mm_rag.orchestrator.evidence.io import (
        ReadEvidenceTool,
        WriteEvidenceTool,
        EvidenceBoardWriter,
    )
    from agentic_mm_rag.tools.surface import build_default_registry

    internal_tools = build_default_registry()
    board = evidence_board or getattr(evidence_writer, "board", None) or EvidenceBoard()
    writer = evidence_writer or EvidenceBoardWriter(board)
    internal_tools.register(ReadEvidenceTool(board))
    internal_tools.register(WriteEvidenceTool(writer))
    profile_tools = {
        profile: build_public_tool_registry(internal_tools, profile=profile) for profile in PROFILE_TOOL_NAMES
    }
    return RegistryBundle(
        internal_tools=internal_tools,
        public_tools=profile_tools["default"],
        profile_tools=profile_tools,
    )
