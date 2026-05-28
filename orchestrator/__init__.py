"""System orchestration, lifecycle, evidence state, and tool profiles."""

from agentic_mm_rag.orchestrator.evidence_board import EvidenceBoard
from agentic_mm_rag.orchestrator.evidence_io import (
    EvidenceBoardWriter,
    ReadEvidenceTool,
    WriteEvidenceTool,
)
from agentic_mm_rag.orchestrator.evidence_pool import (
    EvidencePoolItem,
    RefreshableEvidencePool,
)
from agentic_mm_rag.orchestrator.evidence_audit import audit_evidence
from agentic_mm_rag.orchestrator.tools import (
    DECISION_AGENT_TOOL_NAMES,
    DOC_GRAPH_SUBAGENT_TOOL_NAMES,
    DOC_TEXT_SUBAGENT_TOOL_NAMES,
    DOC_VISUAL_SUBAGENT_TOOL_NAMES,
    PUBLIC_TOOL_NAMES,
    RegistryBundle,
    ToolRegistryProfile,
    VIDEO_GRAPH_SUBAGENT_TOOL_NAMES,
    VIDEO_TEXT_SUBAGENT_TOOL_NAMES,
    VIDEO_VISUAL_SUBAGENT_TOOL_NAMES,
    build_public_tool_registry,
    build_registry_bundle,
    public_tool_names,
)


def __getattr__(name: str):
    if name in {"Orchestrator", "OrchestratorContext"}:
        from agentic_mm_rag.orchestrator.loop import Orchestrator, OrchestratorContext

        return {
            "Orchestrator": Orchestrator,
            "OrchestratorContext": OrchestratorContext,
        }[name]
    raise AttributeError(name)

__all__ = [
    "DECISION_AGENT_TOOL_NAMES",
    "DOC_GRAPH_SUBAGENT_TOOL_NAMES",
    "DOC_TEXT_SUBAGENT_TOOL_NAMES",
    "DOC_VISUAL_SUBAGENT_TOOL_NAMES",
    "EvidenceBoard",
    "EvidenceBoardWriter",
    "EvidencePoolItem",
    "Orchestrator",
    "OrchestratorContext",
    "PUBLIC_TOOL_NAMES",
    "ReadEvidenceTool",
    "RegistryBundle",
    "RefreshableEvidencePool",
    "ToolRegistryProfile",
    "VIDEO_GRAPH_SUBAGENT_TOOL_NAMES",
    "VIDEO_TEXT_SUBAGENT_TOOL_NAMES",
    "VIDEO_VISUAL_SUBAGENT_TOOL_NAMES",
    "WriteEvidenceTool",
    "audit_evidence",
    "build_public_tool_registry",
    "build_registry_bundle",
    "public_tool_names",
]
