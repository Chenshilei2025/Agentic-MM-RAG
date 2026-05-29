"""System orchestration, lifecycle, evidence state, and tool profiles."""

from agentic_mm_rag.orchestrator.types import (
    AgentPlan,
    OrchestrationResult,
    QueryContext,
    ReflectionResult,
    RetrievalTask,
    SubagentResult,
)
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
    if name == "EvidenceBoard":
        from agentic_mm_rag.orchestrator.evidence.board import EvidenceBoard

        return EvidenceBoard
    if name in {"EvidenceBoardWriter", "ReadEvidenceTool", "WriteEvidenceTool"}:
        from agentic_mm_rag.orchestrator.evidence.io import (
            EvidenceBoardWriter,
            ReadEvidenceTool,
            WriteEvidenceTool,
        )

        return {
            "EvidenceBoardWriter": EvidenceBoardWriter,
            "ReadEvidenceTool": ReadEvidenceTool,
            "WriteEvidenceTool": WriteEvidenceTool,
        }[name]
    if name in {"EvidencePoolItem", "RefreshableEvidencePool"}:
        from agentic_mm_rag.orchestrator.evidence.pool import (
            EvidencePoolItem,
            RefreshableEvidencePool,
        )

        return {
            "EvidencePoolItem": EvidencePoolItem,
            "RefreshableEvidencePool": RefreshableEvidencePool,
        }[name]
    if name == "audit_evidence":
        from agentic_mm_rag.orchestrator.evidence.audit import audit_evidence

        return audit_evidence
    raise AttributeError(name)


__all__ = [
    "AgentPlan",
    "DECISION_AGENT_TOOL_NAMES",
    "DOC_GRAPH_SUBAGENT_TOOL_NAMES",
    "DOC_TEXT_SUBAGENT_TOOL_NAMES",
    "DOC_VISUAL_SUBAGENT_TOOL_NAMES",
    "EvidenceBoard",
    "EvidenceBoardWriter",
    "EvidencePoolItem",
    "Orchestrator",
    "OrchestratorContext",
    "OrchestrationResult",
    "PUBLIC_TOOL_NAMES",
    "QueryContext",
    "ReadEvidenceTool",
    "ReflectionResult",
    "RegistryBundle",
    "RefreshableEvidencePool",
    "RetrievalTask",
    "SubagentResult",
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
