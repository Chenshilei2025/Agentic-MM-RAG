"""Agentic multimodal RAG framework.

This package is intentionally storage-native: it wraps already-processed
document and video retrieval workdirs without rebuilding their indexes.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import re

from agentic_mm_rag.batch_runner import BatchRAGRunner, AgenticRunResult
from agentic_mm_rag.config import DEFAULT_MODELS, DEFAULT_PATHS, CorpusPaths, ModelDefaults
from agentic_mm_rag.api import (
    AgenticMMRAG,
    BatchRunSession,
    OrchestratorSession,
    create_batch_session,
    create_orchestrator_session,
    create_runtime,
)
from agentic_mm_rag.agent import (
    AgentPlan,
    DecisionAgent,
    OrchestrationResult,
    QueryContext,
    ReflectionResult,
    RetrievalTask,
    SubagentResult,
)
from agentic_mm_rag.orchestrator import Orchestrator
from agentic_mm_rag.orchestrator.evidence_pool import EvidencePoolItem, RefreshableEvidencePool
from agentic_mm_rag.schemas import EvidenceCard, Locator, ScoreParts
from agentic_mm_rag.tools.runtime import (
    RetrievalBackend,
    QueryBundle,
    QueryRequest,
    RuntimeBackendConfig,
    RuntimeQueryBackend,
    RuntimeQueryRequest,
    SeekRequest,
    build_backend,
    build_corpus_adapter,
    build_doc_backend,
    build_runtime_query_backend,
    build_video_backend,
)
from agentic_mm_rag.runtime import AgenticRuntime
from agentic_mm_rag.tools import ToolMetadata, infer_tool_metadata, tool_metadata
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


def _read_pyproject_version() -> str | None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return None
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _resolve_version() -> str:
    try:
        return _pkg_version("agentic-mm-rag")
    except PackageNotFoundError:
        return _read_pyproject_version() or "0.1.0"


__version__ = _resolve_version()

__all__ = [
    "BatchRAGRunner",
    "AgenticMMRAG",
    "AgenticRunResult",
    "AgentPlan",
    "DECISION_AGENT_TOOL_NAMES",
    "DEFAULT_MODELS",
    "DEFAULT_PATHS",
    "DecisionAgent",
    "DOC_GRAPH_SUBAGENT_TOOL_NAMES",
    "DOC_TEXT_SUBAGENT_TOOL_NAMES",
    "DOC_VISUAL_SUBAGENT_TOOL_NAMES",
    "EvidenceCard",
    "EvidencePoolItem",
    "BatchRunSession",
    "CorpusPaths",
    "Locator",
    "PUBLIC_TOOL_NAMES",
    "QueryBundle",
    "QueryRequest",
    "RegistryBundle",
    "RuntimeBackendConfig",
    "RuntimeQueryBackend",
    "RuntimeQueryRequest",
    "OrchestratorSession",
    "OrchestrationResult",
    "Orchestrator",
    "ModelDefaults",
    "QueryContext",
    "ReflectionResult",
    "RefreshableEvidencePool",
    "RetrievalTask",
    "ScoreParts",
    "SeekRequest",
    "RetrievalBackend",
    "AgenticRuntime",
    "SubagentResult",
    "ToolRegistryProfile",
    "ToolMetadata",
    "VIDEO_GRAPH_SUBAGENT_TOOL_NAMES",
    "VIDEO_TEXT_SUBAGENT_TOOL_NAMES",
    "VIDEO_VISUAL_SUBAGENT_TOOL_NAMES",
    "__version__",
    "build_backend",
    "build_corpus_adapter",
    "build_doc_backend",
    "build_public_tool_registry",
    "build_registry_bundle",
    "build_runtime_query_backend",
    "build_video_backend",
    "create_batch_session",
    "create_orchestrator_session",
    "create_runtime",
    "infer_tool_metadata",
    "public_tool_names",
    "tool_metadata",
]
