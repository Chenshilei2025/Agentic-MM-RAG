"""Structured contracts for the multi-agent retrieval workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agentic_mm_rag.agent.contracts import ToolIntent


CorpusKind = Literal["doc", "video"]
SubagentKind = Literal[
    "doc_text_subagent",
    "doc_visual_subagent",
    "doc_graph_subagent",
    "video_text_subagent",
    "video_visual_subagent",
    "video_graph_subagent",
]
TaskStatus = Literal["pending", "running", "done", "error"]


@dataclass(slots=True)
class QueryContext:
    """Normalized query inputs shared by agents."""

    query_text: str
    doc_query_vector: list[float] | None = None
    video_query_vector: list[float] | None = None
    visual_query_vector: list[float] | None = None
    top_k: int = 12
    doc_root: str | None = None
    video_root: str | None = None
    source_doc_id: str | None = None
    candidate_doc_ids: list[str] | None = None
    doc_profile: dict[str, Any] | None = None
    query_profile: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "doc_query_vector": self.doc_query_vector,
            "video_query_vector": self.video_query_vector,
            "visual_query_vector": self.visual_query_vector,
            "top_k": self.top_k,
            "doc_root": self.doc_root,
            "video_root": self.video_root,
            "source_doc_id": self.source_doc_id,
            "candidate_doc_ids": list(self.candidate_doc_ids) if self.candidate_doc_ids else None,
            "doc_profile": dict(self.doc_profile) if self.doc_profile else None,
            "query_profile": dict(self.query_profile) if self.query_profile else None,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RetrievalTask:
    """A bounded task assigned to one expert subagent."""

    id: str
    agent: SubagentKind
    corpus: CorpusKind
    tool_name: str
    params: dict[str, Any]
    rationale: str
    intent: ToolIntent = "doc_text_seek"
    query: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)
    stop_condition: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "corpus": self.corpus,
            "tool_name": self.tool_name,
            "params": dict(self.params),
            "rationale": self.rationale,
            "intent": self.intent,
            "query": self.query,
            "allowed_tools": list(self.allowed_tools),
            "expected_evidence": list(self.expected_evidence),
            "stop_condition": self.stop_condition,
            "depends_on": list(self.depends_on),
            "status": self.status,
        }


@dataclass(slots=True)
class AgentPlan:
    """Decision agent output."""

    query_context: QueryContext
    tasks: list[RetrievalTask]
    rationale: str
    expected_modalities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_context": self.query_context.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "rationale": self.rationale,
            "expected_modalities": list(self.expected_modalities),
        }


@dataclass(slots=True)
class SubagentResult:
    """Result returned by an expert subagent."""

    task: RetrievalTask
    ok: bool
    evidence: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "ok": self.ok,
            "evidence": list(self.evidence),
            "data": dict(self.data),
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass(slots=True)
class ReflectionResult:
    """Decision agent reflection after initial subagent results."""

    sufficient: bool
    reason: str
    new_tasks: list[RetrievalTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "new_tasks": [task.to_dict() for task in self.new_tasks],
        }


@dataclass(slots=True)
class OrchestrationResult:
    """Final orchestrator output."""

    answer: str
    plan: AgentPlan
    subagent_results: list[SubagentResult]
    fused_evidence: list[dict[str, Any]]
    reflection: ReflectionResult
    warnings: list[str] = field(default_factory=list)
    generation_context: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "plan": self.plan.to_dict(),
            "subagent_results": [result.to_dict() for result in self.subagent_results],
            "fused_evidence": list(self.fused_evidence),
            "reflection": self.reflection.to_dict(),
            "warnings": list(self.warnings),
            "generation_context": dict(self.generation_context),
            "route": dict(self.route),
            "trace": list(self.trace),
        }
