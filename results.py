"""Runtime result types for public agentic retrieval APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_mm_rag.orchestrator.types import AgentPlan, SubagentResult


@dataclass(slots=True)
class AgenticRunResult:
    """Final output of a batch-oriented agentic RAG run."""

    answer: str
    plan: AgentPlan
    results: list[SubagentResult]
    fused_evidence: list[dict[str, Any]]
    warnings: list[str]
    stage_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "plan": self.plan.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "fused_evidence": list(self.fused_evidence),
            "warnings": list(self.warnings),
            "stage_metrics": dict(self.stage_metrics),
        }


@dataclass(slots=True)
class ReflectionDecision:
    """Planner-visible reflection decision for second-pass control."""

    sufficient: bool
    answerable: bool
    needs_visual: bool = False
    needs_text: bool = False
    needs_graph: bool = False
    rationale: str = ""

    def to_hint(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "answerable": self.answerable,
            "needs_visual": self.needs_visual,
            "needs_text": self.needs_text,
            "needs_graph": self.needs_graph,
            "rationale": self.rationale,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_hint()
