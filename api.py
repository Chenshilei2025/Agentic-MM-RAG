"""High-level programmatic facade for agentic multimodal RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_mm_rag.agent.runtime_types import AgenticRunResult
from agentic_mm_rag.agent.types import OrchestrationResult, QueryContext
from agentic_mm_rag.orchestrator.loop import Orchestrator
from agentic_mm_rag.runtime import AgenticRuntime
from agentic_mm_rag.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgenticMMRAG:
    """Programmatic facade for the multimodal RAG orchestrator."""

    orchestrator: Orchestrator

    @classmethod
    def from_defaults(
        cls,
        *,
        tools: ToolRegistry | None = None,
        provider: Any,
        **orchestrator_kwargs: Any,
    ) -> "AgenticMMRAG":
        return cls(
            orchestrator=Orchestrator(
                tools=tools,
                provider=provider,
                **orchestrator_kwargs,
            )
        )

    async def run(self, query: QueryContext) -> OrchestrationResult:
        return await self.orchestrator.run(query)


@dataclass(slots=True)
class OrchestratorSession:
    """Thin session wrapper for callers that prefer explicit sessions."""

    app: AgenticMMRAG

    @classmethod
    def create(
        cls,
        *,
        tools: ToolRegistry | None = None,
        provider: Any,
        **orchestrator_kwargs: Any,
    ) -> "OrchestratorSession":
        return cls(
            app=AgenticMMRAG.from_defaults(
                tools=tools,
                provider=provider,
                **orchestrator_kwargs,
            )
        )

    @property
    def orchestrator(self) -> Orchestrator:
        return self.app.orchestrator

    async def run(self, query: QueryContext) -> OrchestrationResult:
        return await self.app.run(query)


@dataclass(slots=True)
class BatchRunSession:
    """Thin batch wrapper that returns AgenticRunResult."""

    app: AgenticMMRAG

    @classmethod
    def create(
        cls,
        *,
        tools: ToolRegistry | None = None,
        provider: Any,
        **orchestrator_kwargs: Any,
    ) -> "BatchRunSession":
        return cls(
            app=AgenticMMRAG.from_defaults(
                tools=tools,
                provider=provider,
                **orchestrator_kwargs,
            )
        )

    @property
    def orchestrator(self) -> Orchestrator:
        return self.app.orchestrator

    async def run(self, query: QueryContext, *, run_label: str = "default") -> AgenticRunResult:
        result = await self.app.run(query)
        return AgenticRunResult(
            answer=result.answer,
            plan=result.plan,
            results=result.subagent_results,
            fused_evidence=result.fused_evidence,
            warnings=result.warnings,
            stage_metrics={"run_label": run_label, "route": result.route},
        )


def create_runtime() -> AgenticRuntime:
    """Create the tools runtime runtime."""

    return AgenticRuntime()


def create_batch_session(
    *,
    tools: ToolRegistry | None = None,
    provider: Any,
    **orchestrator_kwargs: Any,
) -> BatchRunSession:
    """Create a batch-oriented session over the final orchestrator."""

    return BatchRunSession.create(
        tools=tools,
        provider=provider,
        **orchestrator_kwargs,
    )


def create_orchestrator_session(
    *,
    tools: ToolRegistry | None = None,
    provider: Any,
    **orchestrator_kwargs: Any,
) -> OrchestratorSession:
    """Create the public multi-agent orchestration session."""

    return OrchestratorSession.create(
        tools=tools,
        provider=provider,
        **orchestrator_kwargs,
    )
