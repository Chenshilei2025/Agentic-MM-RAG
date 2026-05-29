"""Batch runner facade backed by the final orchestrator."""

from agentic_mm_rag.results import AgenticRunResult, ReflectionDecision
from agentic_mm_rag.api import BatchRunSession


class BatchRAGRunner:
    """Small helper for batch query execution."""

    def __init__(self, *, tools=None, provider, **orchestrator_kwargs) -> None:
        self.session = BatchRunSession.create(
            tools=tools,
            provider=provider,
            **orchestrator_kwargs,
        )

    async def run(self, query, *, run_label: str = "default") -> AgenticRunResult:
        return await self.session.run(query, run_label=run_label)


__all__ = ["BatchRAGRunner", "AgenticRunResult", "ReflectionDecision"]
