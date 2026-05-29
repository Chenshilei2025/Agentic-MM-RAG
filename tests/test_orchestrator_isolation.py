from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentic_mm_rag.orchestrator.types import QueryContext, SubagentResult
from agentic_mm_rag.orchestrator.loop import Orchestrator
from agentic_mm_rag.providers.base import LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    async def chat(
        self,
        *,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
        response_format=None,
    ) -> LLMResponse:
        if response_format:
            return LLMResponse(content="{}")
        return LLMResponse(content="answer")


@dataclass
class DirectSubagent:
    orchestrator: Orchestrator

    async def run(self, task):
        evidence = {
            "id": f"ev-{task.id}",
            "source_type": task.corpus,
            "modality": "text",
            "source_id": "source",
            "locator": {"doc_id": "source"},
            "content": f"evidence for {task.query}",
            "score": 1.0,
        }
        self.orchestrator.evidence_board.write(
            task_id=task.id,
            agent_name=task.agent,
            tool_used=task.tool_name,
            summary=f"summary for {task.query}",
            evidence=[evidence],
            confidence=0.9,
        )
        return SubagentResult(task=task, ok=True, evidence=[evidence])


class DirectTestOrchestrator(Orchestrator):
    def _spawn_subagent(self, task):
        return DirectSubagent(self)


async def _run_repeated_runs_reset_task_ids_and_do_not_leak_board_state():
    orchestrator = DirectTestOrchestrator(provider=DummyProvider())
    query = QueryContext(query_text="alpha", doc_query_vector=[1.0])

    first = await orchestrator.run(query)
    second = await orchestrator.run(query)

    assert [task.id for task in first.plan.tasks] == ["doc_text-1"]
    assert [task.id for task in second.plan.tasks] == ["doc_text-1"]
    assert first.fused_evidence
    assert second.fused_evidence
    assert all("alpha" in item["content"] for item in second.fused_evidence)


def test_repeated_runs_reset_task_ids_and_do_not_leak_board_state():
    asyncio.run(_run_repeated_runs_reset_task_ids_and_do_not_leak_board_state())


async def _run_evidence_pool_reuse_is_disabled_by_default():
    orchestrator = DirectTestOrchestrator(provider=DummyProvider())

    first = await orchestrator.run(QueryContext(query_text="alpha", doc_query_vector=[1.0]))
    second = await orchestrator.run(QueryContext(query_text="beta", doc_query_vector=[1.0]))

    assert first.fused_evidence
    assert second.fused_evidence
    assert all("beta" in item["content"] for item in second.fused_evidence)
    assert all("alpha" not in item["content"] for item in second.fused_evidence)


def test_evidence_pool_reuse_is_disabled_by_default():
    asyncio.run(_run_evidence_pool_reuse_is_disabled_by_default())


async def _run_evidence_pool_reuse_requires_keyword_overlap_when_enabled():
    orchestrator = DirectTestOrchestrator(provider=DummyProvider(), reuse_evidence_pool=True)

    await orchestrator.run(QueryContext(query_text="alpha", doc_query_vector=[1.0]))
    second = await orchestrator.run(QueryContext(query_text="beta", doc_query_vector=[1.0]))

    assert second.fused_evidence
    assert all("alpha" not in item["content"] for item in second.fused_evidence)


def test_evidence_pool_reuse_requires_keyword_overlap_when_enabled():
    asyncio.run(_run_evidence_pool_reuse_requires_keyword_overlap_when_enabled())
