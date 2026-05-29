from __future__ import annotations

import asyncio

from agentic_mm_rag.agent.decision import DecisionAgent
from agentic_mm_rag.orchestrator.types import AgentPlan, QueryContext
from agentic_mm_rag.providers.base import LLMProvider, LLMResponse


class SpyProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        return LLMResponse(content='{"answer":"unsupported"}')


async def _run_generation_refuses_weak_evidence_without_llm_call():
    provider = SpyProvider()
    agent = DecisionAgent(provider=provider)
    plan = AgentPlan(
        query_context=QueryContext(query_text="What is the alpha answer?"),
        tasks=[],
        rationale="test",
    )
    weak_evidence = [
        {
            "id": "weak-1",
            "source_type": "video",
            "modality": "video_segment",
            "source_id": "source",
            "content": "alpha background context only",
            "score": 0.8,
            "score_parts": {"text": 0.8},
        }
    ]

    answer = await agent.generate_async(plan, [], weak_evidence)

    assert "No direct answer support" in answer
    assert provider.calls == 0


def test_generation_refuses_weak_evidence_without_llm_call():
    asyncio.run(_run_generation_refuses_weak_evidence_without_llm_call())
