from __future__ import annotations

import asyncio

from agentic_mm_rag.agent.runner import AgentRunner
from agentic_mm_rag.agent.subagent import DocTextSubagent
from agentic_mm_rag.orchestrator.types import RetrievalTask
from agentic_mm_rag.orchestrator.evidence.board import EvidenceBoard
from agentic_mm_rag.orchestrator.evidence.io import EvidenceBoardWriter
from agentic_mm_rag.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agentic_mm_rag.schemas import EvidenceCard, Locator, ScoreParts, ToolResponse
from agentic_mm_rag.tools.base import FunctionTool
from agentic_mm_rag.tools.registry import ToolRegistry
from agentic_mm_rag.tools.schema import ArraySchema, NumberSchema, StringSchema, tool_parameters_schema
from agentic_mm_rag.agent.subagent import SeekBudgetRegistry


class SeekOnlyProvider(LLMProvider):
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
        tool_names = {
            item["function"]["name"]
            for item in tools or []
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        assert "write_evidence" not in tool_names
        if self.calls == 1:
            return LLMResponse(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="seek-1",
                        name="doc_text_seek",
                        arguments={"query_vector": [1.0], "top_k": 2},
                    )
                ],
            )
        return LLMResponse(
            content='{"summary":"kept the direct alpha evidence","kept_evidence_ids":["ev-keep"],"gaps":[]}',
            finish_reason="stop",
        )


class NoToolProvider(LLMProvider):
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
        return LLMResponse(content="no tool call")


async def fake_doc_text_seek(query_vector, top_k=2):
    return ToolResponse(
        ok=True,
        tool="doc_text_seek",
        evidence=[
            EvidenceCard(
                id="ev-drop",
                source_type="doc",
                modality="text",
                source_id="doc",
                locator=Locator(doc_id="doc"),
                content="background material unrelated to the alpha answer",
                score=0.9,
                score_parts=ScoreParts(text=0.9),
            ),
            EvidenceCard(
                id="ev-keep",
                source_type="doc",
                modality="text",
                source_id="doc",
                locator=Locator(doc_id="doc"),
                content="alpha is the directly supported answer",
                score=0.5,
                score_parts=ScoreParts(text=0.5),
            ),
        ],
    )


async def _run_runtime_enforces_write_evidence_after_model_selection():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "doc_text_seek",
            "Fake text seek.",
            fake_doc_text_seek,
            tool_parameters_schema(
                query_vector=ArraySchema(NumberSchema("dimension"), min_items=1),
                top_k=NumberSchema("k"),
                required=["query_vector"],
            ),
        )
    )
    board = EvidenceBoard()
    provider = SeekOnlyProvider()
    subagent = DocTextSubagent(
        registry,
        EvidenceBoardWriter(board),
        runner=AgentRunner(provider),
        model="dummy",
    )
    task = RetrievalTask(
        id="doc_text-1",
        agent="doc_text_subagent",
        corpus="doc",
        tool_name="doc_text_seek",
        intent="doc_text_seek",
        params={"query_vector": [1.0]},
        rationale="test",
        query="alpha",
        allowed_tools=["doc_text_seek", "write_evidence"],
    )

    result = await subagent.run(task)

    assert result.ok
    assert result.data["write_mode"] == "runtime_enforced"
    assert [item["id"] for item in result.evidence] == ["ev-keep"]
    reports = board.reports()
    assert len(reports) == 1
    assert reports[0].metadata["model_kept_evidence_ids"] == ["ev-keep"]


def test_runtime_enforces_write_evidence_after_model_selection():
    asyncio.run(_run_runtime_enforces_write_evidence_after_model_selection())


async def _run_runtime_fallback_seeks_when_model_calls_no_tools():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "doc_text_seek",
            "Fake text seek.",
            fake_doc_text_seek,
            tool_parameters_schema(
                query_vector=ArraySchema(NumberSchema("dimension"), min_items=1),
                top_k=NumberSchema("k"),
                required=["query_vector"],
            ),
        )
    )
    board = EvidenceBoard()
    subagent = DocTextSubagent(
        registry,
        EvidenceBoardWriter(board),
        runner=AgentRunner(NoToolProvider()),
        model="dummy",
    )
    task = RetrievalTask(
        id="doc_text-1",
        agent="doc_text_subagent",
        corpus="doc",
        tool_name="doc_text_seek",
        intent="doc_text_seek",
        params={"query_vector": [1.0]},
        rationale="test",
        query="alpha",
        allowed_tools=["doc_text_seek", "write_evidence"],
    )

    result = await subagent.run(task)

    assert result.ok
    assert result.evidence
    assert result.data["reports"][0]["metadata"]["fallback_seek_used"] is True
    assert any("runtime fallback seek executed" in warning for warning in result.warnings)


def test_runtime_fallback_seeks_when_model_calls_no_tools():
    asyncio.run(_run_runtime_fallback_seeks_when_model_calls_no_tools())


async def _run_seek_budget_registry_keeps_required_params():
    registry = ToolRegistry()

    async def capture_tool(query_vector, video_root=None, top_k=1):
        return ToolResponse(
            ok=True,
            tool="video_text_seek",
            data={
                "query_vector": query_vector,
                "video_root": video_root,
                "top_k": top_k,
            },
        )

    registry.register(
        FunctionTool(
            "video_text_seek",
            "Capture params.",
            capture_tool,
            tool_parameters_schema(
                query_vector=ArraySchema(NumberSchema("dimension"), min_items=1),
                video_root=StringSchema("video root", nullable=True),
                top_k=NumberSchema("k"),
                required=["query_vector"],
            ),
        )
    )
    wrapped = SeekBudgetRegistry(
        registry,
        seek_tool="video_text_seek",
        seek_call_budget=1,
        required_params={"query_vector": [9.0], "video_root": "/tmp/video"},
    )

    response = await wrapped.execute(
        "video_text_seek",
        {"query_vector": [1.0], "video_root": "/tmp/override", "top_k": 7},
    )

    assert response.ok
    assert response.data["query_vector"] == [9.0]
    assert response.data["video_root"] == "/tmp/video"
    assert response.data["top_k"] == 7


def test_seek_budget_registry_keeps_required_params():
    asyncio.run(_run_seek_budget_registry_keeps_required_params())
