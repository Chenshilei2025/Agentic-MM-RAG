"""Evidence board tool adapters."""

from __future__ import annotations

from typing import Any

from agentic_mm_rag.orchestrator.evidence_board import EvidenceBoard
from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.base import Tool, tool_metadata
from agentic_mm_rag.tools.schema import (
    ArraySchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

class EvidenceBoardWriter:
    """Tool-like adapter for subagents to report evidence to the decision agent."""

    def __init__(self, board: EvidenceBoard) -> None:
        self.board = board

    async def execute(
        self,
        *,
        task_id: str,
        agent_name: str,
        tool_used: str,
        summary: str,
        evidence: list[dict[str, Any]] | None = None,
        confidence: float = 0.0,
        gaps: list[str] | None = None,
        filtering_notes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResponse:
        report = self.board.write(
            task_id=task_id,
            agent_name=agent_name,
            tool_used=tool_used,
            summary=summary,
            evidence=evidence,
            confidence=confidence,
            gaps=gaps,
            filtering_notes=filtering_notes,
            metadata=metadata,
        )
        return ToolResponse(
            ok=True,
            tool="write_evidence",
            data={
                "task_id": report.task_id,
                "agent_name": report.agent_name,
                "tool_used": report.tool_used,
                "evidence_count": len(report.evidence),
                "confidence": report.confidence,
                "gaps": report.gaps,
                "filtering_notes": report.filtering_notes,
            },
        )


class ReadEvidenceTool(Tool):
    """Model-visible tool for reading the shared evidence board state."""

    def __init__(self, board: EvidenceBoard) -> None:
        self.board = board

    @property
    def name(self) -> str:
        return "read_evidence"

    @property
    def description(self) -> str:
        return "Read the shared evidence board state for planning, reflection, or final answering."

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema()

    @property
    def read_only(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return tool_metadata(
            category="orchestration",
            corpus="shared",
            role="read_evidence",
            stability="stable",
            recommended_usage="Decision agent only: read current reports, facts, gaps, conflicts, sources, and coverage.",
            tags=["shared", "orchestration", "public"],
        )

    async def execute(self, **kwargs: Any) -> ToolResponse:
        return ToolResponse(ok=True, tool="read_evidence", data=self.board.state_snapshot())


class WriteEvidenceTool(Tool):
    """Model-visible tool for writing structured evidence reports."""

    def __init__(self, writer: EvidenceBoardWriter) -> None:
        self.writer = writer

    @property
    def name(self) -> str:
        return "write_evidence"

    @property
    def description(self) -> str:
        return (
            "Write a structured expert evidence report to the shared evidence board. "
            "Every subagent must call this before completing its task."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            task_id=StringSchema("Assigned task id.", min_length=1),
            agent_name=StringSchema(
                "Reporting subagent name.",
                enum=[
                    "doc_text_subagent",
                    "doc_visual_subagent",
                    "doc_graph_subagent",
                    "video_text_subagent",
                    "video_visual_subagent",
                    "video_graph_subagent",
                ],
            ),
            tool_used=StringSchema(
                "Seek tool used by this subagent.",
                enum=[
                    "doc_text_seek",
                    "doc_visual_seek",
                    "doc_graph_seek",
                    "video_text_seek",
                    "video_visual_seek",
                    "video_graph_seek",
                ],
            ),
            summary=StringSchema("Short factual summary of the evidence report.", min_length=1),
            evidence=ArraySchema(
                ObjectSchema(additional_properties=True),
                description="Serialized evidence card dictionaries.",
            ),
            confidence=NumberSchema(
                description="Confidence from 0.0 to 1.0.",
                minimum=0.0,
                maximum=1.0,
            ),
            gaps=ArraySchema(
                StringSchema("Remaining evidence gap."),
                description="Known missing evidence or uncertainty.",
            ),
            filtering_notes=ArraySchema(
                StringSchema("Filtering note."),
                description="Notes about rejected weak, noisy, or unrelated evidence.",
            ),
            metadata=ObjectSchema(
                additional_properties=True,
                description="Optional execution metadata.",
            ),
            required=["task_id", "agent_name", "tool_used", "summary"],
        )

    @property
    def read_only(self) -> bool:
        return False

    @property
    def metadata(self) -> dict[str, Any]:
        return tool_metadata(
            category="orchestration",
            corpus="shared",
            role="report",
            stability="stable",
            recommended_usage="Subagent protocol tool for reporting intermediate evidence to the shared board.",
            tags=["shared", "orchestration", "public", "report"],
        )

    async def execute(self, **kwargs: Any) -> ToolResponse:
        return await self.writer.execute(**kwargs)
