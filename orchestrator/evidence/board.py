"""Enhanced evidence board with iterative state tracking and fact consolidation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentic_mm_rag.config import DEFAULT_MODELS
from agentic_mm_rag.orchestrator.types import SubagentResult
from agentic_mm_rag.agent.prompts import build_evidence_board_prompt
from agentic_mm_rag.providers.base import LLMProvider

@dataclass
class AtomicFact:
    """A single atomic piece of information verified from evidence."""
    id: str
    text: str
    source_id: str
    modality: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidenceGap:
    """Missing or ambiguous information identified by the board."""
    description: str
    urgency: float  # 0.0 to 1.0
    suggested_intent: str | None = None

@dataclass(slots=True)
class EvidenceReport:
    """Structured report written by expert subagents."""

    task_id: str
    agent_name: str
    tool_used: str
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    gaps: list[str] = field(default_factory=list)
    filtering_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceBoard:
    """In-memory report board shared by orchestrator and subagents."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        enable_llm_consolidation: bool = False,
    ) -> None:
        if enable_llm_consolidation and provider is None:
            raise ValueError("EvidenceBoard LLM consolidation requires an injected LLMProvider.")
        self._reports: list[EvidenceReport] = []
        self.facts: list[AtomicFact] = []
        self.gaps: list[EvidenceGap] = []
        self.conflicts: list[dict[str, Any]] = []
        self.round = 1
        self.provider = provider
        self.enable_llm_consolidation = enable_llm_consolidation
        self.model = DEFAULT_MODELS.decision

    def write(
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
    ) -> EvidenceReport:
        report = EvidenceReport(
            task_id=task_id,
            agent_name=agent_name,
            tool_used=tool_used,
            summary=summary,
            evidence=list(evidence or []),
            confidence=max(0.0, min(1.0, float(confidence))),
            gaps=list(gaps or []),
            filtering_notes=list(filtering_notes or []),
            metadata=dict(metadata or {}),
        )
        self._reports.append(report)
        return report

    def reports(self) -> list[EvidenceReport]:
        return list(self._reports)

    def evidence_items(self) -> list[dict[str, Any]]:
        return [item for report in self._reports for item in report.evidence]

    def clear(self) -> None:
        self._reports.clear()
        self.facts.clear()
        self.gaps.clear()
        self.conflicts.clear()
        self.round = 1

    async def ingest_results(self, query: str, results: list[SubagentResult]) -> None:
        """Consolidate subagent results into atomic facts and identify gaps."""
        new_items = []
        for res in results:
            if not res.ok:
                continue
            new_items.extend(res.evidence)
            items = res.data.get("items")
            if isinstance(items, list):
                new_items.extend(items)

        if not new_items:
            return

        self._ingest_heuristic(results, new_items)
        if not self.enable_llm_consolidation:
            return
        if self.provider is None:
            return

        items_summary = []
        for item in new_items[:8]:  # Limit context window
            content = str(item.get("content", ""))[:200]
            m = item.get("modality", "unknown")
            items_summary.append(f"[{m}] {content}")

        current_facts = "\n".join([f"- {f.text}" for f in self.facts])
        
        prompt = build_evidence_board_prompt(
            query=query,
            current_facts=current_facts,
            new_evidence=chr(10).join(items_summary),
        )
        try:
            resp = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.content)
            for f in data.get("new_facts", []):
                self.facts.append(AtomicFact(
                    id=f"fact-{len(self.facts)}",
                    text=f["text"],
                    source_id="multi",
                    modality="text/visual",
                    confidence=f.get("confidence", 1.0)
                ))
            self.gaps = [EvidenceGap(**g) for g in data.get("identified_gaps", [])]
        except Exception:
            pass

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "consolidation": {
                "enabled": self.enable_llm_consolidation,
                "model": self.model,
            },
            "facts": [
                {
                    "id": fact.id,
                    "text": fact.text,
                    "source_id": fact.source_id,
                    "modality": fact.modality,
                    "confidence": fact.confidence,
                    "metadata": dict(fact.metadata),
                }
                for fact in self.facts
            ],
            "gaps": [
                {
                    "description": gap.description,
                    "urgency": gap.urgency,
                    "suggested_intent": gap.suggested_intent,
                }
                for gap in self.gaps
            ],
            "conflicts": list(self.conflicts),
            "reports": [
                {
                    "task_id": report.task_id,
                    "agent_name": report.agent_name,
                    "tool_used": report.tool_used,
                    "summary": report.summary,
                    "confidence": report.confidence,
                    "evidence": list(report.evidence),
                    "evidence_count": len(report.evidence),
                    "gaps": list(report.gaps),
                    "filtering_notes": list(report.filtering_notes),
                    "metadata": dict(report.metadata),
                }
                for report in self._reports
            ],
            "candidate_sources": self.evidence_items(),
            "coverage": self.coverage(),
        }

    def coverage(self) -> dict[str, bool]:
        coverage = {
            "doc_text": False,
            "doc_visual": False,
            "doc_graph": False,
            "video_text": False,
            "video_visual": False,
            "video_graph": False,
        }
        for report in self._reports:
            if report.tool_used.endswith("_seek"):
                key = report.tool_used.removesuffix("_seek")
                if key in coverage:
                    coverage[key] = True
        return coverage

    def _ingest_heuristic(
        self,
        results: list[SubagentResult],
        evidence_items: list[dict[str, Any]],
    ) -> None:
        known_fact_keys = {(fact.source_id, fact.text) for fact in self.facts}
        for item in evidence_items:
            content = " ".join(str(item.get("content") or "").split())
            if not content:
                continue
            source_id = str(item.get("id") or item.get("source_id") or "unknown")
            text = content[:240]
            key = (source_id, text)
            if key in known_fact_keys:
                continue
            score = item.get("score")
            confidence = float(score) if isinstance(score, int | float) else 0.6
            self.facts.append(
                AtomicFact(
                    id=f"fact-{len(self.facts)}",
                    text=text,
                    source_id=source_id,
                    modality=str(item.get("modality") or item.get("source_type") or "unknown"),
                    confidence=max(0.0, min(1.0, confidence)),
                    metadata={
                        "source_type": item.get("source_type"),
                        "locator": item.get("locator", {}),
                    },
                )
            )
            known_fact_keys.add(key)

        gap_records: list[EvidenceGap] = []
        for result in results:
            for gap in result.warnings:
                gap_records.append(
                    EvidenceGap(
                        description=str(gap),
                        urgency=0.7 if "no evidence" in str(gap).lower() else 0.55,
                        suggested_intent="doc_text_seek",
                    )
                )
            items = result.data.get("items") if isinstance(result.data, dict) else None
            if result.ok and not result.evidence and not items:
                gap_records.append(
                    EvidenceGap(
                        description=f"{result.task.id} returned no evidence for {result.task.corpus}",
                        urgency=0.7,
                        suggested_intent="doc_text_seek"
                        if result.task.intent in {"doc_graph_seek", "video_graph_seek"}
                        else result.task.intent,
                    )
                )
        if gap_records:
            existing = {(gap.description, gap.suggested_intent) for gap in self.gaps}
            for gap in gap_records:
                key = (gap.description, gap.suggested_intent)
                if key not in existing:
                    self.gaps.append(gap)
                    existing.add(key)

    def get_consolidated_report(self) -> str:
        """Returns a formatted report of all discovered facts."""
        if not self.facts:
            return "No verified facts discovered yet."
        return "\n".join([f"• {f.text}" for f in self.facts])
