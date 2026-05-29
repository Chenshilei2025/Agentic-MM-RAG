"""Execution subagents for the final seek/write evidence protocol."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from agentic_mm_rag.agent.contracts import allowed_tools_for_contract
from agentic_mm_rag.config import DEFAULT_MODELS
from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.orchestrator.evidence.io import WriteEvidenceTool, EvidenceBoardWriter
from agentic_mm_rag.agent.prompts import (
    DOC_GRAPH_SUBAGENT_ROLE,
    DOC_TEXT_SUBAGENT_ROLE,
    DOC_VISUAL_SUBAGENT_ROLE,
    VISUAL_INSPECTION_SYSTEM_PROMPT,
    VISUAL_INSPECTION_USER_TEXT,
    VIDEO_GRAPH_SUBAGENT_ROLE,
    VIDEO_TEXT_SUBAGENT_ROLE,
    VIDEO_VISUAL_SUBAGENT_ROLE,
    build_subagent_system_prompt,
)
from agentic_mm_rag.orchestrator.types import RetrievalTask, SubagentResult
from agentic_mm_rag.agent.runner import AgentRunSpec, AgentRunner
from agentic_mm_rag.tools.registry import ToolRegistry

# Lazy import to avoid circular dependency with orchestrator.evidence.quality
# which may import from this module indirectly through the agent package.
_guardrail_evidence_batch = None


def _get_guardrail():
    global _guardrail_evidence_batch
    if _guardrail_evidence_batch is None:
        from agentic_mm_rag.orchestrator.evidence.quality import guardrail_evidence_batch
        _guardrail_evidence_batch = guardrail_evidence_batch
    return _guardrail_evidence_batch


@dataclass(slots=True)
class SubagentSession:
    """One concurrent expert session over a shared model/tool runtime."""

    task: RetrievalTask
    role_prompt: str
    allowed_tools: list[str]


class SeekBudgetRegistry(ToolRegistry):
    """Per-session registry that enforces the assigned seek-call budget.

    Automatically injects ``query_vector`` and corpus root paths from the
    task's ``required_seek_params`` into every seek tool call so the LLM
    does not need to reproduce the embedding vector in its tool-call
    arguments.
    """

    def __init__(
        self,
        source: ToolRegistry,
        *,
        seek_tool: str,
        seek_call_budget: int,
        required_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.seek_tool = seek_tool
        self.seek_call_budget = max(0, int(seek_call_budget))
        self.seek_calls = 0
        self._required_params = dict(required_params or {})
        for name in source.tool_names:
            tool = source.get(name)
            if tool is not None:
                self.register(tool)

    async def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResponse:
        if name == self.seek_tool:
            if self.seek_calls >= self.seek_call_budget:
                return ToolResponse(
                    ok=False,
                    tool=name,
                    error=(
                        f"seek_call_budget exceeded for {name}: "
                        f"{self.seek_calls}/{self.seek_call_budget}"
                    ),
                )
            self.seek_calls += 1
            params = self._inject_required_params(params)
        return await super().execute(name, params, **kwargs)

    def _inject_required_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(params or {})
        merged.update(self._required_params)
        return merged


def _visual_asset_path(item: dict[str, Any]) -> str | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    trace = metadata.get("visual_trace") if isinstance(metadata.get("visual_trace"), dict) else {}
    for value in (
        trace.get("asset_path"),
        metadata.get("visual_asset_path"),
        provenance.get("asset_path"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _image_url_content(path: str) -> dict[str, Any] | None:
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        return None
    media_type = guess_type(str(image_path))[0] or "image/jpeg"
    if not media_type.startswith("image/"):
        media_type = "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{payload}"},
    }


def _doc_id_from_evidence(item: dict[str, Any]) -> str | None:
    locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    doc_id = (
        locator.get("doc_id")
        or metadata.get("doc_id")
        or provenance.get("doc_id")
        or item.get("doc_id")
    )
    return str(doc_id) if doc_id else None


def _task_answer_target(task: RetrievalTask) -> dict[str, Any]:
    target = task.params.get("answer_target")
    if isinstance(target, dict):
        return dict(target)
    answer_format = str(task.params.get("answer_format") or "").strip()
    return {
        "question": task.query,
        "answer_format": answer_format or None,
        "direct_answer_first": True,
    }


def _task_evidence_contract(task: RetrievalTask) -> dict[str, Any]:
    contract = task.params.get("evidence_contract")
    if isinstance(contract, dict):
        return dict(contract)
    answer_format = str(task.params.get("answer_format") or "").strip()
    return {
        "must_use_source_image": task.agent == "doc_visual_subagent",
        "must_extract_visible_fields": task.agent == "doc_visual_subagent",
        "prefer_exact_labels": True,
        "prefer_calculable_fields": answer_format in {"Int", "Float"},
        "allow_ocr_only": task.agent != "doc_visual_subagent",
    }


def _filter_task_doc_evidence(
    evidence: list[dict[str, Any]],
    task: RetrievalTask,
) -> tuple[list[dict[str, Any]], int]:
    doc_ids = task.params.get("doc_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return evidence, 0
    allowed = {str(doc_id) for doc_id in doc_ids if str(doc_id).strip()}
    if not allowed:
        return evidence, 0
    filtered: list[dict[str, Any]] = []
    dropped = 0
    for item in evidence:
        source_type = item.get("source_type")
        if source_type not in {None, "doc"}:
            filtered.append(item)
            continue
        if _doc_id_from_evidence(item) in allowed:
            filtered.append(item)
        else:
            dropped += 1
    return filtered, dropped


def _task_rewritten_data(task: RetrievalTask) -> dict[str, Any]:
    rewritten = task.params.get("rewritten_data")
    return dict(rewritten) if isinstance(rewritten, dict) else {}


def _task_retrieval_hints(task: RetrievalTask) -> dict[str, Any] | None:
    hints = task.params.get("retrieval_hints")
    if isinstance(hints, dict):
        return dict(hints)
    rewritten = task.params.get("rewritten_data")
    if isinstance(rewritten, dict) and isinstance(rewritten.get("retrieval_hints"), dict):
        return dict(rewritten["retrieval_hints"])
    return None


def _seek_budget_for_task(task: RetrievalTask) -> dict[str, Any]:
    params = task.params
    if task.tool_name.endswith("_graph_seek"):
        return {
            "seek_call_budget": 2,
            "top_k_entities": {
                "suggested": params.get("top_k_entities"),
                "min": 4,
                "max": max(8, int(params.get("top_k_entities") or 10) * 2),
            },
            "top_k_chunks": {
                "suggested": params.get("top_k_chunks") or params.get("top_k"),
                "min": 4,
                "max": max(8, int(params.get("top_k_chunks") or params.get("top_k") or 10) * 2),
            },
        }
    suggested_top_k = int(params.get("top_k") or 10)
    return {
        "seek_call_budget": 2,
        "top_k": {
            "suggested": suggested_top_k,
            "min": 3,
            "max": max(8, suggested_top_k * 2),
        },
    }


def _seek_required_params(task: RetrievalTask) -> dict[str, Any]:
    required_keys = {
        "query_vector",
        "doc_root",
        "video_root",
        "doc_ids",
        "evidence_pages",
        "page_bias_pages",
        "visual_block_ids",
        "segment_ids",
    }
    return {
        key: value
        for key, value in task.params.items()
        if key in required_keys and value is not None
    }


def _seek_required_params_hint(task: RetrievalTask) -> dict[str, Any]:
    """Lightweight version of required params for the LLM message.

    Replaces the query_vector array with a dimension hint so the
    message stays compact. The actual vector is injected at execution
    time by ``SeekBudgetRegistry``.
    """
    params = _seek_required_params(task)
    if "query_vector" in params and isinstance(params["query_vector"], list):
        params = dict(params)
        params["query_vector"] = f"<{len(params['query_vector'])}-dim vector, auto-injected>"
    return params


def _seek_suggested_params(task: RetrievalTask) -> dict[str, Any]:
    planning_keys = {
        "query_text",
        "rewritten_data",
        "modalities",
        "min_score",
        "include_multimodal",
        "graph_strategy",
        "edge_type_filter",
        "include_mapped_segment_details",
        "exact_detail_lexical",
    }
    return {
        key: value
        for key, value in task.params.items()
        if key in planning_keys and value is not None
    }


def _evidence_items_from_messages(messages: list[dict[str, Any]], seek_tool: str) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != seek_tool:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            warnings.append(f"{seek_tool} returned non-json tool content")
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ok") is False and payload.get("error"):
            warnings.append(str(payload["error"]))
        raw_evidence = payload.get("evidence")
        if not isinstance(raw_evidence, list):
            continue
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "")
            dedupe_key = item_id or json.dumps(item, sort_keys=True, ensure_ascii=False)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            evidence.append(item)
    return evidence, warnings


def _kept_ids_from_final_content(content: str | None) -> list[str]:
    if not content:
        return []
    text = content.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        value = payload.get("kept_evidence_ids")
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    match = re.search(r"kept_evidence_ids\s*[:=]\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
    if match:
        return [
            item.strip().strip("'\"")
            for item in match.group(1).split(",")
            if item.strip().strip("'\"")
        ]
    match = re.search(r"kept_evidence_ids\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if not match:
        return []
    return [
        item.strip().strip("'\"")
        for item in re.split(r"[,;\s]+", match.group(1))
        if item.strip().strip("'\"")
    ]


def _apply_model_selection(
    evidence: list[dict[str, Any]],
    final_content: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    kept_ids = _kept_ids_from_final_content(final_content)
    if not kept_ids:
        return evidence, []
    by_id = {str(item.get("id") or ""): item for item in evidence if isinstance(item, dict)}
    selected = [by_id[item_id] for item_id in kept_ids if item_id in by_id]
    return (selected or evidence), kept_ids


class ExpertSubagent:
    """Base class for concrete expert subagents."""

    agent_name = "doc_text_subagent"
    seek_tool = "doc_text_seek"
    role_prompt = ""
    default_model = DEFAULT_MODELS.text_expert

    def __init__(
        self,
        tools: ToolRegistry,
        evidence_writer: EvidenceBoardWriter,
        *,
        runner: AgentRunner,
        model: str | None = None,
        max_iterations: int = 4,
    ) -> None:
        self.tools = tools
        self.evidence_writer = evidence_writer
        self.runner = runner
        self.provider = runner.provider
        self.model = model or self.default_model
        self.max_iterations = max_iterations

    def build_session(self, task: RetrievalTask) -> SubagentSession:
        allowed = task.allowed_tools or allowed_tools_for_contract(self.agent_name)
        return SubagentSession(task=task, role_prompt=self.role_prompt, allowed_tools=allowed)

    async def run(self, task: RetrievalTask) -> SubagentResult:
        return await self._run_model_session(task)

    async def _run_model_session(self, task: RetrievalTask) -> SubagentResult:
        session = self.build_session(task)
        session_tools = self._session_tool_registry(session)
        task.status = "running"
        run = await self.runner.run(
            AgentRunSpec(
                initial_messages=self._session_messages(task, session),
                tools=session_tools,
                model=self.model,
                max_iterations=self.max_iterations,
                concurrent_tools=True,
                session_key=task.id,
            )
        )
        task.status = "done" if run.error is None else "error"
        runtime_warnings: list[str] = []
        reports = [
            report
            for report in self.evidence_writer.board.reports()
            if report.task_id == task.id and report.agent_name == self.agent_name
        ]
        runtime_write_used = False
        fallback_seek_used = False
        if not reports:
            evidence, runtime_warnings = _evidence_items_from_messages(run.messages, self.seek_tool)
            if not evidence and run.error is None:
                fallback_response = await session_tools.execute(
                    self.seek_tool,
                    _seek_required_params(task),
                )
                if fallback_response.ok and fallback_response.evidence:
                    evidence = [item.to_dict() for item in fallback_response.evidence]
                    runtime_warnings.append(
                        "runtime fallback seek executed because model produced no seek evidence"
                    )
                    fallback_seek_used = True
                elif not fallback_response.ok:
                    runtime_warnings.append(fallback_response.error or "runtime fallback seek failed")
            if evidence:
                selected_evidence, model_kept_ids = _apply_model_selection(evidence, run.final_content)
                reviewed, diagnostics = _get_guardrail()(
                    selected_evidence,
                    question=task.query or str(task.params.get("query_text") or ""),
                    rewritten=_task_rewritten_data(task),
                    retrieval_hints=_task_retrieval_hints(task),
                    min_keep=1,
                )
                runtime_write_used = True
                await self.evidence_writer.execute(
                    task_id=task.id,
                    agent_name=self.agent_name,
                    tool_used=self.seek_tool,
                    summary=self._summarize_report(
                        task,
                        ok=run.error is None,
                        evidence_count=len(reviewed),
                        error=run.error,
                    ),
                    evidence=reviewed,
                    confidence=self._confidence(run.error is None, len(reviewed)),
                    gaps=runtime_warnings,
                    filtering_notes=[
                        (
                            "runtime-enforced write_evidence from seek results; "
                            f"kept {len(reviewed)}/{len(evidence)} item(s)"
                        )
                    ],
                    metadata={
                        "write_mode": "runtime_enforced",
                        "raw_count": len(evidence),
                        "kept_count": len(reviewed),
                        "rejected_count": max(0, len(evidence) - len(reviewed)),
                        "model_kept_evidence_ids": model_kept_ids,
                        "model_selection_count": len(selected_evidence),
                        "quality_report": diagnostics,
                        "model_final_content": run.final_content,
                        "model_tools_used": run.tools_used,
                        "fallback_seek_used": fallback_seek_used,
                    },
                )
                reports = [
                    report
                    for report in self.evidence_writer.board.reports()
                    if report.task_id == task.id and report.agent_name == self.agent_name
                ]
        quality_report = self._review_local_reports(task, reports)
        evidence = [item for report in reports for item in report.evidence]
        gaps = [gap for report in reports for gap in report.gaps]
        protocol_error = None
        if run.error is None and not reports:
            protocol_error = "model session produced no seek evidence for runtime write_evidence"
            task.status = "error"
        return SubagentResult(
            task=task,
            ok=run.error is None and protocol_error is None,
            evidence=evidence,
            data={
                "final_content": run.final_content,
                "tools_used": run.tools_used,
                "seek_call_count": getattr(session_tools, "seek_calls", None),
                "seek_call_budget": getattr(session_tools, "seek_call_budget", None),
                "fallback_seek_used": fallback_seek_used,
                "write_mode": "runtime_enforced" if runtime_write_used else "model_tool_call",
                "usage": run.usage,
                "stop_reason": run.stop_reason,
                "raw_evidence_count": quality_report.get("raw_count", len(evidence)),
                "quality_report": quality_report,
                "reports": [
                    {
                        "summary": report.summary,
                        "confidence": report.confidence,
                        "gaps": report.gaps,
                        "filtering_notes": report.filtering_notes,
                        "metadata": report.metadata,
                    }
                    for report in reports
                ],
            },
            warnings=gaps + runtime_warnings + ([protocol_error] if protocol_error else []),
            error=run.error or protocol_error,
        )

    def _review_local_reports(self, task: RetrievalTask, reports: list[Any]) -> dict[str, Any]:
        raw_items = [item for report in reports for item in report.evidence if isinstance(item, dict)]
        if not raw_items:
            return {
                "reviewer": "subagent_deterministic_local_review",
                "review_mode": "llm_order_guardrail",
                "raw_count": 0,
                "kept_count": 0,
                "dropped_count": 0,
                "dropped_evidence_ids": [],
                "off_topic_evidence_ids": [],
            }
        kept, diagnostics = _get_guardrail()(
            raw_items,
            question=task.query or str(task.params.get("query_text") or ""),
            rewritten=_task_rewritten_data(task),
            retrieval_hints=_task_retrieval_hints(task),
            min_keep=1,
        )
        quality_by_id = {
            str(item.get("id") or index): item
            for index, item in enumerate(kept)
            if isinstance(item, dict)
        }
        for report in reports:
            original_evidence = [item for item in report.evidence if isinstance(item, dict)]
            original_keys = [
                str(item.get("id") or index)
                for index, item in enumerate(original_evidence)
            ]
            reviewed_evidence = [
                quality_by_id[item_key]
                for item_key in original_keys
                if item_key in quality_by_id
            ]
            report.evidence = reviewed_evidence
            note = (
                "local guardrail preserved LLM order and kept "
                f"{len(reviewed_evidence)}/{len(original_evidence)} item(s)"
            )
            if note not in report.filtering_notes:
                report.filtering_notes.append(note)
            metadata = dict(report.metadata)
            metadata["raw_evidence_count"] = len(original_evidence)
            metadata["quality_report"] = diagnostics
            metadata["llm_rerank_order"] = [item.get("id") for item in reviewed_evidence]
            metadata["guardrail_scores"] = {
                str(item.get("id") or index): float((item.get("quality") or {}).get("relevance", 0.0))
                for index, item in enumerate(reviewed_evidence)
            }
            report.metadata = metadata
        return {
            "reviewer": "subagent_deterministic_local_review",
            "review_mode": diagnostics.get("review_mode", "llm_order_guardrail"),
            "raw_count": len(raw_items),
            "kept_count": len(kept),
            "dropped_count": max(0, len(raw_items) - len(kept)),
            "kept_evidence_ids": [item.get("id") for item in kept],
            "ranked_evidence_ids": diagnostics.get("ranked_evidence_ids", []),
            "ranked_scores": diagnostics.get("ranked_scores", {}),
            "dropped_evidence_ids": diagnostics.get("dropped_evidence_ids", []),
            "off_topic_evidence_ids": diagnostics.get("off_topic_evidence_ids", []),
            "covered_tokens": diagnostics.get("covered_tokens", []),
            "required_tokens": diagnostics.get("required_tokens", []),
        }

    def _reports_for_task(self, task: RetrievalTask) -> list[Any]:
        return [
            report
            for report in self.evidence_writer.board.reports()
            if report.task_id == task.id and report.agent_name == self.agent_name
        ]

    def _replace_reports_with_inspected_evidence(
        self,
        task: RetrievalTask,
        *,
        evidence: list[dict[str, Any]],
        inspection_metadata: dict[str, Any],
        filtering_note: str,
    ) -> dict[str, Any]:
        reports = self._reports_for_task(task)
        if not reports:
            return {}
        for report in reports:
            report.evidence = list(evidence)
            if filtering_note not in report.filtering_notes:
                report.filtering_notes.append(filtering_note)
            metadata = dict(report.metadata)
            metadata.update(inspection_metadata)
            report.metadata = metadata
        return self._review_local_reports(task, reports)

    def _sync_result_from_reports(
        self,
        task: RetrievalTask,
        result: SubagentResult,
        quality_report: dict[str, Any],
    ) -> None:
        reports = self._reports_for_task(task)
        result.evidence = [item for report in reports for item in report.evidence]
        result.data["raw_evidence_count"] = quality_report.get("raw_count", len(result.evidence))
        result.data["quality_report"] = quality_report
        result.data["reports"] = [
            {
                "summary": report.summary,
                "confidence": report.confidence,
                "gaps": report.gaps,
                "filtering_notes": report.filtering_notes,
                "metadata": report.metadata,
            }
            for report in reports
        ]

    def _session_tool_registry(self, session: SubagentSession) -> ToolRegistry:
        registry = ToolRegistry()
        for name in session.allowed_tools:
            if name == "write_evidence":
                continue
            tool = self.tools.get(name)
            if tool is not None:
                registry.register(tool)
        budget = _seek_budget_for_task(session.task).get("seek_call_budget", 1)
        return SeekBudgetRegistry(
            registry,
            seek_tool=self.seek_tool,
            seek_call_budget=int(budget or 1),
            required_params=_seek_required_params(session.task),
        )

    def _session_messages(
        self,
        task: RetrievalTask,
        session: SubagentSession,
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": build_subagent_system_prompt(
                    session.role_prompt,
                    answer_target=_task_answer_target(task),
                    evidence_contract=_task_evidence_contract(task),
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": task.id,
                        "agent_name": self.agent_name,
                        "tool_used": self.seek_tool,
                        "query": task.query,
                        "required_seek_params": _seek_required_params_hint(task),
                        "suggested_seek_params": _seek_suggested_params(task),
                        "retrieval_budget": _seek_budget_for_task(task),
                        "allowed_tools": session.allowed_tools,
                        "expected_evidence": task.expected_evidence,
                        "stop_condition": task.stop_condition,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _confidence(ok: bool, evidence_count: int) -> float:
        if not ok:
            return 0.0
        if evidence_count <= 0:
            return 0.2
        return min(0.95, 0.55 + evidence_count * 0.05)

    def _summarize_report(
        self,
        task: RetrievalTask,
        ok: bool,
        evidence_count: int,
        error: str | None,
    ) -> str:
        if not ok:
            return f"{self.agent_name} failed {self.seek_tool}: {error or 'unknown error'}"
        return f"{self.agent_name} completed {self.seek_tool}; reported {evidence_count} evidence items."


class DocTextSubagent(ExpertSubagent):
    agent_name = "doc_text_subagent"
    seek_tool = "doc_text_seek"
    role_prompt = DOC_TEXT_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.text_expert


class DocVisualSubagent(ExpertSubagent):
    agent_name = "doc_visual_subagent"
    seek_tool = "doc_visual_seek"
    role_prompt = DOC_VISUAL_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.visual_expert

    async def _run_model_session(self, task: RetrievalTask) -> SubagentResult:
        result = await super()._run_model_session(task)
        return await self._attach_visual_inspections(task, result)

    async def _attach_visual_inspections(
        self,
        task: RetrievalTask,
        result: SubagentResult,
    ) -> SubagentResult:
        inspections = await self._inspect_visual_assets(task, result.evidence)
        if not inspections:
            return result
        inspection_by_id = {item["evidence_id"]: item for item in inspections}
        for item in result.evidence:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or "")
            inspection = inspection_by_id.get(evidence_id)
            if not inspection:
                continue
            metadata = item.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["vlm_visual_inspection"] = inspection
            score_parts = item.get("score_parts")
            if isinstance(score_parts, dict):
                score_parts["visual"] = float(score_parts.get("visual", 0.0) or 0.0) + 0.45
                score_parts["rerank"] = float(score_parts.get("rerank", 0.0) or 0.0) + 0.2
            item["score"] = float(item.get("score", 0.0) or 0.0) + 0.35
        quality_report = self._replace_reports_with_inspected_evidence(
            task,
            evidence=result.evidence,
            filtering_note="vlm_visual_inspection attached to source visual blocks",
            inspection_metadata={
                "inspection_type": "source_image_vlm",
                "inspection_count": len(inspections),
                "inspected_evidence_ids": [item["evidence_id"] for item in inspections],
            },
        )
        if quality_report:
            self._sync_result_from_reports(task, result, quality_report)
        result.data["vlm_visual_inspections"] = inspections
        return result

    async def _inspect_visual_assets(
        self,
        task: RetrievalTask,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        inspections: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        query_text = task.query.casefold()
        try:
            max_inspections = int(task.params.get("max_visual_inspections") or 0)
        except (TypeError, ValueError):
            max_inspections = 0
        if max_inspections <= 0:
            visual_numeric = bool(task.params.get("doc_ids")) and any(
                token in query_text
                for token in ("chart", "table", "figure", "percentage", "percent", "how many")
            )
            max_inspections = 6 if visual_numeric else 3
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if len(inspections) >= max_inspections:
                break
            asset_path = _visual_asset_path(item)
            if not asset_path or asset_path in seen_paths:
                continue
            image_content = _image_url_content(asset_path)
            if image_content is None:
                continue
            seen_paths.add(asset_path)
            payload = {
                "question": task.query,
                "evidence_id": item.get("id"),
                "modality": item.get("modality"),
                "locator": item.get("locator"),
                "visual_trace": (item.get("metadata") or {}).get("visual_trace")
                if isinstance(item.get("metadata"), dict)
                else None,
            }
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": VISUAL_INSPECTION_USER_TEXT,
                },
                image_content,
            ]
            try:
                response = await self.provider.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": VISUAL_INSPECTION_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": content},
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=900,
                    response_format={"type": "json_object"},
                )
                raw_content = response.content or "{}"
                try:
                    parsed = json.loads(raw_content)
                except Exception:
                    match = re.search(r"\{[\s\S]*\}", raw_content)
                    parsed = json.loads(match.group(0)) if match else {"summary": "", "uncertainty": "non-json VLM response"}
            except Exception as exc:
                parsed = {"summary": "", "uncertainty": f"VLM inspection failed: {exc}"}
            if not isinstance(parsed, dict):
                parsed = {"summary": str(parsed), "uncertainty": "non-object VLM response"}
            parsed["evidence_id"] = str(item.get("id") or "")
            parsed["asset_path"] = asset_path
            inspections.append(parsed)
        return inspections


class DocGraphSubagent(ExpertSubagent):
    agent_name = "doc_graph_subagent"
    seek_tool = "doc_graph_seek"
    role_prompt = DOC_GRAPH_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.graph_expert

    async def _run_model_session(self, task: RetrievalTask) -> SubagentResult:
        result = await super()._run_model_session(task)
        return await self._attach_graph_visual_inspections(task, result)

    async def _attach_graph_visual_inspections(
        self,
        task: RetrievalTask,
        result: SubagentResult,
    ) -> SubagentResult:
        if not result.evidence:
            return result
        visual_candidates = [
            item
            for item in result.evidence
            if isinstance(item, dict)
            if _visual_asset_path(item)
            and str(item.get("modality") or "").lower() in {"table", "chart", "image", "figure"}
        ]
        if not visual_candidates:
            return result
        inspector = DocVisualSubagent(
            self.tools,
            self.evidence_writer,
            runner=self.runner,
            model=self.model,
        )
        inspections = await inspector._inspect_visual_assets(task, visual_candidates)
        if not inspections:
            return result
        inspection_by_id = {item["evidence_id"]: item for item in inspections}
        for item in result.evidence:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or "")
            inspection = inspection_by_id.get(evidence_id)
            if not inspection:
                continue
            metadata = item.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["vlm_visual_inspection"] = inspection
            score_parts = item.get("score_parts")
            if isinstance(score_parts, dict):
                score_parts["visual"] = float(score_parts.get("visual", 0.0) or 0.0) + 0.45
                score_parts["rerank"] = float(score_parts.get("rerank", 0.0) or 0.0) + 0.2
            item["score"] = float(item.get("score", 0.0) or 0.0) + 0.35
        quality_report = self._replace_reports_with_inspected_evidence(
            task,
            evidence=result.evidence,
            filtering_note="vlm_visual_inspection attached to graph-returned visual blocks",
            inspection_metadata={
                "inspection_type": "graph_source_image_vlm",
                "inspection_count": len(inspections),
                "inspected_evidence_ids": [item["evidence_id"] for item in inspections],
            },
        )
        if quality_report:
            self._sync_result_from_reports(task, result, quality_report)
        result.data["vlm_visual_inspections"] = inspections
        return result


class VideoTextSubagent(ExpertSubagent):
    agent_name = "video_text_subagent"
    seek_tool = "video_text_seek"
    role_prompt = VIDEO_TEXT_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.text_expert


class VideoVisualSubagent(ExpertSubagent):
    agent_name = "video_visual_subagent"
    seek_tool = "video_visual_seek"
    role_prompt = VIDEO_VISUAL_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.visual_expert


class VideoGraphSubagent(ExpertSubagent):
    agent_name = "video_graph_subagent"
    seek_tool = "video_graph_seek"
    role_prompt = VIDEO_GRAPH_SUBAGENT_ROLE
    default_model = DEFAULT_MODELS.graph_expert
