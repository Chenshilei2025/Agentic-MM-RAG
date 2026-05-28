"""System-level evidence sufficiency, gap, and conflict audit."""

from __future__ import annotations

import re
from typing import Any

from agentic_mm_rag.agent.evidence_quality import inspect_evidence_batch
from agentic_mm_rag.agent.types import SubagentResult


def _item_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    inspection = metadata.get("vlm_visual_inspection") if isinstance(metadata.get("vlm_visual_inspection"), dict) else {}
    parts = [
        str(item.get("content") or item.get("text") or ""),
        str(metadata.get("transcript") or ""),
        str(metadata.get("caption") or ""),
        str(metadata.get("visual_caption") or ""),
        str(inspection.get("summary") or ""),
        str(inspection.get("answer_candidate") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _item_key(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("source_id") or "unknown")


def _modality_bucket(item: dict[str, Any]) -> str:
    source_type = str(item.get("source_type") or "")
    modality = str(item.get("modality") or "")
    if source_type == "video":
        if modality in {"video_segment", "image", "chart", "table"}:
            return "video_visual" if modality != "video_segment" else "video_text"
        return "video_text"
    if modality in {"image", "chart", "table", "figure"}:
        return "doc_visual"
    if modality in {"entity", "relation"} or "graph" in str(item.get("provenance") or "").lower():
        return "doc_graph"
    return "doc_text" if source_type == "doc" else source_type or modality or "unknown"


def _numeric_values(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def _answer_candidates(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    text = _item_text(item)
    values.update(_numeric_values(text))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    inspection = metadata.get("vlm_visual_inspection") if isinstance(metadata.get("vlm_visual_inspection"), dict) else {}
    for key in ("answer_candidate", "calculation"):
        value = inspection.get(key)
        if isinstance(value, str):
            values.update(_numeric_values(value))
            if value.strip() and len(value.strip()) <= 80:
                values.add(value.strip().casefold())
    return {value for value in values if value}


def audit_evidence(
    *,
    question: str,
    rewritten: dict[str, Any] | None,
    results: list[SubagentResult],
    expected_modalities: list[str],
    retrieval_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic system audit used by decision reflection."""

    items = [item for result in results for item in result.evidence if isinstance(item, dict)]
    inspected, diagnostics = inspect_evidence_batch(
        items,
        question=question,
        rewritten=rewritten,
        retrieval_hints=retrieval_hints,
        limit=None,
        min_keep=1,
    )
    coverage: dict[str, int] = {}
    direct_support_by_modality: dict[str, int] = {}
    weak_support: list[dict[str, Any]] = []
    for item in inspected:
        bucket = _modality_bucket(item)
        coverage[bucket] = coverage.get(bucket, 0) + 1
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        support = quality.get("support") if isinstance(quality.get("support"), dict) else {}
        if support.get("direct_answer_support"):
            direct_support_by_modality[bucket] = direct_support_by_modality.get(bucket, 0) + 1
        elif support.get("background_only") or not quality.get("is_relevant"):
            weak_support.append(
                {
                    "evidence_id": _item_key(item),
                    "modality": bucket,
                    "reason": "background_only" if support.get("background_only") else "low_relevance",
                    "relevance": quality.get("relevance"),
                }
            )

    missing_obligations: list[dict[str, Any]] = []
    for token in diagnostics.get("required_tokens", []):
        if token not in set(diagnostics.get("covered_tokens", [])):
            missing_obligations.append(
                {
                    "type": "required_token",
                    "term": token,
                    "reason": "required entity/action was not covered by kept evidence",
                }
            )
    if not direct_support_by_modality:
        missing_obligations.append(
            {
                "type": "direct_support",
                "reason": "no modality produced direct answer support",
            }
        )

    conflicts: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, str]]] = {}
    for item in inspected:
        bucket = _modality_bucket(item)
        for candidate in _answer_candidates(item):
            by_candidate.setdefault(candidate, []).append(
                {"evidence_id": _item_key(item), "modality": bucket}
            )
    numeric_candidates = [candidate for candidate in by_candidate if re.search(r"\d", candidate)]
    if len(numeric_candidates) > 1:
        conflicts.append(
            {
                "type": "answer_candidate_mismatch",
                "severity": "medium",
                "candidates": {
                    candidate: by_candidate[candidate] for candidate in numeric_candidates[:6]
                },
                "reason": "different numeric answer candidates appear across evidence",
            }
        )

    followup_tasks: list[dict[str, Any]] = []
    expected = set(expected_modalities)
    if "doc" in expected and not any(key.startswith("doc_") for key in coverage):
        followup_tasks.append({"agent": "doc_text_subagent", "reason": "missing document evidence"})
    if "video" in expected and not any(key.startswith("video_") for key in coverage):
        followup_tasks.append({"agent": "video_text_subagent", "reason": "missing video evidence"})
    question_lower = question.casefold()
    if any(token in question_lower for token in ("show", "visible", "chart", "table", "figure", "image")):
        if not any(key.endswith("_visual") for key in coverage):
            followup_tasks.append({"agent": "doc_visual_subagent", "reason": "visual question lacks visual evidence"})
    if conflicts:
        followup_tasks.append({"agent": "doc_visual_subagent", "reason": "resolve cross-modal conflict"})

    sufficient = bool(inspected) and not missing_obligations and not conflicts
    return {
        "reviewer": "deterministic_system_evidence_audit",
        "sufficient": sufficient,
        "modality_coverage": coverage,
        "direct_support_by_modality": direct_support_by_modality,
        "missing_obligations": missing_obligations,
        "cross_modal_conflicts": conflicts,
        "weak_support_claims": weak_support[:12],
        "followup_tasks": followup_tasks,
        "quality_diagnostics": diagnostics,
    }
