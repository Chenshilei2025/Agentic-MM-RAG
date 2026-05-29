"""Shared scoring and fusion helpers for retrieval backends."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from agentic_mm_rag.schemas import ToolResponse


DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "what",
    "when",
    "where",
    "which",
    "about",
    "into",
    "does",
    "are",
    "was",
    "were",
    "how",
    "why",
}


CANONICAL_TOKENS = {
    "drivers": "driver",
    "driving": "driver",
    "blockers": "blocker",
}

TEMPORAL_CHANGE_TERMS = {
    "gain",
    "gained",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "change",
    "changed",
    "difference",
    "delta",
    "most",
    "largest",
}
GROUP_TERMS = {
    "age",
    "ages",
    "category",
    "categories",
    "cohort",
    "cohorts",
    "demographic",
    "demographics",
    "education",
    "gender",
    "group",
    "groups",
    "population",
    "segment",
    "segments",
    "subgroup",
    "subgroups",
}
MEASURE_TERMS = {
    "chart",
    "figure",
    "percent",
    "percentage",
    "point",
    "points",
    "rate",
    "share",
    "survey",
    "table",
    "value",
    "values",
}


def normalize_score(value: float, max_value: float) -> float:
    """Normalize a non-negative score into the closed interval [0, 1]."""

    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def keywords(text: str, *, stopwords: set[str] | None = None) -> set[str]:
    """Extract lightweight retrieval keywords from free text."""

    ignored = DEFAULT_STOPWORDS if stopwords is None else stopwords
    return {
        CANONICAL_TOKENS.get(token, token)
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token not in ignored
    }


def numeric_tokens(text: str) -> set[str]:
    """Extract simple numeric tokens used for answer-focused inspection."""

    return set(re.findall(r"\b\d+(?:[./:-]\d+)*\b", text))


def infer_doc_query_profile(query_text: str) -> dict[str, bool]:
    """Infer coarse document-query traits used to weight retrieval stages."""

    text = query_text.lower()
    table_like = any(token in text for token in ("table", "tabular", "row", "column", "population", "sum", "survey"))
    temporal_like = any(
        token in text for token in ("year", "date", "when", "before", "after", "during", "timeline")
    )
    count_like = any(
        token in text
        for token in ("how many", "count", "number of", "total", "sum", "greater", "less")
    )
    visual_like = any(
        token in text for token in ("chart", "figure", "image", "photo", "diagram", "axis", "page", "visual")
    )
    compare_like = any(
        token in text for token in ("greater", "larger", "higher", "more than", "less than")
    )
    return {
        "table_like": table_like,
        "temporal_like": temporal_like,
        "count_like": count_like,
        "visual_like": visual_like,
        "compare_like": compare_like,
    }


def fuse_evidence_items(
    evidence_items: list[dict[str, Any]],
    *,
    top_k: int = 12,
    diversity_by_source: bool = True,
    query_text: str | None = None,
) -> ToolResponse:
    """Deterministic fusion over serialized EvidenceCard dictionaries."""

    weights = {
        "text": 1.0,
        "graph": 1.25,
        "visual": 1.1,
        "source_filter": 0.35,
        "rerank": 1.4,
    }
    query = (query_text or "").lower()
    if any(token in query for token in ("look", "visible", "scene", "frame", "action", "show")):
        weights["visual"] = 1.35
        weights["graph"] = 1.05
    if any(token in query for token in ("chart", "table", "figure", "percentage", "percent", "how many", "greater", "lower", "less than", "references")):
        weights["visual"] = max(weights["visual"], 1.45)
        weights["rerank"] = max(weights["rerank"], 1.55)
        weights["graph"] = min(weights["graph"], 1.05)
    if any(token in query for token in ("why", "how", "relationship", "relation", "cause", "path", "evolve")):
        weights["graph"] = 1.4
    if any(token in query for token in ("said", "say", "name", "number", "when", "where", "who")):
        weights["text"] = 1.15
    query_terms = keywords(query)
    wants_comparison = any(token in query for token in ("compare", "difference", "versus", "advantage", "over"))
    wants_cause = any(token in query for token in ("why", "cause", "reason", "how", "strategy", "technique", "enable"))
    wants_temporal_delta = (
        len(re.findall(r"\b(?:19|20)\d{2}\b", query)) >= 2
        and any(token in query_terms for token in TEMPORAL_CHANGE_TERMS)
        and bool(query_terms & (GROUP_TERMS | MEASURE_TERMS))
    )
    best_by_id: dict[str, dict[str, Any]] = {}
    for item in evidence_items:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        score_parts = item.get("score_parts") if isinstance(item.get("score_parts"), dict) else {}
        fused_score = sum(
            float(score_parts.get(key, 0.0) or 0.0) * value
            for key, value in weights.items()
        )
        if fused_score <= 0:
            fused_score = float(item.get("score", 0.0) or 0.0)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        modality = str(item.get("modality") or "")
        if "vlm_visual_inspection" in metadata:
            fused_score += 0.65
        if modality in {"chart", "table", "image"} and any(
            token in query
            for token in ("chart", "table", "figure", "percentage", "percent", "how many", "greater", "lower", "less than", "references")
        ):
            fused_score += 0.35
        if str(item.get("modality") or "") == "graph" and any(
            token in query for token in ("chart", "table", "figure", "percentage", "percent", "how many")
        ):
            fused_score -= 0.3
        content = str(item.get("content") or item.get("text") or "")
        if wants_temporal_delta:
            has_delta = bool(re.search(r"\+\s?\d+\s*(?:percentage\s+)?points?", content, re.IGNORECASE))
            content_terms = keywords(content)
            has_group = bool(content_terms & GROUP_TERMS)
            if has_delta and has_group:
                fused_score += 1.35
                if str(item.get("modality") or "") == "text":
                    fused_score += 0.35
        if query_terms:
            content_terms = keywords(content)
            matched = query_terms & content_terms
            coverage = len(matched) / max(len(query_terms), 1)
            multi_anchor_bonus = min(0.35, max(0, len(matched) - 1) * 0.08)
            fused_score += min(0.45, coverage * 0.45) + multi_anchor_bonus
            support = item.get("quality", {}).get("support") if isinstance(item.get("quality"), dict) else None
            if isinstance(support, dict):
                if support.get("direct_answer_support"):
                    fused_score += 0.4
                elif support.get("background_only"):
                    fused_score -= 0.35
                if support.get("contrast_hit") and wants_comparison:
                    fused_score += 0.3
                if support.get("cause_or_mechanism_hit") and wants_cause:
                    fused_score += 0.25
            if not matched and str(item.get("source_type")) == "video":
                fused_score *= 0.65
            elif str(item.get("source_type")) == "video" and coverage < 0.12:
                fused_score *= 0.82
        record = dict(item)
        record["fused_score"] = fused_score
        if item_id not in best_by_id or fused_score > float(
            best_by_id[item_id].get("fused_score", 0.0)
        ):
            best_by_id[item_id] = record

    ranked = sorted(
        best_by_id.values(),
        key=lambda item: float(item.get("fused_score", 0.0)),
        reverse=True,
    )
    sources = {
        str(item.get("source_id") or item.get("source_type") or "unknown")
        for item in ranked
    }
    if diversity_by_source and len(sources) > 1:
        selected: list[dict[str, Any]] = []
        source_counts: dict[str, int] = defaultdict(int)
        for item in ranked:
            source = str(item.get("source_id") or item.get("source_type") or "unknown")
            if source_counts[source] >= max(2, top_k // 3):
                continue
            selected.append(item)
            source_counts[source] += 1
            if len(selected) >= top_k:
                break
        ranked = selected
    else:
        ranked = ranked[:top_k]

    return ToolResponse(
        ok=True,
        tool="evidence_score_fuse",
        data={"items": ranked, "count": len(ranked)},
    )
