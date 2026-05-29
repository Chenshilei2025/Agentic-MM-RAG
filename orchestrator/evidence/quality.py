"""Deterministic evidence inspection shared by orchestration components."""

from __future__ import annotations

import re
from typing import Any


TOKEN_CANONICAL = {
    "drivers": "driver",
    "driving": "driver",
    "drive": "driver",
    "blockers": "blocker",
    "blocking": "blocker",
}


RELEVANCE_STOPWORDS = {
    "about",
    "action",
    "actions",
    "affect",
    "and",
    "answer",
    "based",
    "cause",
    "causes",
    "conditions",
    "describe",
    "difference",
    "does",
    "during",
    "engage",
    "evidence",
    "factors",
    "focusing",
    "from",
    "have",
    "how",
    "including",
    "interactions",
    "involved",
    "its",
    "into",
    "like",
    "make",
    "main",
    "mechanism",
    "mechanisms",
    "process",
    "prompts",
    "question",
    "reasons",
    "roles",
    "scene",
    "search",
    "show",
    "showcasing",
    "such",
    "their",
    "the",
    "these",
    "this",
    "video",
    "visible",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def evidence_excerpt(item: dict[str, Any], *, max_chars: int = 1200) -> str:
    content = str(item.get("content") or item.get("text") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    transcript = str(metadata.get("transcript") or "").strip()
    if transcript and transcript not in content:
        content = f"{content}\nTranscript detail:\n{transcript}"
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content[:max_chars]


def tokenize_for_relevance(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text.casefold()):
        if token in RELEVANCE_STOPWORDS:
            continue
        tokens.add(TOKEN_CANONICAL.get(token, token))
    return tokens


def as_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def term_tokens(terms: list[str]) -> set[str]:
    result: set[str] = set()
    for term in terms:
        result.update(tokenize_for_relevance(term))
    return result


def query_quality_profile(question: str, rewritten: dict[str, Any] | None = None) -> dict[str, Any]:
    rewritten = rewritten or {}
    question_tokens = tokenize_for_relevance(question)
    entity_tokens = term_tokens(as_terms(rewritten.get("entities")))
    action_tokens = term_tokens(as_terms(rewritten.get("actions")))
    keyword_tokens = term_tokens(as_terms(rewritten.get("textual_keywords")))
    anchor_tokens = term_tokens(as_terms(rewritten.get("visual_anchors")))
    if not entity_tokens:
        entity_tokens = question_tokens
    return {
        "question_tokens": sorted(question_tokens),
        "entity_tokens": sorted(entity_tokens),
        "action_tokens": sorted(action_tokens),
        "keyword_tokens": sorted((keyword_tokens | anchor_tokens) - entity_tokens - action_tokens),
        "required_tokens": sorted(entity_tokens | action_tokens),
    }


def support_profile(
    text: str,
    *,
    question: str,
    rewritten: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = query_quality_profile(question, rewritten)
    tokens = tokenize_for_relevance(text)
    entity_hits = set(profile["entity_tokens"]) & tokens
    action_hits = set(profile["action_tokens"]) & tokens
    keyword_hits = set(profile["keyword_tokens"]) & tokens
    question_hits = set(profile["question_tokens"]) & tokens
    numeric_hits = set(re.findall(r"\b\d+(?:[./:%:-]\d+)*\b", text))
    lower_text = text.casefold()
    lower_question = question.casefold()
    wants_comparison = any(token in lower_question for token in ("compare", "difference", "versus", "advantage", "over", "predecessor"))
    wants_cause = any(token in lower_question for token in ("why", "cause", "reason", "prompt", "lead", "enable", "strategy", "technique", "how"))
    relation_markers = {
        "because",
        "therefore",
        "so",
        "leads",
        "enables",
        "helps",
        "improves",
        "reduces",
        "increases",
        "compared",
        "better",
        "worse",
        "than",
        "by",
        "through",
    }
    relation_hits = {marker for marker in relation_markers if marker in lower_text}
    example_markers = {"for example", "such as", "including", "instance", "e.g.", "like"}
    example_hit = any(marker in lower_text for marker in example_markers)
    contrast_hit = wants_comparison and any(marker in lower_text for marker in ("compared", "than", "versus", "better", "worse", "higher", "lower"))
    cause_hit = wants_cause and bool(relation_hits)
    direct_answer_support = bool(entity_hits or question_hits) and (
        bool(action_hits)
        or bool(keyword_hits)
        or bool(numeric_hits)
        or bool(cause_hit)
        or bool(contrast_hit)
        or len(question_hits) >= 2
    )
    background_only = bool(entity_hits or question_hits) and not direct_answer_support
    return {
        "entity_hit": sorted(entity_hits),
        "action_hit": sorted(action_hits),
        "keyword_hit": sorted(keyword_hits),
        "question_hit": sorted(question_hits),
        "metric_hit": sorted(numeric_hits),
        "relation_hit": sorted(relation_hits),
        "example_hit": example_hit,
        "contrast_hit": contrast_hit,
        "cause_or_mechanism_hit": cause_hit,
        "direct_answer_support": direct_answer_support,
        "background_only": background_only,
    }


def inspect_evidence_item(
    item: dict[str, Any],
    *,
    question: str,
    rewritten: dict[str, Any] | None = None,
    retrieval_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = query_quality_profile(question, rewritten)
    entity_tokens = set(profile["entity_tokens"])
    action_tokens = set(profile["action_tokens"])
    keyword_tokens = set(profile["keyword_tokens"])
    question_tokens = set(profile["question_tokens"])
    required_tokens = set(profile["required_tokens"])

    text = evidence_excerpt(item, max_chars=1800)
    support = support_profile(text, question=question, rewritten=rewritten)
    tokens = tokenize_for_relevance(text)
    hints = retrieval_hints or {}
    off_topic_markers = set()
    for term in hints.get("off_topic_markers", []) if isinstance(hints.get("off_topic_markers"), list) else []:
        term_text = str(term).strip()
        if term_text:
            off_topic_markers.add(term_text)
    entity_hits = entity_tokens & tokens
    action_hits = action_tokens & tokens
    keyword_hits = keyword_tokens & tokens
    question_hits = question_tokens & tokens
    noise_hits = (off_topic_markers - entity_tokens) & tokens
    off_topic_hits = noise_hits if not entity_hits else set()

    relevance = 0.0
    relevance += len(entity_hits) * 2.2
    relevance += len(action_hits) * 1.5
    relevance += min(4, len(keyword_hits)) * 0.7
    relevance += min(5, len(question_hits)) * 0.25
    if support["direct_answer_support"]:
        relevance += 1.6
    elif support["background_only"]:
        relevance -= 0.6
    if support["metric_hit"]:
        relevance += 0.5
    if support["example_hit"]:
        relevance += 0.35
    if support["contrast_hit"] or support["cause_or_mechanism_hit"]:
        relevance += 0.6
    if required_tokens and not (required_tokens & tokens):
        relevance -= 1.5
    if entity_tokens and not entity_hits:
        relevance -= 1.0
    if noise_hits:
        relevance -= min(1.8, len(noise_hits) * 0.6)
    if off_topic_hits:
        relevance -= 1.25
    relevance += min(0.75, float(item.get("fused_score", item.get("score", 0.0)) or 0.0) * 0.15)

    return {
        "relevance": round(relevance, 4),
        "is_relevant": relevance > 0 and bool(entity_hits or action_hits or question_hits),
        "off_topic": bool(off_topic_hits),
        "mixed_noise": bool(noise_hits and entity_hits),
        "support": support,
        "hits": {
            "entities": sorted(entity_hits),
            "actions": sorted(action_hits),
            "keywords": sorted(keyword_hits),
            "question": sorted(question_hits),
            "off_topic": sorted(off_topic_hits),
            "noise": sorted(noise_hits),
        },
    }


def inspect_evidence_batch(
    evidence: list[dict[str, Any]],
    *,
    question: str,
    rewritten: dict[str, Any] | None = None,
    retrieval_hints: dict[str, Any] | None = None,
    limit: int | None = None,
    min_keep: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        quality = inspect_evidence_item(
            item,
            question=question,
            rewritten=rewritten,
            retrieval_hints=retrieval_hints,
        )
        enriched = dict(item)
        review_key = f"id:{item.get('id')}" if item.get("id") else f"idx:{index}"
        enriched["_review_key"] = review_key
        enriched["quality"] = quality
        enriched["generation_relevance"] = quality["relevance"]
        enriched["generation_relevance_hits"] = quality["hits"]
        inspected.append(enriched)

    inspected.sort(key=lambda item: float((item.get("quality") or {}).get("relevance", 0.0)), reverse=True)
    selected = [item for item in inspected if (item.get("quality") or {}).get("is_relevant")]
    if len(selected) < min_keep:
        selected = [item for item in inspected if float((item.get("quality") or {}).get("relevance", 0.0)) > -0.25]
    if len(selected) < min_keep:
        selected = inspected[:min(min_keep, len(inspected))]
    if limit is not None:
        selected = selected[:limit]

    covered_tokens: set[str] = set()
    for item in selected:
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        hits = quality.get("hits") if isinstance(quality.get("hits"), dict) else {}
        for values in hits.values():
            if isinstance(values, list):
                covered_tokens.update(str(value) for value in values)

    selected_keys = {str(item.get("_review_key")) for item in selected}
    dropped = [item for item in inspected if str(item.get("_review_key")) not in selected_keys]
    diagnostics = {
        **query_quality_profile(question, rewritten),
        "covered_tokens": sorted(covered_tokens),
        "selected_count": len(selected),
        "dropped_count": len(dropped),
        "ranked_evidence_ids": [item.get("id") for item in selected],
        "ranked_scores": {
            str(item.get("id") or index): float((item.get("quality") or {}).get("relevance", 0.0))
            for index, item in enumerate(selected)
        },
        "dropped_evidence_ids": [item.get("id") for item in dropped],
        "off_topic_evidence_ids": [
            item.get("id")
            for item in inspected
            if (item.get("quality") or {}).get("off_topic")
        ],
    }
    for item in inspected:
        item.pop("_review_key", None)
    return selected, diagnostics


def guardrail_evidence_batch(
    evidence: list[dict[str, Any]],
    *,
    question: str,
    rewritten: dict[str, Any] | None = None,
    retrieval_hints: dict[str, Any] | None = None,
    min_keep: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve LLM ordering while removing only obvious noise."""

    inspected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in evidence:
        quality = inspect_evidence_item(
            item,
            question=question,
            rewritten=rewritten,
            retrieval_hints=retrieval_hints,
        )
        enriched = dict(item)
        enriched["quality"] = quality
        enriched["generation_relevance"] = quality["relevance"]
        enriched["generation_relevance_hits"] = quality["hits"]
        obvious_noise = bool(quality.get("off_topic")) or (
            float(quality.get("relevance", 0.0)) < -0.75
            and not (quality.get("hits") or {}).get("entities")
            and not (quality.get("hits") or {}).get("question")
        )
        if obvious_noise:
            rejected.append(enriched)
        else:
            inspected.append(enriched)

    if len(inspected) < min_keep and evidence:
        kept_ids = {id(item) for item in inspected}
        for item in evidence:
            if id(item) in kept_ids:
                continue
            quality = inspect_evidence_item(
                item,
                question=question,
                rewritten=rewritten,
                retrieval_hints=retrieval_hints,
            )
            enriched = dict(item)
            enriched["quality"] = quality
            enriched["generation_relevance"] = quality["relevance"]
            enriched["generation_relevance_hits"] = quality["hits"]
            inspected.append(enriched)
            break

    covered_tokens: set[str] = set()
    for item in inspected:
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        hits = quality.get("hits") if isinstance(quality.get("hits"), dict) else {}
        for values in hits.values():
            if isinstance(values, list):
                covered_tokens.update(str(value) for value in values)

    diagnostics = {
        **query_quality_profile(question, rewritten),
        "review_mode": "llm_order_guardrail",
        "covered_tokens": sorted(covered_tokens),
        "selected_count": len(inspected),
        "dropped_count": len(rejected),
        "ranked_evidence_ids": [item.get("id") for item in inspected],
        "ranked_scores": {
            str(item.get("id") or index): float((item.get("quality") or {}).get("relevance", 0.0))
            for index, item in enumerate(inspected)
        },
        "dropped_evidence_ids": [item.get("id") for item in rejected],
        "off_topic_evidence_ids": [
            item.get("id")
            for item in rejected
            if (item.get("quality") or {}).get("off_topic")
        ],
    }
    return inspected, diagnostics
