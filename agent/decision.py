"""Decision agent for planning, reflection, and final synthesis."""

from __future__ import annotations

import json
from itertools import count
import re
import ast
from typing import Any

from agentic_mm_rag.agent.contracts import (
    allowed_tools_for_contract,
    choose_doc_planner_route,
    infer_doc_edge_type_filter,
    infer_doc_query_profile,
    preferred_doc_modalities,
)
from agentic_mm_rag.orchestrator.types import AgentPlan, QueryContext, ReflectionResult, RetrievalTask, SubagentResult
from agentic_mm_rag.orchestrator.evidence.quality import inspect_evidence_batch, query_quality_profile
from agentic_mm_rag.agent.prompts import generation_system_prompt
from agentic_mm_rag.config import DEFAULT_MODELS
from agentic_mm_rag.orchestrator.evidence.audit import audit_evidence
from agentic_mm_rag.providers import LLMProvider


def _humanize_generated_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested in value.items():
            label = str(key).replace("_", " ").strip().capitalize()
            rendered = _humanize_generated_value(nested)
            if rendered:
                parts.append(f"{label}: {rendered}.")
        return " ".join(parts)
    if isinstance(value, list):
        rendered_items = [_humanize_generated_value(item).strip() for item in value]
        rendered_items = [item.rstrip(".") for item in rendered_items if item]
        return "; ".join(rendered_items)
    return str(value).strip()


def _question_coverage_plan(question: str, rewritten: dict[str, Any] | None = None) -> list[str]:
    """Return answer obligations inferred from the question."""
    text = question.lower()
    obligations: list[str] = []
    if any(token in text for token in ("how", "mechanism", "process", "construct", "build", "coordinate", "evade")):
        obligations.append("explain the mechanism or sequence of actions")
    if any(token in text for token in ("why", "what prompts", "cause", "reason", "affect")):
        obligations.append("identify causes, triggers, and effects")
    if any(token in text for token in ("difference", "compare", "compared", "versus", "advantages", "over")):
        obligations.append("compare the alternatives point by point")
    if any(token in text for token in ("theme", "morality", "destiny", "freedom", "ethical", "considerations")):
        obligations.append("connect concrete events to broader themes or implications")
    if any(token in text for token in ("conflict", "resolution", "story")):
        obligations.append("name each major conflict and its resolution")
    if any(token in text for token in ("strategy", "priorities", "advancements", "enhance", "reducing")):
        obligations.append("cover goals, methods, examples, and expected outcomes")
    if any(token in text for token in ("what", "describe", "main")):
        obligations.append("answer the direct question first, then add supporting detail")
    if rewritten:
        qtype = str(rewritten.get("question_type") or "").lower()
        if qtype in {"comparison", "relation", "cause", "temporal", "mechanism"}:
            obligations.append(f"treat this as a {qtype} question and make the reasoning explicit")
    obligations.append("use concrete names, roles, scenes, text facts, or timestamps from evidence")
    obligations.append("separate evidence-supported facts from cautious synthesis")
    return list(dict.fromkeys(obligations))


def _routing_skill_profile(question: str, rewritten: dict[str, Any] | None = None) -> dict[str, Any]:
    """Intent signals for optional follow-up routing."""
    rewritten = rewritten or {}
    text = question.casefold()
    qtype = str(rewritten.get("question_type") or "").casefold()
    entities = [str(item).strip() for item in rewritten.get("entities", []) if str(item).strip()]
    actions = [str(item).strip() for item in rewritten.get("actions", []) if str(item).strip()]
    factual_markers = (
        "what", "which", "who", "where", "when", "phrase", "term",
        "called", "name", "specific", "goal", "advantages", "differences",
    )
    event_markers = (
        "how", "why", "what prompts", "encounter", "challenge", "avoid",
        "communicate", "coordinate", "interact", "roles", "conflict", "outcome",
        "process", "workflow", "steps", "method", "mechanism",
    )
    skills = {
        "factual_detail": qtype == "factual" or any(token in text for token in factual_markers),
        "event_chain": qtype in {"mechanism", "cause", "relation", "temporal", "challenge"} or any(token in text for token in event_markers),
        "procedural_detail": any(token in text for token in ("workflow", "steps", "procedure", "method", "technique", "system", "tool")),
        "needs_graph": False,
        "needs_visual_crosscheck": False,
        "needs_text_precision": False,
    }
    skills["needs_graph"] = bool(
        skills["event_chain"]
        or qtype in {"comparison", "relation", "temporal"}
    )
    skills["needs_visual_crosscheck"] = bool(
        any(token in text for token in ("visible", "scene", "look", "show", "frame", "image", "chart", "table", "figure", "action"))
    )
    skills["needs_text_precision"] = bool(skills["factual_detail"] or skills["procedural_detail"] or len(entities) >= 2)
    skills["core_terms"] = list(dict.fromkeys(entities + actions))
    return skills


def _retrieval_hints(query: QueryContext) -> dict[str, Any]:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    hints = metadata.get("retrieval_hints")
    if isinstance(hints, dict):
        return dict(hints)
    return {}


def _hint_answer_format(query: QueryContext) -> str:
    hints = _retrieval_hints(query)
    if hints.get("answer_format"):
        return str(hints.get("answer_format")).strip()
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    return str(metadata.get("answer_format") or "").strip()


def _hint_evidence_pages(query: QueryContext) -> list[int] | None:
    hints = _retrieval_hints(query)
    pages = hints.get("evidence_pages")
    if pages is None and isinstance(query.metadata, dict):
        pages = query.metadata.get("evidence_pages")
    if isinstance(pages, str):
        try:
            pages = json.loads(pages)
        except Exception:
            try:
                pages = ast.literal_eval(pages)
            except Exception:
                return None
    if not isinstance(pages, list):
        return None
    result: list[int] = []
    for page in pages:
        try:
            result.append(int(page))
        except (TypeError, ValueError):
            continue
    return result or None


def _hint_page_bias_pages(query: QueryContext) -> list[int] | None:
    hints = _retrieval_hints(query)
    pages = hints.get("page_bias_pages")
    if pages is None and isinstance(query.metadata, dict):
        pages = query.metadata.get("page_bias_pages")
    if isinstance(pages, str):
        try:
            pages = json.loads(pages)
        except Exception:
            try:
                pages = ast.literal_eval(pages)
            except Exception:
                pages = None
    if isinstance(pages, list):
        result: list[int] = []
        for page in pages:
            try:
                result.append(int(page))
            except (TypeError, ValueError):
                continue
        if result:
            return sorted({page for page in result if page > 0}) or None
    return _extract_page_bias_pages(query.query_text)


def _extract_page_bias_pages(question: str) -> list[int] | None:
    text = question.casefold().replace("first page", "page 1").replace("second page", "page 2")
    pages: set[int] = set()
    for match in re.finditer(r"\bpage\s+(\d+)\b", text):
        try:
            pages.add(int(match.group(1)))
        except ValueError:
            continue
    if "second page" in text:
        pages.add(2)
    if "third page" in text:
        pages.add(3)
    if "last page" in text:
        pages.add(-1)
    return sorted(page for page in pages if page > 0) or None


def _page_fanout_pages(
    evidence_pages: list[int] | None,
    page_bias_pages: list[int] | None,
    *,
    max_pages: int = 8,
) -> list[int]:
    pages = evidence_pages or page_bias_pages or []
    clean = sorted({int(page) for page in pages if isinstance(page, int) and page > 0})
    expanded: set[int] = set(clean)
    for page in clean:
        if page > 1:
            expanded.add(page - 1)
        expanded.add(page + 1)
    return sorted(expanded)[:max_pages]


def _visual_first(question: str, answer_format: str, hints: dict[str, Any] | None = None) -> bool:
    hints = hints or {}
    if hints.get("visual_first") is True:
        return True
    text = question.casefold()
    markers = (
        "chart",
        "table",
        "figure",
        "image",
        "diagram",
        "percentage",
        "percent",
        "how many",
        "greater",
        "higher",
        "lower",
        "increase",
        "decrease",
        "row",
        "column",
        "reference",
        "references",
    )
    return answer_format in {"Int", "Float"} or any(marker in text for marker in markers)


def _focused_query(
    question: str,
    rewritten: dict[str, Any] | None,
    skill_profile: dict[str, Any],
    missing_terms: list[str] | None = None,
) -> str:
    rewritten = rewritten or {}
    parts = [
        question,
        str(rewritten.get("text_query") or ""),
        str(rewritten.get("graph_query") or ""),
    ]
    parts.extend(str(item) for item in skill_profile.get("core_terms", []) if str(item).strip())
    parts.extend(str(item) for item in (missing_terms or []) if str(item).strip())
    if skill_profile.get("event_chain"):
        parts.append("event sequence cause action outcome")
    if skill_profile.get("procedural_detail"):
        parts.append("procedure workflow steps method tool system")
    return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _compact_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("transcript", "caption", "visual_caption", "frame_times", "query_text"):
        value = metadata.get(key)
        if value in (None, "", []):
            continue
        compact[key] = str(value)[:700] if isinstance(value, str) else value
    return compact


def _evidence_excerpt(item: dict[str, Any], *, max_chars: int = 1200) -> str:
    content = str(item.get("content") or item.get("text") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    transcript = str(metadata.get("transcript") or "").strip()
    if transcript and transcript not in content:
        content = f"{content}\nTranscript detail:\n{transcript}"
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content[:max_chars]


def _has_direct_answer_support(item: dict[str, Any]) -> bool:
    quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
    support = quality.get("support") if isinstance(quality.get("support"), dict) else {}
    return bool(support.get("direct_answer_support"))

def _rank_evidence_for_generation(
    question: str,
    rewritten: dict[str, Any] | None,
    fused: list[dict[str, Any]],
    *,
    retrieval_hints: dict[str, Any] | None = None,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return inspect_evidence_batch(
        fused,
        question=question,
        rewritten=rewritten,
        retrieval_hints=retrieval_hints,
        limit=limit,
        min_keep=max(4, min(6, limit)),
    )


class DecisionAgent:
    """Decision-only agent that produces expert task plans and final answers."""

    allowed_tools = ["read_evidence"]

    def __init__(
        self,
        provider: LLMProvider,
        generation_model: str = DEFAULT_MODELS.decision,
        enable_guided_routing: bool = False,
    ) -> None:
        if provider is None:
            raise ValueError("DecisionAgent requires an LLMProvider.")
        self._ids = count(1)
        self.provider = provider
        self.generation_model = generation_model
        self.enable_guided_routing = enable_guided_routing

    def _task_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids)}"

    def reset_task_ids(self) -> None:
        """Reset per-run task ids so repeated runs are deterministic."""

        self._ids = count(1)

    def plan(self, query: QueryContext) -> AgentPlan:
        rewritten = query.metadata.get("rewritten_data") if isinstance(query.metadata, dict) else None
        retrieval_hints = _retrieval_hints(query)
        if retrieval_hints and isinstance(rewritten, dict):
            rewritten = dict(rewritten)
            rewritten.setdefault("retrieval_hints", retrieval_hints)
        search_text = str(
            (rewritten or {}).get("text_query")
            or (rewritten or {}).get("expanded_query")
            or query.query_text
        )
        visual_query = str((rewritten or {}).get("visual_query") or search_text)
        graph_query = str((rewritten or {}).get("graph_query") or search_text)
        question_type = str((rewritten or {}).get("question_type") or "").lower()
        rewritten_params = dict(rewritten or {})
        answer_format = _hint_answer_format(query)
        evidence_pages = _hint_evidence_pages(query)
        page_bias_pages = _hint_page_bias_pages(query)
        if answer_format:
            rewritten_params.setdefault("answer_format", answer_format)
        if evidence_pages:
            rewritten_params.setdefault("evidence_pages", evidence_pages)
        if page_bias_pages:
            rewritten_params.setdefault("page_bias_pages", page_bias_pages)
        if retrieval_hints:
            rewritten_params.setdefault("retrieval_hints", retrieval_hints)
        visual_first = _visual_first(query.query_text, answer_format, retrieval_hints)
        answer_target = {
            "question": query.query_text,
            "answer_format": answer_format or retrieval_hints.get("answer_format"),
            "direct_answer_first": True,
        }
        visual_evidence_contract = {
            "must_use_source_image": True,
            "must_extract_visible_fields": True,
            "required_fields": [
                "label_value_pairs",
                "colors",
                "object_counts",
                "axes_or_headers",
                "support_cells",
                "answer_candidate",
            ],
            "prefer_calculable_fields": answer_format in {"Int", "Float"},
            "allow_ocr_only": False,
        }
        text_evidence_contract = {
            "prefer_exact_page_hits": True,
            "prefer_direct_answer_spans": True,
            "prefer_calculable_fields": answer_format in {"Int", "Float"},
        }
        graph_evidence_contract = {
            "prefer_explicit_relations": True,
            "use_for_followup_when_direct_evidence_is_sparse": True,
        }
        skill_profile = _routing_skill_profile(query.query_text, rewritten_params) if self.enable_guided_routing else {}
        if self.enable_guided_routing:
            query.metadata["routing_skill_profile"] = skill_profile

        tasks: list[RetrievalTask] = []
        expected_modalities: list[str] = []
        text = query.query_text.lower()
        complexity_signals = sum(
            [
                any(token in text for token in ("how many", "compare", "difference", "higher", "lower", "trend")),
                any(token in text for token in ("figure", "chart", "table", "image", "diagram", "layout")),
                any(token in text for token in ("why", "how", "cause", "relation", "relationship", "path", "evolve")),
                len(query.query_text.split()) > 18,
                bool(query.doc_profile and query.doc_profile.get("visual_heavy")),
            ]
        )
        graph_terms = (
            "compare",
            "contrast",
            "difference",
            "relationship",
            "relation",
            "interconnection",
            "interconnections",
            "network",
            "timeline",
            "trend",
            "patterns",
            "over time",
            "evolve",
            "evolved",
            "impact",
            "influence",
            "trade-off",
            "tradeoff",
        )
        causal_terms = ("why", "reason", "cause", "causes", "driven", "factors")
        wants_graph = bool(skill_profile.get("needs_graph")) or question_type in {"relation", "comparison", "temporal"} or any(
            token in text for token in graph_terms
        ) or (question_type == "cause" and any(token in text for token in causal_terms))
        wants_visual = bool(skill_profile.get("needs_visual_crosscheck")) or question_type in {"mechanism", "description", "comparison", "challenge"} or any(
            token in text for token in ("figure", "image", "chart", "table", "look", "show", "visible", "frame", "action", "scene", "describe")
        )

        if query.doc_query_vector is not None:
            expected_modalities.append("doc")
            query_profile = query.query_profile or infer_doc_query_profile(query.query_text)
            doc_route = choose_doc_planner_route(query.doc_profile or {}, query_profile)
            doc_ids = [query.source_doc_id] if query.source_doc_id else query.candidate_doc_ids
            common_doc_params = {
                "doc_root": query.doc_root,
                "doc_ids": doc_ids,
                "evidence_pages": evidence_pages,
                "answer_format": answer_format,
                "answer_target": answer_target,
            }
            if visual_first:
                modalities = preferred_doc_modalities(query_profile) or []
                if answer_format in {"Int", "Float"}:
                    modalities = list(dict.fromkeys(["table", "chart", "image"] + modalities))
                fanout_pages = _page_fanout_pages(
                    evidence_pages,
                    page_bias_pages,
                )
                if fanout_pages:
                    for page in fanout_pages:
                        page_query = f"{visual_query} page {page}"
                        tasks.append(
                            self._task(
                                agent="doc_visual_subagent",
                                corpus="doc",
                                tool_name="doc_visual_seek",
                                query=page_query,
                                params={
                                    "query_vector": query.doc_query_vector,
                                    "query_text": page_query,
                                    "rewritten_data": {
                                        **rewritten_params,
                                        "page_focus": page,
                                        "page_fanout": True,
                                    },
                                    "modalities": modalities,
                                    "top_k": max(4, min(query.top_k, 8)),
                                    "max_visual_inspections": int(retrieval_hints.get("max_visual_inspections") or 6),
                                    "evidence_contract": visual_evidence_contract,
                                    **common_doc_params,
                                    "evidence_pages": [page],
                                    "page_bias_pages": [page],
                                },
                                expected_evidence=[
                                    "page_scoped_visual_block",
                                    "image_path",
                                    "visual_caption",
                                    "nearby_chunk_ids",
                                ],
                                rationale=f"Page-scoped visual retrieval task for page {page}.",
                            )
                        )
                        tasks.append(
                            self._task(
                                agent="doc_text_subagent",
                                corpus="doc",
                                tool_name="doc_text_seek",
                                query=f"{search_text} page {page}",
                                params={
                                    "query_vector": query.doc_query_vector,
                                    "query_text": f"{search_text} page {page}",
                                    "rewritten_data": {
                                        **rewritten_params,
                                        "page_focus": page,
                                        "page_fanout": True,
                                    },
                                    "top_k": max(4, min(query.top_k, 6)),
                                    "min_score": None,
                                    "include_multimodal": True,
                                    "evidence_contract": text_evidence_contract,
                                    **common_doc_params,
                                    "evidence_pages": [page],
                                    "page_bias_pages": [page],
                                },
                                expected_evidence=[
                                    "page_scoped_chunk",
                                    "doc_id",
                                    "page",
                                    "text",
                                    "linked_visual_blocks",
                                ],
                                rationale=f"Page-scoped text/OCR retrieval task for page {page}.",
                            )
                        )
                    return AgentPlan(
                        query_context=query,
                        tasks=tasks,
                        rationale="Decision agent decomposed retrieval into page-scoped subtasks so evidence can be aggregated across pages.",
                        expected_modalities=list(dict.fromkeys(expected_modalities)),
                    )
                tasks.append(
                    self._task(
                        agent="doc_visual_subagent",
                        corpus="doc",
                        tool_name="doc_visual_seek",
                        query=visual_query,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": visual_query,
                            "rewritten_data": rewritten_params,
                            "modalities": modalities,
                            "top_k": max(query.top_k, 12),
                            "max_visual_inspections": int(retrieval_hints.get("max_visual_inspections") or 6),
                            "evidence_contract": visual_evidence_contract,
                            **common_doc_params,
                            "page_bias_pages": page_bias_pages,
                        },
                        expected_evidence=["visual_block_id", "image_path", "visual_caption", "nearby_chunk_ids"],
                        rationale="Chart, table, or numeric question requires visual or structured evidence first.",
                    )
                )
                tasks.append(
                    self._task(
                        agent="doc_text_subagent",
                        corpus="doc",
                        tool_name="doc_text_seek",
                        query=search_text,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": search_text,
                            "rewritten_data": rewritten_params,
                            "top_k": max(query.top_k, 10),
                            "min_score": None,
                            "include_multimodal": True,
                            "evidence_contract": text_evidence_contract,
                            **common_doc_params,
                            "page_bias_pages": page_bias_pages,
                        },
                        expected_evidence=["chunk_id", "doc_id", "page", "text", "linked_visual_blocks"],
                        rationale="Visual-first route still needs nearby OCR/text support for exact short answers.",
                    )
                )
            initial_doc_graph_enabled = (
                wants_graph or doc_route == "doc_graph_first"
            ) and not retrieval_hints.get("disable_initial_graph")
            if initial_doc_graph_enabled:
                tasks.append(
                    self._task(
                        agent="doc_graph_subagent",
                        corpus="doc",
                        tool_name="doc_graph_seek",
                        query=graph_query,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": graph_query,
                            "rewritten_data": rewritten_params,
                            "top_k_entities": max(8, min(16, query.top_k * 2)),
                            "top_k_chunks": max(query.top_k, 12),
                            **common_doc_params,
                            "graph_strategy": "hybrid",
                            "edge_type_filter": infer_doc_edge_type_filter(query.query_text),
                            "evidence_contract": graph_evidence_contract,
                        },
                        expected_evidence=["entity_path", "relations", "edge_semantics", "supporting_chunk_ids"],
                        rationale="Graph relation evidence is needed for explanation or sparse document coverage.",
                    )
                )
            if not visual_first and (wants_visual or doc_route == "doc_visual_first"):
                tasks.append(
                    self._task(
                        agent="doc_visual_subagent",
                        corpus="doc",
                        tool_name="doc_visual_seek",
                        query=visual_query,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": visual_query,
                            "rewritten_data": rewritten_params,
                            "modalities": preferred_doc_modalities(query_profile) or ["image", "chart", "table"],
                            "top_k": max(query.top_k, 10),
                            "max_visual_inspections": int(retrieval_hints.get("max_visual_inspections") or 6),
                            "evidence_contract": visual_evidence_contract,
                            **common_doc_params,
                            "page_bias_pages": page_bias_pages,
                        },
                        expected_evidence=["visual_block_id", "image_path", "visual_caption", "nearby_chunk_ids"],
                        rationale="Visual document evidence is required by the query or document profile.",
                    )
                )
            if not tasks or (not visual_first and doc_route == "doc_text_first"):
                tasks.append(
                    self._task(
                        agent="doc_text_subagent",
                        corpus="doc",
                        tool_name="doc_text_seek",
                        query=search_text,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": search_text,
                            "rewritten_data": rewritten_params,
                            "top_k": query.top_k,
                            "min_score": None,
                            "include_multimodal": True,
                            "evidence_contract": text_evidence_contract,
                            **common_doc_params,
                            "page_bias_pages": page_bias_pages,
                        },
                        expected_evidence=["chunk_id", "doc_id", "page", "text", "linked_visual_blocks"],
                        rationale="Text chunks are the primary document evidence candidate set.",
                    )
                )
            if (
                complexity_signals >= 3
                and query.doc_query_vector is not None
                and doc_route == "doc_text_first"
                and not retrieval_hints.get("disable_initial_graph")
            ):
                tasks.append(
                    self._task(
                        agent="doc_graph_subagent",
                        corpus="doc",
                        tool_name="doc_graph_seek",
                        query=graph_query,
                        params={
                            "query_vector": query.doc_query_vector,
                            "query_text": graph_query,
                            "rewritten_data": rewritten_params,
                            "top_k_entities": max(6, query.top_k),
                            "top_k_chunks": max(query.top_k, 8),
                            **common_doc_params,
                            "graph_strategy": "hybrid",
                            "edge_type_filter": infer_doc_edge_type_filter(query.query_text),
                            "evidence_contract": graph_evidence_contract,
                        },
                        expected_evidence=["entity_path", "relations", "edge_semantics", "supporting_chunk_ids"],
                        rationale="Complex document queries benefit from a second inspector to verify paths and relation support.",
                    )
                )

        if query.video_query_vector is not None or query.visual_query_vector is not None:
            expected_modalities.append("video")
            video_text_top_k = int(retrieval_hints.get("video_text_top_k") or max(query.top_k, 24))
            video_visual_top_k = int(retrieval_hints.get("video_visual_top_k") or max(query.top_k, 16))
            use_mapped_segment_details = not bool(retrieval_hints.get("disable_mapped_segment_details"))
            use_exact_detail_lexical = not bool(retrieval_hints.get("disable_exact_detail_lexical"))
            if query.video_query_vector is not None:
                tasks.append(
                    self._task(
                        agent="video_text_subagent",
                        corpus="video",
                        tool_name="video_text_seek",
                        query=search_text,
                        params={
                            "query_vector": query.video_query_vector,
                            "query_text": search_text,
                            "rewritten_data": rewritten_params,
                            "top_k": video_text_top_k,
                            "min_score": None,
                            "video_root": query.video_root,
                            "include_mapped_segment_details": use_mapped_segment_details,
                            "exact_detail_lexical": use_exact_detail_lexical,
                        },
                        expected_evidence=["segment_id", "video_id", "start_time", "end_time", "text", "text_type"],
                        rationale="Video text is the first evidence layer for ASR, subtitles, captions, or OCR.",
                    )
                )
            if query.visual_query_vector is not None and not retrieval_hints.get("disable_video_visual"):
                tasks.append(
                    self._task(
                        agent="video_visual_subagent",
                        corpus="video",
                        tool_name="video_visual_seek",
                        query=visual_query,
                        params={
                            "query_vector": query.visual_query_vector,
                            "query_text": visual_query,
                            "rewritten_data": rewritten_params,
                            "top_k": video_visual_top_k,
                            "min_score": None,
                            "video_root": query.video_root,
                        },
                        expected_evidence=["frame_id", "segment_id", "timestamp", "frame_path", "visual_caption"],
                        rationale="Visual frame evidence is needed for scene, action, object, or visibility details.",
                    )
                )
            video_graph_needed = wants_graph and (
                any(token in text for token in graph_terms)
                or sum(token in text for token in causal_terms) >= 2
                or len(query.query_text.split()) > 16
                or bool(skill_profile.get("event_chain"))
                or bool(skill_profile.get("needs_graph"))
            )
            if video_graph_needed and query.video_query_vector is not None and not retrieval_hints.get("disable_video_graph"):
                tasks.append(
                    self._task(
                        agent="video_graph_subagent",
                        corpus="video",
                        tool_name="video_graph_seek",
                        query=graph_query,
                        params={
                            "query_vector": query.video_query_vector,
                            "query_text": graph_query,
                            "rewritten_data": rewritten_params,
                            "top_k": max(query.top_k, 12) if self.enable_guided_routing else max(query.top_k, 10),
                            "top_k_entities": max(16, query.top_k + 4) if self.enable_guided_routing else max(14, query.top_k + 2),
                            "top_k_chunks": max(12, query.top_k + 2) if self.enable_guided_routing else max(10, query.top_k),
                            "video_root": query.video_root,
                        },
                        expected_evidence=["entity_path", "event_path", "relations", "edge_semantics"],
                        rationale="Graph evidence is needed for causal, relational, or multi-hop video reasoning.",
                    )
                )

        if not tasks and query.doc_query_vector is None and query.video_query_vector is None:
            tasks = []

        return AgentPlan(
            query_context=query,
            tasks=tasks,
            rationale="Decision agent planned only final-protocol expert tasks.",
            expected_modalities=list(dict.fromkeys(expected_modalities)),
        )

    def _task(
        self,
        *,
        agent: str,
        corpus: str,
        tool_name: str,
        query: str,
        params: dict[str, Any],
        expected_evidence: list[str],
        rationale: str,
    ) -> RetrievalTask:
        return RetrievalTask(
            id=self._task_id(agent.replace("_subagent", "")),
            agent=agent,  # type: ignore[arg-type]
            corpus=corpus,  # type: ignore[arg-type]
            intent=tool_name,  # type: ignore[arg-type]
            tool_name=tool_name,
            params=params,
            rationale=rationale,
            query=query,
            allowed_tools=allowed_tools_for_contract(agent),
            expected_evidence=expected_evidence,
            stop_condition="Write a filtered evidence report with confidence, gaps, and traceable anchors.",
        )

    def reflect(self, plan: AgentPlan, results: list[SubagentResult]) -> ReflectionResult:
        query = plan.query_context
        existing_agents = {result.task.agent for result in results}
        evidence = [item for result in results for item in result.evidence]
        new_tasks: list[RetrievalTask] = []
        lower_query = query.query_text.lower()
        rewritten = query.metadata.get("rewritten_data", {}) if isinstance(query.metadata, dict) else {}
        skill_profile = _routing_skill_profile(query.query_text, rewritten) if self.enable_guided_routing else {}
        if self.enable_guided_routing:
            query.metadata["routing_skill_profile"] = skill_profile
        audit = self._audit_evidence_quality(query.query_text, rewritten, results)
        query.metadata["reflection_quality_audit"] = audit
        evidence_audit = audit_evidence(
            question=query.query_text,
            rewritten=rewritten,
            results=results,
            expected_modalities=plan.expected_modalities,
            retrieval_hints=_retrieval_hints(query),
        )
        query.metadata["evidence_audit"] = evidence_audit
        needs_quality_followup = bool(
            audit.get("missing_required_tokens")
            or audit.get("off_topic_count")
            or audit.get("direct_support_count", 0) < 1
            or evidence_audit.get("missing_obligations")
            or evidence_audit.get("cross_modal_conflicts")
        )

        doc_text_mentions_visual = any(
            result.task.agent == "doc_text_subagent"
            and any(
                token in str(item).lower()
                for item in result.evidence
                for token in ("figure", "image", "chart", "table", "formula", "screenshot")
            )
            for result in results
        )
        if doc_text_mentions_visual and "doc_visual_subagent" not in existing_agents and query.doc_query_vector is not None:
            new_tasks.append(
                self._task(
                    agent="doc_visual_subagent",
                    corpus="doc",
                    tool_name="doc_visual_seek",
                    query=query.query_text,
                    params={"query_vector": query.doc_query_vector, "top_k": query.top_k, "doc_root": query.doc_root},
                    expected_evidence=["visual_block_id", "image_path", "visual_caption"],
                    rationale="Text evidence references document visual blocks that need visual follow-up.",
                )
            )

        needs_video_visual = bool(skill_profile.get("needs_visual_crosscheck")) or any(token in lower_query for token in ("action", "doing", "look", "visible", "scene", "frame"))
        if needs_video_visual and "video_visual_subagent" not in existing_agents and query.visual_query_vector is not None:
            visual_followup_query = (
                _focused_query(query.query_text, rewritten, skill_profile, audit.get("missing_required_tokens"))
                if self.enable_guided_routing
                else query.query_text
            )
            new_tasks.append(
                self._task(
                    agent="video_visual_subagent",
                    corpus="video",
                    tool_name="video_visual_seek",
                    query=visual_followup_query,
                    params={
                        "query_vector": query.visual_query_vector,
                        "query_text": visual_followup_query,
                        **({"rewritten_data": rewritten} if self.enable_guided_routing else {}),
                        "top_k": max(query.top_k, 16),
                        "video_root": query.video_root,
                    },
                    expected_evidence=["frame_id", "timestamp", "frame_path", "visual_caption"],
                    rationale="Video text is insufficient for action or visual detail.",
                )
            )

        needs_graph = (
            bool(skill_profile.get("needs_graph"))
            or any(token in lower_query for token in ("why", "how", "cause", "relation", "relationship", "path", "evolve"))
            or (
                needs_quality_followup
                and audit["direct_support_count"] < 2
            )
        )
        conflict_agents = {
            str(task.get("agent"))
            for task in evidence_audit.get("followup_tasks", [])
            if isinstance(task, dict)
            and str(task.get("agent")) in {
                "doc_text_subagent",
                "doc_visual_subagent",
                "doc_graph_subagent",
                "video_text_subagent",
                "video_visual_subagent",
                "video_graph_subagent",
            }
        }
        if (
            "doc_visual_subagent" in conflict_agents
            and "doc_visual_subagent" not in existing_agents
            and query.doc_query_vector is not None
        ):
            new_tasks.append(
                self._task(
                    agent="doc_visual_subagent",
                    corpus="doc",
                    tool_name="doc_visual_seek",
                    query=query.query_text,
                    params={
                        "query_vector": query.doc_query_vector,
                        "query_text": query.query_text,
                        "top_k": max(query.top_k, 12),
                        "doc_root": query.doc_root,
                        "doc_ids": [query.source_doc_id] if query.source_doc_id else query.candidate_doc_ids,
                        "rewritten_data": rewritten,
                    },
                    expected_evidence=["visual_block_id", "image_path", "visual_caption", "conflict_resolution"],
                    rationale="Evidence audit requested visual follow-up to resolve a gap or cross-modal conflict.",
                )
            )
        if needs_graph and "doc" in plan.expected_modalities and "doc_graph_subagent" not in existing_agents and query.doc_query_vector is not None:
            doc_ids = [query.source_doc_id] if query.source_doc_id else query.candidate_doc_ids
            new_tasks.append(
                self._task(
                    agent="doc_graph_subagent",
                    corpus="doc",
                    tool_name="doc_graph_seek",
                    query=query.query_text,
                    params={
                        "query_vector": query.doc_query_vector,
                        "query_text": query.query_text,
                        "top_k_entities": min(6, query.top_k),
                        "top_k_chunks": query.top_k,
                        "doc_root": query.doc_root,
                        "doc_ids": doc_ids,
                        "evidence_pages": _hint_evidence_pages(query),
                    },
                    expected_evidence=["entity_path", "relations", "edge_semantics"],
                    rationale="Question asks for relation, cause, path, or evolution.",
                )
            )
        if needs_graph and "video" in plan.expected_modalities and "video_graph_subagent" not in existing_agents and query.video_query_vector is not None:
            graph_followup_query = (
                _focused_query(query.query_text, rewritten, skill_profile, audit.get("missing_required_tokens"))
                if self.enable_guided_routing
                else query.query_text
            )
            new_tasks.append(
                self._task(
                    agent="video_graph_subagent",
                    corpus="video",
                    tool_name="video_graph_seek",
                    query=graph_followup_query,
                    params={
                        "query_vector": query.video_query_vector,
                        "query_text": graph_followup_query,
                        **({"rewritten_data": rewritten} if self.enable_guided_routing else {}),
                        "top_k": max(query.top_k, 12),
                        "top_k_entities": max(14, query.top_k + 2) if self.enable_guided_routing else max(12, query.top_k),
                        "top_k_chunks": max(10, query.top_k) if self.enable_guided_routing else max(8, query.top_k),
                        "video_root": query.video_root,
                    },
                    expected_evidence=["entity_path", "event_path", "relations", "edge_semantics"],
                    rationale="Question asks for relation, cause, path, or evolution.",
                )
            )

        entity_missing = [token for token in audit["entity_tokens"] if token not in set(audit["covered_tokens"])]
        action_missing = [token for token in audit["action_tokens"] if token not in set(audit["covered_tokens"])]
        severe_missing = entity_missing or (
            len(audit["action_tokens"]) <= 2 and bool(action_missing) and audit["relevant_evidence_count"] < 6
        )
        needs_quality_followup = (
            audit["relevant_evidence_count"] < 4
            or bool(severe_missing)
            or audit["off_topic_count"] > max(2, audit["relevant_evidence_count"])
            or bool(evidence_audit.get("missing_obligations"))
            or bool(evidence_audit.get("cross_modal_conflicts"))
            or (
                self.enable_guided_routing
                and bool(skill_profile.get("factual_detail") or skill_profile.get("procedural_detail"))
                and audit["direct_support_count"] < 4
            )
            or (
                self.enable_guided_routing
                and bool(skill_profile.get("event_chain"))
                and audit["direct_support_count"] < 6
            )
        )
        if needs_quality_followup and query.video_query_vector is not None:
            if self.enable_guided_routing:
                focused_query = _focused_query(query.query_text, rewritten, skill_profile, audit.get("missing_required_tokens"))
                planned_agents = existing_agents | {task.agent for task in new_tasks}
            else:
                focus_terms = " ".join(audit["missing_required_tokens"] or audit["required_tokens"] or [query.query_text])
                focused_query = f"{query.query_text} {focus_terms}".strip()
                planned_agents = existing_agents
            text_quality = audit["agent_quality"].get("video_text_subagent", {})
            if (
                "video_text_subagent" not in planned_agents
                or text_quality.get("relevant", 0) < 2
                or (
                    self.enable_guided_routing
                    and bool(skill_profile.get("needs_text_precision"))
                    and text_quality.get("direct_support", 0) < 4
                )
            ):
                new_tasks.append(
                    self._task(
                        agent="video_text_subagent",
                        corpus="video",
                        tool_name="video_text_seek",
                        query=focused_query,
                        params={
                            "query_vector": query.video_query_vector,
                            "query_text": focused_query,
                            "rewritten_data": rewritten,
                            "top_k": max(query.top_k, 24),
                            "min_score": None,
                            "video_root": query.video_root,
                            "include_mapped_segment_details": True,
                            "exact_detail_lexical": True,
                        },
                        expected_evidence=["segment_id", "video_id", "start_time", "end_time", "text", "text_type"],
                        rationale="Reflection audit found weak entity/action coverage; rerun text retrieval with focused core terms.",
                    )
                )
            planned_agents = existing_agents | {task.agent for task in new_tasks}
            if (
                query.visual_query_vector is not None
                and "video_visual_subagent" not in planned_agents
                and (
                    audit["relevant_evidence_count"] < 4
                    or (self.enable_guided_routing and bool(skill_profile.get("needs_visual_crosscheck")))
                )
            ):
                new_tasks.append(
                    self._task(
                        agent="video_visual_subagent",
                        corpus="video",
                        tool_name="video_visual_seek",
                        query=focused_query,
                        params={
                            "query_vector": query.visual_query_vector,
                            "query_text": focused_query,
                            "rewritten_data": rewritten,
                            "top_k": max(query.top_k, 16),
                            "min_score": None,
                            "video_root": query.video_root,
                        },
                        expected_evidence=["frame_id", "segment_id", "timestamp", "frame_path", "visual_caption"],
                        rationale="Reflection audit found insufficient relevant visual evidence after local inspection.",
                    )
                )
            planned_agents = existing_agents | {task.agent for task in new_tasks}
            graph_quality = audit["agent_quality"].get("video_graph_subagent", {})
            if (
                self.enable_guided_routing
                and bool(skill_profile.get("needs_graph"))
                and "video_graph_subagent" not in planned_agents
                and graph_quality.get("direct_support", 0) < 3
            ):
                new_tasks.append(
                    self._task(
                        agent="video_graph_subagent",
                        corpus="video",
                        tool_name="video_graph_seek",
                        query=focused_query,
                        params={
                            "query_vector": query.video_query_vector,
                            "query_text": focused_query,
                            "rewritten_data": rewritten,
                            "top_k": max(query.top_k, 14),
                            "top_k_entities": max(18, query.top_k + 6),
                            "top_k_chunks": max(14, query.top_k + 2),
                            "video_root": query.video_root,
                        },
                        expected_evidence=["entity_path", "event_path", "relations", "edge_semantics"],
                        rationale="Guided routing skill requested graph/event-chain backfill after direct support remained weak.",
                    )
                )

        if evidence and not new_tasks and not needs_quality_followup:
            return ReflectionResult(
                sufficient=True,
                reason=(
                    "Evidence quality audit passed: "
                    f"relevant={audit['relevant_evidence_count']}, "
                    f"off_topic={audit['off_topic_count']}, "
                    f"missing={severe_missing}."
                ),
                new_tasks=[],
            )
        if new_tasks:
            return ReflectionResult(
                sufficient=False,
                reason=(
                    "Evidence quality audit requires follow-up: "
                    f"relevant={audit['relevant_evidence_count']}, "
                    f"off_topic={audit['off_topic_count']}, "
                    f"missing={severe_missing}."
                ),
                new_tasks=new_tasks,
            )
        return ReflectionResult(
            sufficient=bool(evidence) and not needs_quality_followup,
            reason=(
                "No further final-protocol tasks are available after quality audit: "
                f"relevant={audit['relevant_evidence_count']}, "
                f"missing={severe_missing}."
            ),
            new_tasks=[],
        )

    def _audit_evidence_quality(
        self,
        question: str,
        rewritten: dict[str, Any] | None,
        results: list[SubagentResult],
    ) -> dict[str, Any]:
        profile = query_quality_profile(question, rewritten)
        required = set(profile["required_tokens"])
        covered: set[str] = set()
        agent_quality: dict[str, dict[str, Any]] = {}
        relevant_count = 0
        off_topic_count = 0
        direct_support_count = 0
        background_only_count = 0
        for result in results:
            relevant = 0
            off_topic = 0
            direct_support = 0
            background_only = 0
            raw = len(result.evidence)
            local_dropped = 0
            for item in result.evidence:
                quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
                if not quality:
                    hints = (
                        rewritten.get("retrieval_hints")
                        if isinstance(rewritten, dict) and isinstance(rewritten.get("retrieval_hints"), dict)
                        else None
                    )
                    inspected, _ = inspect_evidence_batch(
                        [item],
                        question=question,
                        rewritten=rewritten,
                        retrieval_hints=hints,
                        limit=1,
                        min_keep=1,
                    )
                    quality = inspected[0].get("quality", {}) if inspected else {}
                if quality.get("is_relevant"):
                    relevant += 1
                    relevant_count += 1
                if quality.get("off_topic"):
                    off_topic += 1
                    off_topic_count += 1
                support = quality.get("support") if isinstance(quality.get("support"), dict) else {}
                if support.get("direct_answer_support"):
                    direct_support += 1
                    direct_support_count += 1
                if support.get("background_only"):
                    background_only += 1
                    background_only_count += 1
                hits = quality.get("hits") if isinstance(quality.get("hits"), dict) else {}
                for values in hits.values():
                    if isinstance(values, list):
                        covered.update(str(value) for value in values)
            data = result.data if isinstance(result.data, dict) else {}
            report = data.get("quality_report") if isinstance(data.get("quality_report"), dict) else {}
            local_dropped = int(report.get("dropped_count", 0) or 0)
            agent_quality[result.task.agent] = {
                "raw": int(data.get("raw_evidence_count", raw) or raw),
                "kept": raw,
                "relevant": relevant,
                "off_topic": off_topic,
                "direct_support": direct_support,
                "background_only": background_only,
                "local_dropped": local_dropped,
                "marginal_relevance_rate": round(relevant / raw, 4) if raw else 0.0,
            }
        return {
            **profile,
            "covered_tokens": sorted(covered),
            "missing_required_tokens": sorted(required - covered),
            "relevant_evidence_count": relevant_count,
            "off_topic_count": off_topic_count,
            "direct_support_count": direct_support_count,
            "background_only_count": background_only_count,
            "agent_quality": agent_quality,
        }

    def generate(
        self,
        plan: AgentPlan,
        results: list[SubagentResult],
        fused: list[dict[str, Any]],
        *,
        generation_context: dict[str, Any] | None = None,
    ) -> str:
        board = (generation_context or {}).get("evidence_board") if generation_context else None
        reports = board.get("reports", []) if isinstance(board, dict) else []
        if not reports and not fused:
            return "没有找到足够的结构化证据来回答该问题。"
        lines = [f"### {plan.query_context.query_text}", ""]
        for report in reports:
            anchor = report.get("tool_used", "evidence")
            summary = str(report.get("summary", "")).strip()
            confidence = report.get("confidence", 0.0)
            if summary:
                lines.append(f"- [{anchor}] {summary} (confidence={confidence})")
        if fused:
            lines.append("")
            lines.append("Evidence anchors:")
            for item in fused[:5]:
                source = item.get("id") or item.get("source_id") or item.get("source_type") or "unknown"
                content = str(item.get("content") or item.get("text") or "")[:240]
                lines.append(f"- {source}: {content}")
        gaps = board.get("gaps", []) if isinstance(board, dict) else []
        if gaps:
            lines.append("")
            lines.append("Uncertainty:")
            for gap in gaps[:3]:
                lines.append(f"- {gap.get('description', gap)}")
        return "\n".join(lines).strip()

    async def generate_async(
        self,
        plan: AgentPlan,
        results: list[SubagentResult],
        fused: list[dict[str, Any]],
        *,
        generation_context: dict[str, Any] | None = None,
    ) -> str:
        board = (generation_context or {}).get("evidence_board") if generation_context else None
        if not fused and not (isinstance(board, dict) and board.get("reports")):
            return "没有找到足够的结构化证据来回答该问题。"
        return await self._generate_with_llm(plan, fused, board)

    async def _generate_with_llm(
        self,
        plan: AgentPlan,
        fused: list[dict[str, Any]],
        board: dict[str, Any] | None,
    ) -> str:
        rewritten = (
            plan.query_context.metadata.get("rewritten_data", {})
            if isinstance(plan.query_context.metadata, dict)
            else {}
        )
        retrieval_hints = _retrieval_hints(plan.query_context)
        if retrieval_hints and isinstance(rewritten, dict):
            rewritten = dict(rewritten)
            rewritten.setdefault("retrieval_hints", retrieval_hints)
        ranker = globals().get("_rank_evidence_for_generation")
        if callable(ranker):
            generation_evidence, evidence_filter = ranker(
                plan.query_context.query_text,
                rewritten,
                fused,
                retrieval_hints=retrieval_hints,
                limit=16,
            )
        else:
            generation_evidence, evidence_filter = inspect_evidence_batch(
                fused,
                question=plan.query_context.query_text,
                rewritten=rewritten,
                retrieval_hints=retrieval_hints,
                limit=16,
                min_keep=4,
            )
        if not generation_evidence:
            return self._fallback_summary(
                plan,
                fused,
                board,
                reason="No direct answer support was found in the filtered evidence.",
            )
        if not any(_has_direct_answer_support(item) for item in generation_evidence):
            return self._fallback_summary(
                plan,
                generation_evidence,
                board,
                reason="No direct answer support was found in the filtered evidence.",
            )
        prompt = {
            "question": plan.query_context.query_text,
            "rewritten_query": rewritten,
            "evidence_filter": evidence_filter,
            "answer_format": retrieval_hints.get("answer_format"),
            "filtered_evidence": [
                {
                    "id": item.get("id"),
                    "source_type": item.get("source_type"),
                    "modality": item.get("modality"),
                    "score": item.get("fused_score", item.get("score", 0.0)),
                    "generation_relevance": item.get("generation_relevance"),
                    "generation_relevance_hits": item.get("generation_relevance_hits"),
                    "support_profile": (
                        item.get("quality", {}).get("support")
                        if isinstance(item.get("quality"), dict)
                        else None
                    ),
                    "content": _evidence_excerpt(item, max_chars=1500),
                    "locator": item.get("locator"),
                    "metadata": _compact_metadata(item.get("metadata")),
                    "vlm_visual_inspection": (
                        item.get("metadata", {}).get("vlm_visual_inspection")
                        if isinstance(item.get("metadata"), dict)
                        else None
                    ),
                }
                for item in generation_evidence
            ],
            "gaps": board.get("gaps", []) if isinstance(board, dict) else [],
            "evidence_audit": (
                plan.query_context.metadata.get("evidence_audit")
                if isinstance(plan.query_context.metadata, dict)
                else None
            ),
        }
        wants_structured_short_answer = bool(retrieval_hints.get("answer_format"))
        system_prompt, max_tokens = generation_system_prompt(
            structured_short_answer=wants_structured_short_answer,
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ]

        try:
            response = await self.provider.chat(
                messages=messages,
                model=self.generation_model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            content = response.content or ""
        except Exception:
            return self._fallback_summary(plan, fused, board)
        try:
            payload = json.loads(content)
        except Exception:
            answer = content.strip()
            return answer if answer else self._fallback_summary(plan, fused, board)
        raw_answer = payload.get("answer") or payload.get("content") or ""
        answer = _humanize_generated_value(raw_answer)
        if not answer:
            return self._fallback_summary(plan, fused, board)
        confidence = payload.get("confidence")
        uncertainty = str(payload.get("uncertainty") or "").strip()
        if uncertainty and confidence not in {None, "", 1, 1.0, "1"}:
            answer = f"{answer}\n\nUncertainty: {uncertainty}"
        return answer

    def _fallback_summary(
        self,
        plan: AgentPlan,
        fused: list[dict[str, Any]],
        board: dict[str, Any] | None,
        *,
        reason: str | None = None,
    ) -> str:
        lines = [f"### {plan.query_context.query_text}", ""]
        if reason:
            lines.append(reason)
            lines.append("")
        for item in fused[:5]:
            source = item.get("id") or item.get("source_id") or item.get("source_type") or "unknown"
            content = str(item.get("content") or item.get("text") or "")[:240]
            lines.append(f"- {source}: {content}")
        gaps = board.get("gaps", []) if isinstance(board, dict) else []
        if gaps:
            lines.append("")
            lines.append("Uncertainty:")
            for gap in gaps[:3]:
                lines.append(f"- {gap.get('description', gap)}")
        return "\n".join(lines).strip()
