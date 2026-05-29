"""Multi-agent orchestrator for agentic multimodal RAG."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any

from agentic_mm_rag.agent.decision import DecisionAgent
from agentic_mm_rag.orchestrator.evidence.board import EvidenceBoard
from agentic_mm_rag.orchestrator.evidence.io import EvidenceBoardWriter, ReadEvidenceTool, WriteEvidenceTool
from agentic_mm_rag.orchestrator.evidence.pool import RefreshableEvidencePool
from agentic_mm_rag.agent.subagent import (
    DocGraphSubagent,
    DocTextSubagent,
    DocVisualSubagent,
    ExpertSubagent,
    VideoGraphSubagent,
    VideoTextSubagent,
    VideoVisualSubagent,
)
from agentic_mm_rag.orchestrator.types import (
    AgentPlan,
    OrchestrationResult,
    QueryContext,
    ReflectionResult,
    RetrievalTask,
    SubagentResult,
)
from agentic_mm_rag.agent.runner import AgentRunner
from agentic_mm_rag.config import DEFAULT_MODELS, ModelDefaults
from agentic_mm_rag.providers import LLMProvider
from agentic_mm_rag.orchestrator.tools import build_registry_bundle
from agentic_mm_rag.tools.runtime.scoring import fuse_evidence_items
from agentic_mm_rag.tools.registry import ToolRegistry


@dataclass(slots=True)
class OrchestratorContext:
    """Per-query runtime context managed by the orchestrator."""

    query: QueryContext
    plan: AgentPlan | None = None
    results: list[SubagentResult] = field(default_factory=list)
    fused_evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Orchestrator:
    """System scheduler: tools, context, dynamic subagents, reflection, generation."""

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        decision_agent: DecisionAgent | None = None,
        evidence_board: EvidenceBoard | None = None,
        evidence_pool: RefreshableEvidencePool | None = None,
        *,
        provider: LLMProvider,
        model_defaults: ModelDefaults | None = None,
        decision_model: str | None = None,
        text_model: str | None = None,
        visual_model: str | None = None,
        graph_model: str | None = None,
        max_reflection_rounds: int = 1,
        enable_guided_routing: bool = False,
        reuse_evidence_pool: bool = False,
        **legacy_kwargs: Any,
    ) -> None:
        if provider is None:
            raise ValueError("Orchestrator requires an LLMProvider; model sessions are always enabled.")
        models = model_defaults or DEFAULT_MODELS
        decision_model = decision_model or models.decision
        text_model = text_model or models.text_expert
        visual_model = visual_model or models.visual_expert
        graph_model = graph_model or models.graph_expert
        self.decision_agent = decision_agent or DecisionAgent(
            provider=provider,
            generation_model=decision_model,
            enable_guided_routing=enable_guided_routing,
        )
        self.evidence_board = evidence_board or EvidenceBoard()
        self.evidence_pool = evidence_pool or RefreshableEvidencePool()
        self.evidence_writer = EvidenceBoardWriter(self.evidence_board)
        self.tools = tools or build_registry_bundle(
            evidence_board=self.evidence_board,
            evidence_writer=self.evidence_writer,
        ).internal_tools
        self.provider = provider
        self.runner = AgentRunner(provider)
        
        self.decision_model = decision_model
        self.text_model = text_model
        self.visual_model = visual_model
        self.graph_model = graph_model
        
        self.subagent_execution = "model_session"
        self.max_reflection_rounds = max_reflection_rounds
        self.reuse_evidence_pool = reuse_evidence_pool
        self.trace: list[dict[str, Any]] = []
        self._run_lock = asyncio.Lock()

    @contextmanager
    def _temporary_evidence_state(self):
        """Use an isolated evidence board/writer for one orchestration run."""

        original_board = self.evidence_board
        original_writer = self.evidence_writer
        original_tools = self.tools
        board = EvidenceBoard(
            provider=getattr(original_board, "provider", None),
            enable_llm_consolidation=bool(getattr(original_board, "enable_llm_consolidation", False)),
        )
        board.model = getattr(original_board, "model", board.model)
        writer = EvidenceBoardWriter(board)
        tools = ToolRegistry()
        for name in original_tools.tool_names:
            if name in {"read_evidence", "write_evidence"}:
                continue
            tool = original_tools.get(name)
            if tool is not None:
                tools.register(tool)
        tools.register(ReadEvidenceTool(board))
        tools.register(WriteEvidenceTool(writer))
        self.evidence_board = board
        self.evidence_writer = writer
        self.tools = tools
        try:
            yield
        finally:
            self.evidence_board = original_board
            self.evidence_writer = original_writer
            self.tools = original_tools

    def available_tools(self) -> list[str]:
        return self.tools.names

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.tools.get_definitions()

    def public_tool_definitions(self) -> list[dict[str, Any]]:
        return build_registry_bundle().public_tools.get_definitions()

    def register_tool_registry(self, registry: ToolRegistry) -> None:
        for name in self.tools.tool_names:
            tool = self.tools.get(name)
            if tool is not None:
                registry.register(tool)

    async def run_query(
        self,
        query_text: str,
        *,
        doc_query_vector: list[float] | None = None,
        video_query_vector: list[float] | None = None,
        visual_query_vector: list[float] | None = None,
        top_k: int = 12,
        doc_root: str | None = None,
        video_root: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        query = QueryContext(
            query_text=query_text,
            doc_query_vector=doc_query_vector,
            video_query_vector=video_query_vector,
            visual_query_vector=visual_query_vector,
            top_k=top_k,
            doc_root=doc_root,
            video_root=video_root,
            metadata=dict(metadata or {}),
        )
        return await self.run(dataclasses.replace(query))

    async def run(self, query: QueryContext) -> OrchestrationResult:
        async with self._run_lock:
            with self._temporary_evidence_state():
                return await self._run_multi_agent(dataclasses.replace(query, metadata=dict(query.metadata or {})))

    async def refresh_evidence_pool(
        self,
        query: QueryContext,
        *,
        mark_existing_stale: bool = True,
    ) -> dict[str, Any]:
        """Refresh pool candidates for a query by rerunning the current retrieval plan."""

        async with self._run_lock:
            query = dataclasses.replace(query, metadata=dict(query.metadata or {}))
            with self._temporary_evidence_state():
                if mark_existing_stale:
                    query_terms = set(query.query_text.casefold().split())
                    if query_terms:
                        self.evidence_pool.mark_stale(
                            lambda item: bool(
                                query_terms
                                & set(str(item.evidence.get("content") or "").casefold().split())
                            ),
                            reason="explicit_refresh",
                        )
                    else:
                        self.evidence_pool.mark_stale(reason="explicit_refresh")
                reset_task_ids = getattr(self.decision_agent, "reset_task_ids", None)
                if callable(reset_task_ids):
                    reset_task_ids()
                plan = self.decision_agent.plan(query)
                results = await self._run_tasks(plan.tasks)
                await self._update_evidence_state(query, results)
                return {
                    "plan": plan.to_dict(),
                    "results": [result.to_dict() for result in results],
                    "pool": self.evidence_pool.snapshot(include_evidence=False),
                }

    async def _run_multi_agent(
        self,
        query: QueryContext,
    ) -> OrchestrationResult:
        self.trace = []
        reset_task_ids = getattr(self.decision_agent, "reset_task_ids", None)
        if callable(reset_task_ids):
            reset_task_ids()
        if "rewritten_data" not in (query.metadata or {}):
            ctype = "video" if query.video_query_vector is not None else "doc"
            rewritten = await self._rewrite_query(
                query.query_text,
                context_type=ctype,
                retrieval_hints=(
                    query.metadata.get("retrieval_hints")
                    if isinstance(query.metadata, dict)
                    else None
                ),
            )
            query.metadata["rewritten_data"] = {
                "text_query": rewritten.get("expanded_query", query.query_text),
                "visual_query": rewritten.get("visual_query", rewritten.get("expanded_query", query.query_text)),
                "graph_query": rewritten.get("graph_query", rewritten.get("expanded_query", query.query_text)),
                "subquestions": rewritten.get("subquestions", [query.query_text]),
                "question_type": rewritten.get("question_type"),
                "entities": rewritten.get("entities", []),
                "actions": rewritten.get("actions", []),
                "textual_keywords": rewritten.get("textual_keywords", []),
                "visual_anchors": rewritten.get("visual_anchors", []),
            }
            self.trace.append(
                {
                    "step": "query_rewrite",
                    "context_type": ctype,
                    "result": dict(query.metadata["rewritten_data"]),
                }
            )

        context = OrchestratorContext(query=query)
        plan = self.decision_agent.plan(query)
        context.plan = plan
        self.trace.append(
            {
                "step": "plan",
                "rationale": plan.rationale,
                "expected_modalities": list(plan.expected_modalities),
                "tasks": [task.to_dict() for task in plan.tasks],
            }
        )

        initial_results = await self._run_tasks(plan.tasks)
        context.results.extend(initial_results)
        await self._update_evidence_state(query, initial_results)
        self.trace.append(
            {
                "step": "initial_results",
                "results": [result.to_dict() for result in initial_results],
                "evidence_snapshot": self.evidence_board.state_snapshot(),
            }
        )

        reflection = self.decision_agent.reflect(plan, context.results)
        self.trace.append(
            {
                "step": "reflection",
                "sufficient": reflection.sufficient,
                "reason": reflection.reason,
                "quality_audit": query.metadata.get("reflection_quality_audit"),
                "evidence_audit": query.metadata.get("evidence_audit"),
                "new_tasks": [task.to_dict() for task in reflection.new_tasks],
            }
        )
        for _round in range(self.max_reflection_rounds):
            if reflection.sufficient or not reflection.new_tasks:
                break
            followup_results = await self._run_tasks(reflection.new_tasks)
            context.results.extend(followup_results)
            await self._update_evidence_state(query, followup_results)
            reflection = self.decision_agent.reflect(plan, context.results)
            self.trace.append(
                {
                    "step": "followup_results",
                    "results": [result.to_dict() for result in followup_results],
                    "next_reflection": {
                        "sufficient": reflection.sufficient,
                        "reason": reflection.reason,
                        "quality_audit": query.metadata.get("reflection_quality_audit"),
                        "evidence_audit": query.metadata.get("evidence_audit"),
                        "new_tasks": [task.to_dict() for task in reflection.new_tasks],
                    },
                    "evidence_snapshot": self.evidence_board.state_snapshot(),
                }
            )

        fused = await self._fuse_results(
            context.results,
            top_k=query.top_k,
            query_text=query.query_text,
            query=query,
        )
        context.fused_evidence = fused
        context.warnings = [warning for result in context.results for warning in result.warnings]
        generation_context = self._build_generation_context(context.results)
        answer = await self.decision_agent.generate_async(
            plan,
            context.results,
            fused,
            generation_context=generation_context,
        )
        return OrchestrationResult(
            answer=answer,
            plan=plan,
            subagent_results=context.results,
            fused_evidence=fused,
            reflection=reflection,
            warnings=context.warnings,
            generation_context=generation_context,
            route={
                "query_type": "video" if query.video_query_vector is not None else "doc" if query.doc_query_vector is not None else "text",
                "subagent_execution": self.subagent_execution,
                "max_reflection_rounds": self.max_reflection_rounds,
                "coverage": self.evidence_board.state_snapshot().get("coverage", {}),
                "evidence_pool": self.evidence_pool.snapshot(include_evidence=False),
            },
            trace=list(self.trace),
        )

    async def _run_tasks(self, tasks: list[RetrievalTask]) -> list[SubagentResult]:
        if not tasks:
            return []
        coroutines = [self._spawn_subagent(task).run(task) for task in tasks]
        return list(await asyncio.gather(*coroutines))

    def _spawn_subagent(self, task: RetrievalTask) -> ExpertSubagent:
        if task.agent == "doc_text_subagent":
            return DocTextSubagent(
                self.tools,
                self.evidence_writer,
                runner=self.runner,
                model=self.text_model,
            )
        if task.agent == "doc_visual_subagent":
            return DocVisualSubagent(
                self.tools,
                self.evidence_writer,
                runner=self.runner,
                model=self.visual_model,
            )
        if task.agent == "doc_graph_subagent":
            return DocGraphSubagent(
                self.tools,
                self.evidence_writer,
                runner=self.runner,
                model=self.graph_model,
            )
        if task.agent == "video_visual_subagent":
            return VideoVisualSubagent(
                self.tools,
                self.evidence_writer,
                runner=self.runner,
                model=self.visual_model,
            )
        if task.agent == "video_graph_subagent":
            return VideoGraphSubagent(
                self.tools,
                self.evidence_writer,
                runner=self.runner,
                model=self.graph_model,
            )
        return VideoTextSubagent(
            self.tools,
            self.evidence_writer,
            runner=self.runner,
            model=self.text_model,
        )

    async def _fuse_results(
        self,
        results: list[SubagentResult],
        *,
        top_k: int,
        query_text: str | None = None,
        query: QueryContext | None = None,
    ) -> list[dict[str, Any]]:
        pool_items = (
            self.evidence_pool.candidates(
                query,
                include_stale=True,
                limit=max(top_k * 3, 24),
                min_keyword_overlap=1,
            )
            if self.reuse_evidence_pool
            else []
        )
        items: list[dict[str, Any]] = list(pool_items)
        items.extend(self.evidence_board.evidence_items())
        for result in results:
            items.extend(result.evidence)
            data_items = result.data.get("items") if isinstance(result.data, dict) else None
            if isinstance(data_items, list):
                items.extend(item for item in data_items if isinstance(item, dict))
        items = self._filter_doc_evidence_for_query(items, query)
        if not items:
            return []
        has_video_only = bool(items) and all(item.get("source_type") == "video" for item in items)
        response = fuse_evidence_items(
            items,
            top_k=top_k,
            diversity_by_source=not has_video_only,
            query_text=query_text,
        )
        fused = response.data.get("items") if isinstance(response.data, dict) else None
        return list(fused) if isinstance(fused, list) else items[:top_k]

    @staticmethod
    def _filter_doc_evidence_for_query(
        items: list[dict[str, Any]],
        query: QueryContext | None,
    ) -> list[dict[str, Any]]:
        if query is None or query.doc_query_vector is None:
            return items
        target_docs = set(query.candidate_doc_ids or [])
        if query.source_doc_id:
            target_docs.add(query.source_doc_id)
        if not target_docs:
            return items
        filtered: list[dict[str, Any]] = []
        for item in items:
            source_type = item.get("source_type")
            if source_type not in {None, "doc"}:
                filtered.append(item)
                continue
            locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
            doc_id = (
                locator.get("doc_id")
                or metadata.get("doc_id")
                or provenance.get("doc_id")
                or item.get("doc_id")
            )
            if doc_id in target_docs:
                filtered.append(item)
        return filtered

    async def _update_evidence_state(
        self,
        query: QueryContext,
        results: list[SubagentResult],
    ) -> None:
        if not results:
            return
        await self.evidence_board.ingest_results(query.query_text, results)
        updated_pool_items = self.evidence_pool.upsert_results(query, results)
        snapshot = self.evidence_board.state_snapshot()
        query.metadata["evidence_facts"] = snapshot["facts"]
        query.metadata["evidence_gaps"] = snapshot["gaps"]
        query.metadata["evidence_reports"] = snapshot["reports"]
        query.metadata["evidence_pool"] = self.evidence_pool.snapshot(include_evidence=False)
        if updated_pool_items:
            self.trace.append(
                {
                    "step": "evidence_pool_update",
                    "updated_ids": [item.evidence_id for item in updated_pool_items],
                    "pool": self.evidence_pool.snapshot(include_evidence=False),
                }
            )

    def _build_generation_context(self, results: list[SubagentResult]) -> dict[str, Any]:
        text_contexts: list[str] = []
        video_contexts: list[str] = []
        query_rewrites: list[dict[str, Any]] = []
        for result in results:
            data = result.data if isinstance(result.data, dict) else {}
            text_context = data.get("retrieved_chunk_context")
            if isinstance(text_context, str) and text_context.strip():
                text_contexts.append(text_context)
            video_context = data.get("retrieved_video_context")
            if isinstance(video_context, str) and video_context.strip():
                video_contexts.append(video_context)
            rewrites = data.get("query_rewrites")
            if isinstance(rewrites, dict):
                query_rewrites.append(rewrites)
        return {
            "retrieved_chunk_context": "\n\n".join(text_contexts),
            "retrieved_video_context": "\n\n".join(video_contexts),
            "query_rewrites": query_rewrites,
            "evidence_board": self.evidence_board.state_snapshot(),
            "evidence_pool": self.evidence_pool.snapshot(include_evidence=False),
        }

    async def _rewrite_query(
        self,
        query_text: str,
        *,
        context_type: str,
        retrieval_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fallback = self._heuristic_rewrite(
            query_text,
            context_type=context_type,
            retrieval_hints=retrieval_hints,
        )
        prompt = self._build_query_rewrite_prompt(query_text, context_type)
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.decision_model,
                temperature=0.1,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.content or "{}")
        except Exception:
            return fallback
        return self._normalize_rewrite(data, fallback) if isinstance(data, dict) else fallback

    @staticmethod
    def _normalize_rewrite(data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        result = dict(fallback)
        for key in (
            "question_type",
            "entities",
            "actions",
            "textual_keywords",
            "visual_anchors",
            "text_query",
            "visual_query",
            "graph_query",
            "expanded_query",
        ):
            value = data.get(key)
            if isinstance(fallback.get(key), list):
                if isinstance(value, list):
                    result[key] = [str(item).strip() for item in value if str(item).strip()]
                elif isinstance(value, str) and value.strip():
                    result[key] = [value.strip()]
            elif isinstance(value, str) and value.strip():
                result[key] = value.strip()
        return result

    @staticmethod
    def _build_query_rewrite_prompt(query_text: str, context_type: str) -> str:
        return f"""Rewrite the user question into retrieval-focused forms for a {context_type} corpus.

USER QUERY: "{query_text}"

Return JSON only with these fields:
- "question_type": one of mechanism, cause, relation, comparison, challenge, temporal, description, factual.
- "entities": named people, objects, systems, organizations, methods, places, or domain terms.
- "actions": visible, spoken, or textual actions/events to search for.
- "textual_keywords": exact words and close paraphrases likely to appear in source text.
- "visual_anchors": objects, actors, scenes, interactions, diagrams, charts, tables, screens, or visible processes.
- "text_query": a concise text-retrieval query preserving entities and answer intent.
- "visual_query": a concrete scene, object, layout, chart, or action description for visual retrieval.
- "graph_query": an event/relation query for entity graph retrieval.
- "expanded_query": one dense sentence combining the above for broad recall.

Rules:
- Preserve the user's original entities exactly.
- For "how" questions, emphasize process, steps, mechanisms, and outcomes.
- For "why/what factors" questions, emphasize causes, reasons, conditions, and consequences.
- For comparisons, include both sides and comparison dimensions.
- For challenges, include obstacle, response, adaptation, and result.
- Do not add facts not implied by the question.
"""

    @classmethod
    def _heuristic_rewrite(
        cls,
        query_text: str,
        *,
        context_type: str,
        retrieval_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = query_text.strip()
        lower = text.lower()
        question_type = "factual"
        if lower.startswith("describe") or lower.startswith("illustrate"):
            question_type = "description"
        elif any(token in lower for token in ("compare", "difference", "differ", "trade-off", "tradeoff")):
            question_type = "comparison"
        elif any(token in lower for token in ("why", "factor", "cause", "reason", "prompt")):
            question_type = "cause"
        elif any(token in lower for token in ("relationship", "relation", "coordinate", "interact", "strategy")):
            question_type = "relation"
        elif any(token in lower for token in ("challenge", "problem", "difficulty", "hindering", "obstacle")):
            question_type = "challenge"
        elif any(token in lower for token in ("timeline", "when", "before", "after", "during")):
            question_type = "temporal"
        elif lower.startswith("how "):
            question_type = "mechanism"

        entities = cls._rewrite_entities(text, retrieval_hints=retrieval_hints)
        actions = cls._rewrite_action_terms(text)
        keywords = cls._rewrite_keywords(text)
        anchors = list(dict.fromkeys(entities + actions + cls._rewrite_visual_terms(text)))

        intent_terms = {
            "mechanism": "process steps behavior outcome",
            "cause": "causes reasons conditions factors consequences outcome",
            "relation": "relationship coordination interaction roles strategy",
            "comparison": "compare differences similarities advantages disadvantages",
            "challenge": "challenges obstacles response adaptation solution result",
            "temporal": "timeline sequence before after during stages",
            "description": "visible scene actions participants setting details",
            "factual": "specific answer evidence statement",
        }[question_type]

        entity_text = ", ".join(entities) if entities else text
        text_query = f"{text} {intent_terms}".strip()
        visual_query = (
            f"Visible evidence for {entity_text}: {' '.join(actions) or text}. "
            "Scene, actions, objects, interactions, and outcomes."
        )
        graph_query = (
            f"{entity_text}: {intent_terms}. Event chain, entity links, relations, "
            f"causes, and outcomes for: {text}"
        )
        expanded_query = (
            f"{text} Search for {intent_terms}; key entities: {entity_text}; "
            f"visual anchors: {', '.join(anchors[:12]) or text}."
        )

        return {
            "question_type": question_type,
            "entities": entities,
            "actions": actions,
            "textual_keywords": keywords,
            "visual_anchors": anchors[:16],
            "text_query": text_query,
            "visual_query": visual_query,
            "graph_query": graph_query,
            "expanded_query": expanded_query,
            "context_type": context_type,
        }

    @staticmethod
    def _rewrite_keywords(text: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "what",
            "when",
            "where",
            "which",
            "does",
            "are",
            "was",
            "were",
            "how",
            "why",
            "some",
            "their",
            "into",
            "about",
        }
        return list(
            dict.fromkeys(
                token
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text.lower())
                if token not in stopwords
            )
        )[:24]

    @staticmethod
    def _rewrite_entities(text: str, *, retrieval_hints: dict[str, Any] | None = None) -> list[str]:
        terms = re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)*\b", text)
        hinted_terms: list[str] = []
        hints = retrieval_hints or {}
        hint_values = hints.get("entity_terms")
        for term in hint_values if isinstance(hint_values, list) else []:
            term_text = str(term).strip()
            if term_text and re.search(rf"\b{re.escape(term_text)}s?\b", text, flags=re.IGNORECASE):
                hinted_terms.append(term_text)
        return list(dict.fromkeys([term.strip() for term in terms + hinted_terms if term.strip()]))[:16]

    @staticmethod
    def _rewrite_action_terms(text: str) -> list[str]:
        candidates = re.findall(r"\b[A-Za-z]+(?:ing|ed|es|s)\b", text.lower())
        return list(dict.fromkeys(term for term in candidates if term not in {"does", "questions"}))[:12]

    @staticmethod
    def _rewrite_visual_terms(text: str) -> list[str]:
        lower = text.lower()
        terms = []
        for token in (
            "scene",
            "chart",
            "table",
            "figure",
            "diagram",
            "slide",
            "screen",
            "demo",
            "award",
            "lecture",
            "interview",
        ):
            if token in lower:
                terms.append(token)
        return terms
