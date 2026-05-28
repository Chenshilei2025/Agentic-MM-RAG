"""Shared retrieval backends for document and video corpora."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic_mm_rag.schemas import EvidenceCard, ScoreParts, ToolResponse
from agentic_mm_rag.tools.runtime.contracts import (
    SeekRequest,
    RetrievalBackend,
)
from agentic_mm_rag.tools.runtime.filters import doc_chunk_allowed, resolve_doc_allowlist
from agentic_mm_rag.tools.runtime.evidence import (
    doc_evidence_from_chunk,
    video_caption_context,
    video_evidence_from_segment,
    video_text_evidence_from_chunk,
)
from agentic_mm_rag.tools.runtime.scoring import infer_doc_query_profile, normalize_score
from agentic_mm_rag.tools.runtime.scoring import fuse_evidence_items
from agentic_mm_rag.tools.runtime.stores.doc_store import DocRAGStore
from agentic_mm_rag.tools.runtime.stores.video_store import VideoRAGStore


def _best_scored_candidates(
    candidates: list[tuple[float, str, float]],
) -> list[tuple[float, str, float]]:
    best: dict[str, tuple[float, str, float]] = {}
    for score, item_id, raw_score in candidates:
        current = best.get(item_id)
        if current is None or score > current[0]:
            best[item_id] = (score, item_id, raw_score)
    return sorted(best.values(), key=lambda item: item[0], reverse=True)


@dataclass(slots=True)
class DocumentRetrievalBackend:
    """Shared retrieval backend for long-document retrieval."""

    store: DocRAGStore
    corpus_type: str = "doc"

    @staticmethod
    def _allowed_pages(pages: list[int] | None) -> set[int] | None:
        if not pages:
            return None
        allowed: set[int] = set()
        for page in pages:
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue
            allowed.add(page_int)
            allowed.add(page_int - 1)
        return allowed

    @staticmethod
    def _page_bias_pages(pages: list[int] | None) -> list[int]:
        if not pages:
            return []
        return [page for page in pages if isinstance(page, int) and page > 0]

    @staticmethod
    def _chunk_page_allowed(chunk: dict[str, Any], allowed_pages: set[int] | None) -> bool:
        if allowed_pages is None:
            return True
        try:
            return int(chunk.get("page_idx")) in allowed_pages
        except (TypeError, ValueError):
            return False

    def _focus_page_chunk_ids(
        self,
        query_text: str,
        *,
        doc_ids: set[str] | None,
        include_multimodal: bool,
        allowed_pages: set[int] | None,
    ) -> list[str]:
        focus_pages_by_doc = self.store.lexical_focus_pages(query_text, doc_ids=doc_ids)
        if not focus_pages_by_doc:
            return []
        chunk_ids: list[str] = []
        for chunk_id, chunk in self.store.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            if not doc_chunk_allowed(
                chunk,
                allow_docs=doc_ids,
                include_multimodal=include_multimodal,
            ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            doc_id = str(chunk.get("full_doc_id") or "")
            focus_pages = focus_pages_by_doc.get(doc_id, set())
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                continue
            if page_idx in focus_pages:
                chunk_ids.append(str(chunk_id))
        return chunk_ids

    async def doc_text_seek(self, request: SeekRequest) -> ToolResponse:
        allow_docs = resolve_doc_allowlist(self.store, request.doc_ids)
        allowed_pages = self._allowed_pages(request.evidence_pages)
        page_bias_pages = self._page_bias_pages(request.page_bias_pages)
        allow_ids = self.store.chunk_ids_for_docs(allow_docs, include_multimodal=request.include_multimodal)
        lexical_scores = self.store.lexical_chunk_scores(request.query_text or "", doc_ids=allow_docs)
        delta_scores = self.store.delta_text_chunk_scores(request.query_text or "", doc_ids=allow_docs)
        for chunk_id, score in delta_scores.items():
            lexical_scores[chunk_id] = max(lexical_scores.get(chunk_id, 0.0), score)
        visual_lexical_scores = self.store.visual_lexical_chunk_scores(request.query_text or "", doc_ids=allow_docs)
        for chunk_id, score in visual_lexical_scores.items():
            lexical_scores[chunk_id] = max(lexical_scores.get(chunk_id, 0.0), score)
        hits = self.store.chunks_vdb.query(
            request.query_vector,
            top_k=request.top_k * 12,
            min_score=request.min_score,
            ids_allowlist=allow_ids,
        )
        candidates: list[tuple[float, str, float]] = []
        profile = None
        if request.doc_ids and len(request.doc_ids) == 1:
            profile = self.store.document_profile(request.doc_ids[0])
        for hit in hits:
            chunk = self.store.chunk_by_id(hit.id)
            if not chunk:
                continue
            if not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                    include_multimodal=request.include_multimodal,
                ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            lexical = lexical_scores.get(hit.id, 0.0)
            modality_bonus = 0.08 if chunk.get("is_multimodal") and lexical > 0 else 0.0
            delta_bonus = min(1.2, delta_scores.get(hit.id, 0.0) * 0.25)
            page_bonus = 0.0
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            if page_idx is not None and page_idx in page_bias_pages:
                page_bonus = 1.6
            elif page_idx is not None and page_bias_pages and any(abs(page_idx - bias) <= 1 for bias in page_bias_pages):
                page_bonus = 0.8
            combined = hit.score + min(0.25, lexical * 0.2) + delta_bonus + modality_bonus + page_bonus
            candidates.append((combined, hit.id, hit.score))
        for chunk_id, lexical in lexical_scores.items():
            if any(existing_id == chunk_id for _score, existing_id, _raw in candidates):
                continue
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk or not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                include_multimodal=request.include_multimodal,
            ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            page_bonus = 0.0
            if page_idx is not None and page_idx in page_bias_pages:
                page_bonus = 1.6
            elif page_idx is not None and page_bias_pages and any(abs(page_idx - bias) <= 1 for bias in page_bias_pages):
                page_bonus = 0.8
            lexical_cap = 1.4 if chunk_id in delta_scores else 0.35
            candidates.append((min(lexical_cap, lexical * 0.25) + page_bonus, chunk_id, 0.0))
        for rank, chunk_id in enumerate(
            self._focus_page_chunk_ids(
                request.query_text or "",
                doc_ids=allow_docs,
                include_multimodal=request.include_multimodal,
                allowed_pages=allowed_pages,
            )
        ):
            if any(existing_id == chunk_id for _score, existing_id, _raw in candidates):
                continue
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk:
                continue
            candidates.append((0.32 - min(rank, 20) * 0.005, chunk_id, 0.0))
        candidates = _best_scored_candidates(candidates)
        evidence: list[EvidenceCard] = []
        neighbor_evidence: list[EvidenceCard] = []
        seen: set[str] = set()
        for combined, chunk_id, vector_score in candidates:
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk or chunk_id in seen:
                continue
            seen.add(chunk_id)
            lexical = lexical_scores.get(chunk_id, 0.0)
            score_parts = ScoreParts(text=vector_score or combined, rerank=min(0.25, lexical * 0.2))
            if profile and profile.get("text_sparse"):
                score_parts.source_filter += 0.15
            evidence.append(
                doc_evidence_from_chunk(
                    self.store,
                    chunk_id,
                    chunk,
                    score=combined,
                    retriever="doc_text_seek",
                    score_parts=score_parts,
                )
            )
            for sibling_id in self.store.sibling_chunk_ids(chunk_id, window=2, same_page=True):
                if sibling_id in seen:
                    continue
                sibling = self.store.chunk_by_id(sibling_id)
                if not sibling or not doc_chunk_allowed(
                    sibling,
                    allow_docs=allow_docs,
                    include_multimodal=request.include_multimodal,
                ):
                    continue
                if not self._chunk_page_allowed(sibling, allowed_pages):
                    continue
                seen.add(sibling_id)
                neighbor_card = doc_evidence_from_chunk(
                    self.store,
                    sibling_id,
                    sibling,
                    score=combined * 0.82,
                    retriever="doc_text_seek_neighbor",
                    score_parts=ScoreParts(text=vector_score * 0.5, rerank=0.1),
                )
                neighbor_card.provenance["anchor_chunk_id"] = chunk_id
                neighbor_evidence.append(neighbor_card)
            if len(evidence) >= request.top_k:
                break
        if len(evidence) < request.top_k:
            evidence.extend(neighbor_evidence[: request.top_k - len(evidence)])
        return ToolResponse(ok=True, tool="doc_text_seek", evidence=evidence)

    async def doc_visual_seek(self, request: SeekRequest) -> ToolResponse:
        modalities = list(request.visual_anchors or [])
        if any(modality in {"image", "figure", "diagram", "visual", "chart"} for modality in modalities):
            modalities.extend(["image", "chart"])
        allow_docs = resolve_doc_allowlist(self.store, request.doc_ids)
        allowed_pages = self._allowed_pages(request.evidence_pages)
        page_bias_pages = self._page_bias_pages(request.page_bias_pages)
        allowed_modalities = {m.lower() for m in modalities} if modalities else None
        allow_ids = self.store.multimodal_chunk_ids(modalities=modalities, doc_ids=allow_docs)
        requested_ids = list(dict.fromkeys(str(item) for item in request.visual_block_ids or []))
        traced_ids: list[str] = []
        for requested_id in requested_ids:
            chunk = self.store.chunk_by_id(requested_id)
            if not isinstance(chunk, dict):
                continue
            if chunk.get("is_multimodal"):
                traced_ids.append(requested_id)
            else:
                traced_ids.extend(self.store.linked_multimodal_chunk_ids(requested_id))
        traced_ids = list(dict.fromkeys(traced_ids))
        lexical_scores = self.store.visual_lexical_chunk_scores(request.query_text or "", doc_ids=allow_docs)
        hits = self.store.chunks_vdb.query(
            request.query_vector,
            top_k=request.top_k * 6,
            min_score=request.min_score,
            ids_allowlist=allow_ids,
        )
        candidates: list[tuple[float, str, float]] = []
        profile = None
        if request.doc_ids and len(request.doc_ids) == 1:
            profile = self.store.document_profile(request.doc_ids[0])
        for rank, chunk_id in enumerate(traced_ids):
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk or not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                include_multimodal=True,
                modalities=allowed_modalities,
            ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            page_bonus = 0.0
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            if page_idx is not None and page_idx in page_bias_pages:
                page_bonus = 1.6
            elif page_idx is not None and page_bias_pages and any(abs(page_idx - bias) <= 1 for bias in page_bias_pages):
                page_bonus = 0.8
            candidates.append((1.4 - rank * 0.01 + page_bonus, chunk_id, 0.0))
        for hit in hits:
            chunk = self.store.chunk_by_id(hit.id)
            if not chunk:
                continue
            if not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                include_multimodal=True,
                    modalities=allowed_modalities,
                ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            original_type = str(chunk.get("original_type") or "").lower()
            type_bonus = 0.18 if original_type in {"chart", "table", "image"} else 0.06
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            page_bonus = 0.0
            if page_idx is not None and page_idx in page_bias_pages:
                page_bonus = 1.6
            elif page_idx is not None and page_bias_pages and any(abs(page_idx - bias) <= 1 for bias in page_bias_pages):
                page_bonus = 0.8
            combined = hit.score + type_bonus + min(0.55, lexical_scores.get(hit.id, 0.0) * 0.22)
            combined += page_bonus
            candidates.append((combined, hit.id, hit.score))
        for chunk_id, lexical in lexical_scores.items():
            if any(existing_id == chunk_id for _score, existing_id, _raw in candidates):
                continue
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk:
                continue
            if not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                include_multimodal=True,
                modalities=allowed_modalities,
            ):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            page_bonus = 0.0
            if page_idx is not None and page_idx in page_bias_pages:
                page_bonus = 1.6
            elif page_idx is not None and page_bias_pages and any(abs(page_idx - bias) <= 1 for bias in page_bias_pages):
                page_bonus = 0.8
            candidates.append((min(1.9, 0.45 + lexical * 0.45) + page_bonus, chunk_id, 0.0))
        for rank, chunk_id in enumerate(
            self._focus_page_chunk_ids(
                request.query_text or "",
                doc_ids=allow_docs,
                include_multimodal=True,
                allowed_pages=allowed_pages,
            )
        ):
            if any(existing_id == chunk_id for _score, existing_id, _raw in candidates):
                continue
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk or not doc_chunk_allowed(
                chunk,
                allow_docs=allow_docs,
                include_multimodal=True,
                modalities=allowed_modalities,
            ):
                continue
            candidates.append((0.5 - min(rank, 20) * 0.005, chunk_id, 0.0))
        candidates = _best_scored_candidates(candidates)
        evidence: list[EvidenceCard] = []
        neighbor_evidence: list[EvidenceCard] = []
        seen: set[str] = set()
        for combined, chunk_id, vector_score in candidates:
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk or chunk_id in seen:
                continue
            seen.add(chunk_id)
            score_parts = ScoreParts(text=vector_score, visual=0.18)
            if chunk_id in traced_ids:
                score_parts.visual += 0.5
                score_parts.source_filter += 0.2
            if profile and (profile.get("visual_heavy") or profile.get("text_sparse")):
                score_parts.visual += 0.2
            card = doc_evidence_from_chunk(
                self.store,
                chunk_id,
                chunk,
                score=combined,
                retriever="doc_visual_trace" if chunk_id in traced_ids else "doc_visual_seek",
                score_parts=score_parts,
            )
            if chunk_id in traced_ids:
                card.metadata["inspection_mode"] = "source_visual_block"
                card.metadata["requested_visual_block_ids"] = requested_ids
            neighbor_context: list[dict[str, Any]] = []
            for sibling_id in self.store.sibling_chunk_ids(chunk_id, window=2, same_page=True):
                sibling = self.store.chunk_by_id(sibling_id)
                if not isinstance(sibling, dict):
                    continue
                neighbor_context.append(
                    {
                        "chunk_id": sibling_id,
                        "modality": sibling.get("original_type") if sibling.get("is_multimodal") else "text",
                        "content": str(sibling.get("content") or "")[:700],
                    }
                )
                if len(neighbor_context) >= 3:
                    break
            if neighbor_context:
                card.metadata["nearby_page_context"] = neighbor_context
            evidence.append(card)
            for sibling_id in self.store.sibling_chunk_ids(chunk_id, window=2, same_page=True):
                if sibling_id in seen:
                    continue
                sibling = self.store.chunk_by_id(sibling_id)
                if not sibling or not doc_chunk_allowed(
                    sibling,
                    allow_docs=allow_docs,
                    include_multimodal=True,
                ):
                    continue
                if not self._chunk_page_allowed(sibling, allowed_pages):
                    continue
                seen.add(sibling_id)
                neighbor_card = doc_evidence_from_chunk(
                    self.store,
                    sibling_id,
                    sibling,
                    score=combined * 0.8,
                    retriever="doc_visual_seek_neighbor",
                    score_parts=ScoreParts(visual=0.12, rerank=0.08),
                )
                neighbor_card.provenance["anchor_chunk_id"] = chunk_id
                neighbor_evidence.append(neighbor_card)
            if len(evidence) >= request.top_k:
                break
        if len(evidence) < request.top_k:
            evidence.extend(neighbor_evidence[: request.top_k - len(evidence)])
        return ToolResponse(ok=True, tool="doc_visual_seek", evidence=evidence)

    async def doc_graph_seek(self, request: SeekRequest) -> ToolResponse:
        entity_names: list[str] = []
        warnings: list[str] = []
        allow_docs = resolve_doc_allowlist(self.store, request.doc_ids)
        allowed_pages = self._allowed_pages(request.evidence_pages)
        doc_profile = None
        if request.doc_ids and len(request.doc_ids) == 1:
            doc_profile = self.store.document_profile(request.doc_ids[0])
        query_profile = infer_doc_query_profile(request.query_text or "")
        if request.graph_mode == "light_graph" and request.query_text:
            entity_names = self.store.graph.keyword_seed_entities(
                request.query_text,
                top_k=request.graph_top_k_entities or max(5, request.top_k),
            )
            if not entity_names:
                warnings.append("light_graph keyword seeding found no matching entities")
        else:
            hits = self.store.entities_vdb.query(
                request.query_vector,
                top_k=request.graph_top_k_entities or max(5, request.top_k),
                min_score=request.min_score,
            )
            for hit in hits:
                entity_name = str(hit.metadata.get("entity_name") or hit.metadata.get("entity_id") or hit.id)
                if self.store.is_noisy_entity(entity_name):
                    continue
                node = self.store.graph.get_node(entity_name)
                if node is None:
                    warnings.append(f"entity {entity_name} missing from graph")
                    continue
                clean = self.store.graph.clean_data(node)
                entity_names.append(str(clean.get("entity_name") or entity_name))
        relation_hits = self.store.relationships_vdb.query(
            request.query_vector,
            top_k=max(8, request.top_k * 3),
            min_score=request.min_score,
        )
        relation_chunk_scores: dict[str, float] = {}
        relation_breakdown: dict[str, dict[str, float]] = {}
        for rank, hit in enumerate(relation_hits):
            src_id = str(hit.metadata.get("src_id") or "")
            tgt_id = str(hit.metadata.get("tgt_id") or "")
            source_ids = [
                chunk_id
                for raw in str(hit.metadata.get("source_id") or "").split("<SEP>")
                for chunk_id in [raw.strip()]
                if chunk_id
            ]
            if not source_ids:
                source_ids = self.store.chunks_for_relation(src_id, tgt_id)
            rel_score = max(0.0, hit.score) * (1.0 / (rank + 1) ** 0.25)
            for chunk_id in source_ids[:6]:
                chunk = self.store.chunk_by_id(chunk_id)
                if not chunk or not doc_chunk_allowed(
                    chunk,
                    allow_docs=allow_docs,
                    include_multimodal=True,
                ):
                    continue
                if not self._chunk_page_allowed(chunk, allowed_pages):
                    continue
                relation_chunk_scores[chunk_id] = relation_chunk_scores.get(chunk_id, 0.0) + rel_score
                relation_breakdown.setdefault(chunk_id, {})[f"relationship:{src_id}->{tgt_id}"] = rel_score
            for entity_name in (src_id, tgt_id):
                if entity_name and entity_name not in entity_names and not self.store.is_noisy_entity(entity_name):
                    entity_names.append(entity_name)

        typed_edge_weights = {
            "semantic": 1.0,
            "temporal": 1.15 if query_profile["temporal_like"] else 1.05,
            "table": 1.35 if query_profile["table_like"] or query_profile["count_like"] else 1.15,
            "visual": 1.25 if query_profile["visual_like"] else 1.1,
        }
        if doc_profile and doc_profile.get("text_sparse"):
            typed_edge_weights["table"] += 0.1
            typed_edge_weights["visual"] += 0.1
        if request.graph_mode == "light_graph":
            chunk_scores, typed_breakdown = self.store.graph.typed_related_chunk_scores(
                entity_names,
                edge_type_weights=typed_edge_weights,
                max_neighbor_chunks_per_edge=1,
                allowed_edge_types=request.edge_type_filter,
            )
        else:
            chunk_scores, typed_breakdown = self.store.graph.typed_related_chunk_scores(
                entity_names,
                edge_type_weights=typed_edge_weights,
                max_neighbor_chunks_per_edge=3,
                allowed_edge_types=request.edge_type_filter,
            )
        if not chunk_scores:
            warnings.append("no graph-related chunks found from entity hits")
            for entity_name in entity_names:
                node = self.store.graph.get_node(entity_name)
                for chunk_id in self.store.graph.source_chunks(node):
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + 1.0
        for chunk_id, score in relation_chunk_scores.items():
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score
            typed_breakdown.setdefault(chunk_id, {}).update(relation_breakdown.get(chunk_id, {}))

        ranked_chunks = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        max_graph_score = ranked_chunks[0][1] if ranked_chunks else 0.0
        evidence: list[EvidenceCard] = []
        for chunk_id, graph_score in ranked_chunks:
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk:
                continue
            if not doc_chunk_allowed(chunk, allow_docs=allow_docs, include_multimodal=True):
                continue
            if not self._chunk_page_allowed(chunk, allowed_pages):
                continue
            score = normalize_score(graph_score, max_graph_score)
            card = doc_evidence_from_chunk(
                self.store,
                chunk_id,
                chunk,
                score=score,
                retriever="doc_graph_seek",
                score_parts=ScoreParts(
                    graph=score,
                    visual=0.15 if chunk.get("is_multimodal") else 0.0,
                    source_filter=0.15 if doc_profile and doc_profile.get("text_sparse") else 0.0,
                ),
            )
            card.provenance["entities"] = entity_names[: max(5, request.top_k)]
            card.metadata["graph_score_raw"] = graph_score
            card.metadata["typed_edge_contributions"] = typed_breakdown.get(chunk_id, {})
            evidence.append(card)
            if len(evidence) >= (request.graph_top_k_chunks or request.top_k):
                break
        return ToolResponse(ok=True, tool="doc_graph_seek", evidence=evidence, warnings=warnings)



@dataclass(slots=True)
class VideoRetrievalBackend:
    """Shared retrieval backend for long-video retrieval."""

    store: VideoRAGStore
    corpus_type: str = "video"

    def _chunk_ids_for_segment_ids(self, segment_ids: list[str]) -> list[str]:
        wanted = set(segment_ids)
        if not wanted:
            return []
        chunk_ids: list[str] = []
        for chunk_id, chunk in self.store.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            chunk_segments = chunk.get("video_segment_id")
            if not isinstance(chunk_segments, list):
                continue
            if wanted.intersection({str(segment_id) for segment_id in chunk_segments}):
                chunk_ids.append(str(chunk_id))
        return chunk_ids

    def _segment_ids_for_chunk(self, chunk: dict[str, Any], *, window: int = 0) -> list[str]:
        segment_ids = [
            str(segment_id)
            for segment_id in chunk.get("video_segment_id", [])
            if isinstance(segment_id, (str, int))
        ]
        if window <= 0:
            return segment_ids
        expanded: list[str] = []
        for segment_id in segment_ids:
            expanded.extend(self.store.expand_segment_ids(segment_id, window))
        return list(dict.fromkeys(expanded))

    async def video_text_seek(self, request: SeekRequest) -> ToolResponse:
        lexical_scores = self.store.lexical_chunk_scores(
            request.query_text or "",
            exact_detail=request.exact_detail_lexical,
        )
        lexical_segment_scores = self.store.lexical_segment_scores(
            request.query_text or "",
            exact_detail=request.exact_detail_lexical,
        )
        hits = self.store.chunks_vdb.query(
            request.query_vector,
            top_k=max(request.top_k * 8, request.top_k),
            min_score=request.min_score,
        )
        candidates: list[tuple[float, str, str, float]] = []
        for hit in hits:
            chunk = self.store.chunk_by_id(hit.id)
            if not chunk:
                continue
            lexical = lexical_scores.get(hit.id, 0.0)
            candidates.append(("chunk", hit.score + min(1.4, lexical * 0.9), hit.id, hit.score))
        for chunk_id, lexical in lexical_scores.items():
            if any(kind == "chunk" and existing_id == chunk_id for kind, _score, existing_id, _raw in candidates):
                continue
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk:
                continue
            candidates.append(("chunk", min(1.1, lexical * 0.75), chunk_id, 0.0))
        for segment_id, lexical in lexical_segment_scores.items():
            if lexical <= 0:
                continue
            if not self.store.segment_by_id(segment_id):
                continue
            candidates.append(("segment", min(1.45, lexical * 0.85 + 0.2), segment_id, 0.0))
            for chunk_id in self._chunk_ids_for_segment_ids(self.store.expand_segment_ids(segment_id, 1)):
                if any(kind == "chunk" and existing_id == chunk_id for kind, _score, existing_id, _raw in candidates):
                    continue
                chunk = self.store.chunk_by_id(chunk_id)
                if not chunk:
                    continue
                candidates.append(("chunk", min(0.95, lexical * 0.55 + 0.15), chunk_id, 0.0))
        candidates.sort(key=lambda item: item[1], reverse=True)
        evidence: list[EvidenceCard] = []
        neighbor_evidence: list[EvidenceCard] = []
        seen: set[str] = set()
        for kind, combined, item_id, vector_score in candidates:
            if item_id in seen:
                continue
            if kind == "segment":
                segment = self.store.segment_by_id(item_id)
                if not segment:
                    continue
                seen.add(item_id)
                lexical = lexical_segment_scores.get(item_id, 0.0)
                evidence.append(
                    video_evidence_from_segment(
                        self.store,
                        item_id,
                        segment,
                        score=combined,
                        retriever="video_text_segment_seek",
                        score_parts=ScoreParts(
                            text=combined,
                            rerank=min(1.0, lexical * 0.75),
                        ),
                    )
                )
                if len(evidence) >= request.top_k:
                    break
                continue
            chunk = self.store.chunk_by_id(item_id)
            if not chunk:
                continue
            seen.add(item_id)
            lexical = lexical_scores.get(item_id, 0.0)
            evidence.append(
                video_text_evidence_from_chunk(
                    self.store,
                    item_id,
                    chunk,
                    score=combined,
                    score_parts=ScoreParts(
                        text=vector_score or combined,
                        rerank=min(1.0, lexical * 0.75),
                    ),
                    include_mapped_segment_details=request.include_mapped_segment_details,
                )
            )
            if request.include_mapped_segment_details:
                mapped_segment_ids = [str(segment_id) for segment_id in chunk.get("video_segment_id", []) if isinstance(segment_id, (str, int))]
                for mapped_segment_id in mapped_segment_ids:
                    if len(evidence) >= request.top_k + 4:
                        break
                    segment = self.store.segment_by_id(mapped_segment_id)
                    if not segment or mapped_segment_id in seen:
                        continue
                    seen.add(mapped_segment_id)
                    evidence.append(
                        video_evidence_from_segment(
                            self.store,
                            mapped_segment_id,
                            segment,
                            score=combined * 0.9,
                            retriever="video_text_seek_mapped_segment",
                            score_parts=ScoreParts(
                                text=combined * 0.6,
                                rerank=min(1.0, lexical * 0.65),
                            ),
                        )
                    )
            neighbor_segment_ids = self._segment_ids_for_chunk(chunk, window=2)
            for sibling_id in self._chunk_ids_for_segment_ids(neighbor_segment_ids):
                if sibling_id in seen:
                    continue
                sibling = self.store.chunk_by_id(sibling_id)
                if not sibling:
                    continue
                seen.add(sibling_id)
                neighbor_evidence.append(
                    video_text_evidence_from_chunk(
                        self.store,
                        sibling_id,
                        sibling,
                        score=combined * 0.82,
                        score_parts=ScoreParts(text=vector_score * 0.5, rerank=0.1),
                        include_mapped_segment_details=request.include_mapped_segment_details,
                    )
                )
            if len(evidence) >= request.top_k:
                break
        if len(evidence) < request.top_k:
            evidence.extend(neighbor_evidence[: request.top_k - len(evidence)])
        return ToolResponse(ok=True, tool="video_text_seek", evidence=evidence)

    async def video_visual_seek(self, request: SeekRequest) -> ToolResponse:
        lexical_scores = self.store.lexical_segment_scores(request.query_text or "")
        query_vector = request.visual_query_vector or request.query_vector
        hits = self.store.segment_feature_vdb.query(
            query_vector,
            top_k=max(request.top_k * 10, request.top_k),
            min_score=request.min_score,
        )
        requested_segment_ids = list(dict.fromkeys(str(item) for item in request.segment_ids or []))
        candidates: list[tuple[float, str, float, str]] = []
        for rank, segment_id in enumerate(requested_segment_ids):
            for expanded_segment_id in self.store.expand_segment_ids(segment_id, 2):
                if not self.store.segment_by_id(expanded_segment_id):
                    continue
                lexical = lexical_scores.get(expanded_segment_id, 0.0)
                trace_score = 1.65 - rank * 0.03 + min(0.35, lexical * 0.2)
                if expanded_segment_id != segment_id:
                    trace_score *= 0.9
                candidates.append((trace_score, expanded_segment_id, 0.0, "video_visual_trace"))
        for hit in hits:
            segment = self.store.segment_by_id(hit.id)
            if not segment:
                continue
            lexical = lexical_scores.get(hit.id, 0.0)
            candidates.append((hit.score + min(1.4, lexical * 0.9), hit.id, hit.score, "video_visual_seek"))
        for segment_id, lexical in lexical_scores.items():
            if any(existing_id == segment_id for _score, existing_id, _raw, _source in candidates):
                continue
            if not self.store.segment_by_id(segment_id):
                continue
            candidates.append((min(1.1, lexical * 0.75), segment_id, 0.0, "video_visual_lexical"))
        best_candidates: dict[str, tuple[float, str, float, str]] = {}
        for score, segment_id, raw_score, source in candidates:
            current = best_candidates.get(segment_id)
            if current is None or score > current[0]:
                best_candidates[segment_id] = (score, segment_id, raw_score, source)
        candidates = sorted(best_candidates.values(), key=lambda item: item[0], reverse=True)
        evidence: list[EvidenceCard] = []
        neighbor_evidence: list[EvidenceCard] = []
        seen: set[str] = set()
        for combined, segment_id, vector_score, retriever in candidates:
            if segment_id in seen:
                continue
            segment = self.store.segment_by_id(segment_id)
            if not segment:
                continue
            seen.add(segment_id)
            card = video_evidence_from_segment(
                self.store,
                segment_id,
                segment,
                score=combined,
                retriever=retriever,
                score_parts=ScoreParts(
                    visual=vector_score or combined,
                    rerank=min(1.0, lexical_scores.get(segment_id, 0.0) * 0.75),
                    source_filter=0.2 if retriever == "video_visual_trace" else 0.0,
                ),
            )
            if requested_segment_ids:
                card.metadata["requested_segment_ids"] = requested_segment_ids
            evidence.append(card)
            for sibling_id in self.store.expand_segment_ids(segment_id, 1):
                if sibling_id in seen:
                    continue
                sibling = self.store.segment_by_id(sibling_id)
                if not sibling:
                    continue
                seen.add(sibling_id)
                neighbor_evidence.append(
                    video_evidence_from_segment(
                        self.store,
                        sibling_id,
                        sibling,
                        score=combined * 0.85,
                        retriever="video_visual_seek_neighbor",
                        score_parts=ScoreParts(visual=vector_score * 0.6, rerank=0.08),
                    )
                )
            if len(evidence) >= request.top_k:
                break
        if len(evidence) < request.top_k:
            evidence.extend(neighbor_evidence[: request.top_k - len(evidence)])
        return ToolResponse(ok=True, tool="video_visual_seek", evidence=evidence)

    async def video_graph_seek(self, request: SeekRequest) -> ToolResponse:
        query_text = request.query_text or ""
        hits = self.store.entities_vdb.query(
            request.query_vector,
            top_k=request.graph_top_k_entities or max(8, request.top_k),
            min_score=request.min_score,
        )
        entity_names: list[str] = []
        warnings: list[str] = []
        for hit in hits:
            entity_name = str(hit.metadata.get("entity_name") or hit.id)
            node = self.store.graph.get_node(entity_name)
            if node is None:
                warnings.append(f"entity {entity_name} missing from graph")
                continue
            node_data = self.store.graph.clean_data(node)
            entity_names.append(str(node_data.get("entity_name") or entity_name))

        chunk_scores = self.store.graph.related_chunk_scores(entity_names)
        ranked_chunks = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        graph_chunk_limit = request.graph_top_k_chunks or max(4, request.top_k)
        if graph_chunk_limit > 0:
            ranked_chunks = ranked_chunks[:graph_chunk_limit]

        segment_scores: dict[str, float] = {}
        segment_source_chunks: dict[str, list[str]] = defaultdict(list)
        for chunk_id, graph_score in ranked_chunks:
            chunk = self.store.chunk_by_id(chunk_id)
            if not chunk:
                continue
            for segment_id in chunk.get("video_segment_id", []):
                segment_scores[segment_id] = max(segment_scores.get(segment_id, 0.0), graph_score)
                segment_source_chunks[segment_id].append(chunk_id)

        if not segment_scores:
            warnings.append("no graph-ranked segments found; falling back to direct entity chunks")
            for entity_name in entity_names:
                node = self.store.graph.get_node(entity_name)
                for chunk_id in self.store.graph.source_chunks(node):
                    chunk = self.store.chunk_by_id(chunk_id)
                    if not chunk:
                        continue
                    for segment_id in chunk.get("video_segment_id", []):
                        segment_scores[segment_id] = max(segment_scores.get(segment_id, 0.0), 1.0)
                        segment_source_chunks[segment_id].append(chunk_id)

        ranked_segments = sorted(segment_scores.items(), key=lambda item: item[1], reverse=True)
        max_graph_score = ranked_segments[0][1] if ranked_segments else 0.0
        evidence: list[EvidenceCard] = []
        for segment_id, graph_score in ranked_segments:
            segment = self.store.segment_by_id(segment_id)
            if not segment:
                continue
            score = normalize_score(graph_score, max_graph_score)
            card = video_evidence_from_segment(
                self.store,
                segment_id,
                segment,
                score=score,
                retriever="video_graph_seek",
                score_parts=ScoreParts(graph=score),
            )
            card.provenance["entities"] = entity_names[: max(5, request.top_k)]
            card.provenance["source_chunks"] = segment_source_chunks.get(segment_id, [])
            card.metadata["graph_score_raw"] = graph_score
            if request.edge_type_filter:
                card.metadata["requested_edge_types"] = list(request.edge_type_filter)
            if query_text:
                card.metadata["query_text"] = query_text
            evidence.append(card)
            if len(evidence) >= request.top_k:
                break
        return ToolResponse(ok=True, tool="video_graph_seek", evidence=evidence, warnings=warnings)



def _resolve_doc_root(root: str | Path | None = None) -> Path:
    return (Path(root) if root is not None else DocRAGStore().root).resolve()  # type: ignore[arg-type]


def _resolve_video_root(root: str | Path | None = None) -> Path:
    return (Path(root) if root is not None else VideoRAGStore().root).resolve()  # type: ignore[arg-type]


@lru_cache(maxsize=8)
def _build_doc_backend_cached(root: str) -> DocumentRetrievalBackend:
    return DocumentRetrievalBackend(store=DocRAGStore(Path(root)))


@lru_cache(maxsize=8)
def _build_video_backend_cached(root: str) -> VideoRetrievalBackend:
    return VideoRetrievalBackend(store=VideoRAGStore(Path(root)))


def build_doc_backend(*, root: str | Path | None = None) -> DocumentRetrievalBackend:
    return _build_doc_backend_cached(str(_resolve_doc_root(root)))


def build_video_backend(*, root: str | Path | None = None) -> VideoRetrievalBackend:
    return _build_video_backend_cached(str(_resolve_video_root(root)))


def build_backend(
    corpus_type: str,
    *,
    root: str | Path | None = None,
) -> RetrievalBackend:
    normalized = corpus_type.lower()
    if normalized in {"doc", "docs", "document", "documents"}:
        return build_doc_backend(root=root)
    if normalized in {"video", "videos"}:
        return build_video_backend(root=root)
    raise ValueError(f"unsupported corpus_type: {corpus_type}")
