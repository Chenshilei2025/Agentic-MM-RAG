"""Adapter for processed RAGAnything storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from agentic_mm_rag.config import DEFAULT_PATHS
from agentic_mm_rag.tools.runtime.stores.graph import GraphMLStore
from agentic_mm_rag.tools.runtime.stores.json_utils import load_json, safe_count_pattern
from agentic_mm_rag.tools.runtime.stores.vector import JsonVectorStore


NOISY_ENTITY_TERMS = {
    "ocr artifact",
    "discarded content",
    "discarded content analysis",
    "document layout",
    "page layout",
    "pagination artifact",
    "page header",
    "page footer",
    "bbox",
    "page_idx",
    "bounding box",
    "source document",
}


@dataclass(slots=True)
class DocRAGStore:
    root: Path = DEFAULT_PATHS.doc_rag_dir
    visual_asset_roots: tuple[Path, ...] = DEFAULT_PATHS.doc_visual_asset_roots
    chunks_vdb: JsonVectorStore = field(init=False)
    entities_vdb: JsonVectorStore = field(init=False)
    relationships_vdb: JsonVectorStore = field(init=False)
    graph: GraphMLStore = field(init=False)
    _document_profile_cache: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    _visual_asset_cache: dict[str, dict[tuple[int, str], list[str]]] = field(
        init=False,
        default_factory=dict,
    )
    _focus_page_cache: dict[tuple[str, str], set[int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.visual_asset_roots = tuple(Path(root) for root in self.visual_asset_roots)
        self.chunks_vdb = JsonVectorStore(self.root / "vdb_chunks.json")
        self.entities_vdb = JsonVectorStore(self.root / "vdb_entities.json")
        self.relationships_vdb = JsonVectorStore(self.root / "vdb_relationships.json")
        self.graph = GraphMLStore(self.root / "graph_chunk_entity_relation.graphml")

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def doc_status(self) -> dict[str, Any]:
        return load_json(self.path("kv_store_doc_status.json"))

    @property
    def text_chunks(self) -> dict[str, Any]:
        return load_json(self.path("kv_store_text_chunks.json"))

    def manifest(self) -> dict[str, Any]:
        status = self.doc_status
        processed = sum(1 for item in status.values() if item.get("status") == "processed")
        failed = sum(1 for item in status.values() if item.get("status") == "failed")
        text_chunks = self.text_chunks
        multimodal_chunks = sum(
            1 for item in text_chunks.values() if item.get("is_multimodal") is True
        )
        return {
            "root": str(self.root),
            "documents": len(status),
            "processed_documents": processed,
            "failed_documents": failed,
            "text_chunks": len(text_chunks),
            "multimodal_chunks": multimodal_chunks,
            "chunk_vectors": safe_count_pattern(self.path("vdb_chunks.json"), '"__id__"'),
            "entity_vectors_estimate": safe_count_pattern(
                self.path("vdb_entities.json"), '"__id__"'
            ),
            "relationship_vectors_estimate": safe_count_pattern(
                self.path("vdb_relationships.json"), '"__id__"'
            ),
        }

    def file_for_doc(self, doc_id: str | None) -> str | None:
        if not doc_id:
            return None
        item = self.doc_status.get(doc_id)
        if isinstance(item, dict):
            return item.get("file_path")
        return None

    def resolve_doc_id(self, doc_id: str | None) -> str | None:
        if not doc_id:
            return None
        if doc_id in self.doc_status:
            return str(doc_id)
        target = str(doc_id).strip()
        target_name = Path(target).name
        normalized = target.replace("_origin", "")
        normalized_name = Path(normalized).name
        for internal_doc_id, item in self.doc_status.items():
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file_path") or "").strip()
            if not file_path:
                continue
            candidates = {
                file_path,
                Path(file_path).name,
                file_path.replace("_origin", ""),
                Path(file_path.replace("_origin", "")).name,
            }
            if target in candidates or target_name in candidates or normalized in candidates or normalized_name in candidates:
                return str(internal_doc_id)
        return None

    def chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        item = self.text_chunks.get(chunk_id)
        return item if isinstance(item, dict) else None

    def resolve_visual_asset_path(self, chunk: dict[str, Any]) -> str | None:
        """Best-effort source image path for multimodal chunks missing Image Path text."""

        doc_id = str(chunk.get("full_doc_id") or "")
        original_type = str(chunk.get("original_type") or "").lower()
        page_idx = chunk.get("page_idx")
        if not doc_id or original_type not in {"chart", "table", "image"}:
            return None
        try:
            page = int(page_idx)
        except (TypeError, ValueError):
            return None
        index = self._visual_asset_index(doc_id)
        candidates = index.get((page, original_type)) or index.get((page, "image")) or []
        if not candidates:
            return None
        order = chunk.get("chunk_order_index")
        try:
            text_chunks = self.text_chunks
        except FileNotFoundError:
            text_chunks = {}
        same_page_chunks = [
            item
            for item in text_chunks.values()
            if isinstance(item, dict)
            and item.get("full_doc_id") == doc_id
            and item.get("page_idx") == page
            and str(item.get("original_type") or "").lower() == original_type
        ]
        same_page_chunks.sort(key=lambda item: int(item.get("chunk_order_index") or 0))
        rank = 0
        for idx, item in enumerate(same_page_chunks):
            if item.get("chunk_order_index") == order:
                rank = idx
                break
        return candidates[min(rank, len(candidates) - 1)]

    def _visual_asset_index(self, doc_id: str) -> dict[tuple[int, str], list[str]]:
        cached = self._visual_asset_cache.get(doc_id)
        if cached is not None:
            return cached
        index: dict[tuple[int, str], list[str]] = {}
        stem = Path(doc_id).stem
        search_roots = (self.root,) + self.visual_asset_roots
        candidates: list[Path] = []
        for search_root in search_roots:
            base = Path(search_root)
            candidates.extend(base.glob(f"{stem}_origin_*"))
            candidates.extend(base.glob(f"**/{stem}_origin_*"))
        for base in candidates:
            for content_list in base.glob("**/*content_list.json"):
                try:
                    items = load_json(content_list)
                except Exception:
                    continue
                if not isinstance(items, list):
                    continue
                auto_dir = content_list.parent
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    typ = str(item.get("type") or "").lower()
                    rel_path = item.get("img_path")
                    if typ not in {"chart", "table", "image"} or not rel_path:
                        continue
                    try:
                        page = int(item.get("page_idx"))
                    except (TypeError, ValueError):
                        continue
                    path = auto_dir / str(rel_path)
                    index.setdefault((page, typ), []).append(str(path))
        self._visual_asset_cache[doc_id] = index
        return index

    def sibling_chunk_ids(
        self,
        chunk_id: str,
        *,
        window: int = 1,
        same_page: bool = False,
    ) -> list[str]:
        """Return nearby chunks in the same document/order, optionally same page."""

        chunk = self.chunk_by_id(chunk_id)
        if not isinstance(chunk, dict):
            return []
        doc_id = chunk.get("full_doc_id")
        order = chunk.get("chunk_order_index")
        page_idx = chunk.get("page_idx")
        if not isinstance(doc_id, str) or not isinstance(order, int):
            return []
        results: list[tuple[int, str]] = []
        for candidate_id, candidate in self.text_chunks.items():
            if candidate_id == chunk_id or not isinstance(candidate, dict):
                continue
            if candidate.get("full_doc_id") != doc_id:
                continue
            candidate_order = candidate.get("chunk_order_index")
            if not isinstance(candidate_order, int):
                continue
            if abs(candidate_order - order) > window:
                continue
            if same_page and page_idx is not None and candidate.get("page_idx") != page_idx:
                continue
            results.append((abs(candidate_order - order), str(candidate_id)))
        results.sort(key=lambda item: item[0])
        return [candidate_id for _distance, candidate_id in results]

    def multimodal_chunk_ids(
        self,
        modalities: list[str] | None = None,
        doc_ids: set[str] | None = None,
    ) -> set[str]:
        modalities_set = {m.lower() for m in modalities} if modalities else None
        result: set[str] = set()
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict) or not chunk.get("is_multimodal"):
                continue
            if doc_ids is not None and chunk.get("full_doc_id") not in doc_ids:
                continue
            original_type = str(chunk.get("original_type", "unknown")).lower()
            if modalities_set is not None and original_type not in modalities_set:
                continue
            result.add(chunk_id)
        return result

    def linked_multimodal_chunk_ids(
        self,
        chunk_id: str,
        *,
        window: int = 2,
    ) -> list[str]:
        """Return multimodal chunks near a text chunk for visual follow-up."""

        chunk = self.chunk_by_id(chunk_id)
        if not isinstance(chunk, dict):
            return []
        if chunk.get("is_multimodal"):
            return [chunk_id]
        doc_id = chunk.get("full_doc_id")
        order = chunk.get("chunk_order_index")
        page_idx = chunk.get("page_idx")
        if not isinstance(doc_id, str):
            return []
        results: list[tuple[int, str]] = []
        for candidate_id, candidate in self.text_chunks.items():
            if not isinstance(candidate, dict) or not candidate.get("is_multimodal"):
                continue
            if candidate.get("full_doc_id") != doc_id:
                continue
            candidate_page = candidate.get("page_idx")
            candidate_order = candidate.get("chunk_order_index")
            if page_idx is not None and candidate_page == page_idx:
                distance = (
                    abs(candidate_order - order)
                    if isinstance(candidate_order, int) and isinstance(order, int)
                    else 0
                )
                results.append((distance, str(candidate_id)))
                continue
            if isinstance(candidate_order, int) and isinstance(order, int):
                distance = abs(candidate_order - order)
                if distance <= window:
                    results.append((distance + 10, str(candidate_id)))
        results.sort(key=lambda item: item[0])
        return list(dict.fromkeys(candidate_id for _distance, candidate_id in results))

    def chunks_for_relation(self, src_id: str | None, tgt_id: str | None) -> list[str]:
        if not src_id or not tgt_id:
            return []
        keys = [f"{src_id}<SEP>{tgt_id}", f"{tgt_id}<SEP>{src_id}"]
        relation_chunks = load_json(self.path("kv_store_relation_chunks.json"))
        for key in keys:
            payload = relation_chunks.get(key)
            if isinstance(payload, dict) and isinstance(payload.get("chunk_ids"), list):
                return [str(chunk_id) for chunk_id in payload["chunk_ids"]]
        return []

    def lexical_chunk_scores(
        self,
        query_text: str,
        *,
        doc_ids: set[str] | None = None,
    ) -> dict[str, float]:
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", query_text.casefold())
            if token
            not in {
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
            }
        }
        if not query_tokens:
            return {}
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            if doc_ids is not None and chunk.get("full_doc_id") not in doc_ids:
                continue
            content = str(chunk.get("content") or "").casefold()
            if not content:
                continue
            overlap = sum(1 for token in query_tokens if token in content)
            if overlap:
                scores[str(chunk_id)] = overlap / max(len(query_tokens), 1)
        return scores

    def delta_text_chunk_scores(
        self,
        query_text: str,
        *,
        doc_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Boost text chunks that explicitly state temporal subgroup deltas."""

        query = query_text.casefold()
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        wants_delta = (
            len(years) >= 2
            and any(token in query for token in ("gain", "gained", "increase", "change", "most", "largest"))
            and any(token in query for token in ("subgroup", "subgroups", "hispanic", "latino", "confidence"))
        )
        if not wants_delta:
            return {}
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            if doc_ids is not None and chunk.get("full_doc_id") not in doc_ids:
                continue
            content = str(chunk.get("content") or "").casefold()
            if not content:
                continue
            score = 0.0
            score += len(years & set(re.findall(r"\b(?:19|20)\d{2}\b", content))) * 0.8
            if any(token in content for token in ("subgroup", "subgroups", "demographic")):
                score += 1.0
            if any(token in content for token in ("education", "college", "high school", "u.s.-born", "foreign-born", "ages")):
                score += 1.2
            if re.search(r"\+\s?\d+\s*(?:percentage\s+)?points?", content):
                score += 2.5
            if "percentage point" in content:
                score += 1.5
            if any(token in content for token in ("excellent", "good", "confidence", "personal finances", "economic optimism")):
                score += 0.8
            if score > 0:
                scores[str(chunk_id)] = score
        return scores

    def lexical_focus_pages(
        self,
        query_text: str,
        *,
        doc_ids: set[str] | None = None,
    ) -> dict[str, set[int]]:
        """Find target-document pages that should be prefiltered for special question types."""

        query = query_text.casefold()
        wants_references = any(token in query for token in ("reference", "references", "citation", "citations", "bibliography"))
        wants_subgroup_change = (
            any(token in query for token in ("subgroup", "subgroups", "demographic", "education", "college"))
            and any(token in query for token in ("gain", "gained", "increase", "change", "from", "between"))
            and len(re.findall(r"\b(?:19|20)\d{2}\b", query)) >= 2
        )
        wants_temporal_delta = (
            any(token in query for token in ("gain", "gained", "increase", "decrease", "change", "most", "largest"))
            and len(re.findall(r"\b(?:19|20)\d{2}\b", query)) >= 2
        )
        if not (wants_references or wants_subgroup_change or wants_temporal_delta):
            return {}
        focus_kind = "references" if wants_references else "subgroup_delta"
        doc_pool = doc_ids or {
            str(chunk.get("full_doc_id"))
            for chunk in self.text_chunks.values()
            if isinstance(chunk, dict) and chunk.get("full_doc_id")
        }
        focus_by_doc: dict[str, set[int]] = {}
        for doc_id in doc_pool:
            cache_key = (str(doc_id), focus_kind)
            cached = self._focus_page_cache.get(cache_key)
            if cached is not None:
                if cached:
                    focus_by_doc[str(doc_id)] = set(cached)
                continue
            candidates: list[tuple[float, int]] = []
            profile = self.document_profile(str(doc_id))
            unique_pages = int(profile.get("unique_pages") or 0)
            for chunk in self.text_chunks.values():
                if not isinstance(chunk, dict) or chunk.get("full_doc_id") != doc_id:
                    continue
                try:
                    page = int(chunk.get("page_idx"))
                except (TypeError, ValueError):
                    continue
                content = str(chunk.get("content") or "").casefold()
                modal_name = str(chunk.get("modal_entity_name") or "").casefold()
                text = f"{modal_name}\n{content}"
                score = 0.0
                if wants_references:
                    if re.search(r"\bappendix\s+[a-z0-9]*\s*[:.-]?\s*references\b", text):
                        score += 6.0
                    if re.search(r"(^|\n)\s*(appendix|references|bibliography|citations)\b", text):
                        score += 3.5
                    if "references" in text:
                        score += 1.4
                    if "reference" in text or "citation" in text:
                        score += 0.7
                    if unique_pages and page >= max(0, unique_pages - 8):
                        score += 1.0
                    if str(chunk.get("original_type") or "").lower() in {"page_footnote", "aside_text"}:
                        score -= 0.8
                if wants_subgroup_change or wants_temporal_delta:
                    year_hits = len(set(re.findall(r"\b(?:19|20)\d{2}\b", text)) & set(re.findall(r"\b(?:19|20)\d{2}\b", query)))
                    score += year_hits * 1.0
                    if any(token in text for token in ("subgroup", "subgroups", "demographic", "generation", "education", "college", "high school", "u.s.-born", "foreign-born", "ages")):
                        score += 1.6
                    if any(token in text for token in ("gain", "gains", "increase", "increased", "change", "percentage point")):
                        score += 1.5
                    if any(token in text for token in ("excellent", "good", "confidence", "optimistic", "personal finances")):
                        score += 1.0
                    if str(chunk.get("original_type") or "").lower() in {"chart", "table"}:
                        score += 0.8
                if score > 0:
                    candidates.append((score, page))
            candidates.sort(key=lambda item: item[0], reverse=True)
            pages: set[int] = set()
            max_score = candidates[0][0] if candidates else 0.0
            max_candidates = 8 if wants_references else 4
            for score, page in candidates[:max_candidates]:
                threshold = 2.0 if wants_references else max(3.0, max_score - 1.0)
                if score < threshold and pages:
                    continue
                pages.update({page - 1, page, page + 1})
            pages = {page for page in pages if page >= 0}
            self._focus_page_cache[cache_key] = pages
            if pages:
                focus_by_doc[str(doc_id)] = set(pages)
        return focus_by_doc

    def visual_lexical_chunk_scores(
        self,
        query_text: str,
        *,
        doc_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Lexical prior for target-document visual/table/chart chunks."""

        base_scores = self.lexical_chunk_scores(query_text, doc_ids=doc_ids)
        query = query_text.casefold()
        focus_pages_by_doc = self.lexical_focus_pages(query_text, doc_ids=doc_ids)
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict) or not chunk.get("is_multimodal"):
                continue
            if doc_ids is not None and chunk.get("full_doc_id") not in doc_ids:
                continue
            original_type = str(chunk.get("original_type") or "").lower()
            if original_type not in {"chart", "table", "image", "page_footnote"}:
                continue
            content = str(chunk.get("content") or "").casefold()
            modal_name = str(chunk.get("modal_entity_name") or "").casefold()
            text = f"{content}\n{modal_name}"
            score = base_scores.get(str(chunk_id), 0.0)
            doc_id = str(chunk.get("full_doc_id") or "")
            try:
                page_idx = int(chunk.get("page_idx"))
            except (TypeError, ValueError):
                page_idx = None
            focus_pages = focus_pages_by_doc.get(doc_id, set())
            if page_idx is not None and page_idx in focus_pages:
                score += 1.4
            if original_type in {"chart", "table"} and any(
                token in query
                for token in ("chart", "table", "figure", "percentage", "percent", "how many")
            ):
                score += 0.5
            if "reference" in query or "references" in query:
                if re.search(r"\bappendix\s+[a-z0-9]*\s*[:.-]?\s*references\b", text):
                    score += 2.2
                elif re.search(r"(^|\n)\s*(appendix|references|bibliography|citations)\b", text):
                    score += 1.6
                elif "reference" in text or "appendix" in text or "citation" in text:
                    score += 1.2
                profile = self.document_profile(str(chunk.get("full_doc_id") or ""))
                unique_pages = int(profile.get("unique_pages") or 0)
                if page_idx is not None and unique_pages and page_idx >= max(0, unique_pages - 6):
                    score += 0.8
                if original_type == "page_footnote":
                    score -= 0.5
            if any(token in query for token in ("methodology", "survey", "sample", "interview")):
                if any(token in text for token in ("methodology", "sample", "interview", "survey")):
                    score += 0.8
            if (
                any(token in query for token in ("subgroup", "subgroups", "demographic", "education", "college", "gain", "gained", "increase", "change"))
                and len(re.findall(r"\b(?:19|20)\d{2}\b", query)) >= 2
            ):
                if page_idx is not None and page_idx in focus_pages:
                    score += 1.4
                if any(token in text for token in ("subgroup", "subgroups", "demographic", "education", "college", "high school", "percentage point")):
                    score += 1.2
                if any(year in text for year in re.findall(r"\b(?:19|20)\d{2}\b", query)):
                    score += 0.7
            if score > 0:
                scores[str(chunk_id)] = score
        return scores

    def chunk_ids_for_docs(
        self,
        doc_ids: set[str] | None = None,
        *,
        include_multimodal: bool = True,
    ) -> set[str] | None:
        if doc_ids is None:
            return None
        result: set[str] = set()
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            if chunk.get("full_doc_id") not in doc_ids:
                continue
            if not include_multimodal and chunk.get("is_multimodal"):
                continue
            result.add(str(chunk_id))
        return result

    @property
    def full_docs(self) -> dict[str, Any]:
        return load_json(self.path("kv_store_full_docs.json"))

    def document_profile(self, doc_id: str | None) -> dict[str, Any]:
        internal_doc_id = self.resolve_doc_id(doc_id)
        cache_key = internal_doc_id or str(doc_id or "")
        cached = self._document_profile_cache.get(cache_key)
        if cached is not None:
            return cached
        if not internal_doc_id:
            profile = {
                "doc_id": doc_id,
                "resolved_doc_id": None,
                "total_chunks": 0,
                "searchable_chunks": 0,
                "text_chunks": 0,
                "multimodal_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "equation_chunks": 0,
                "discarded_chunks": 0,
                "chunks_with_page_idx": 0,
                "unique_pages": 0,
                "text_ratio": 0.0,
                "multimodal_ratio": 0.0,
                "page_coverage_ratio": 0.0,
                "sparse": True,
                "text_sparse": True,
                "visual_heavy": False,
            }
            self._document_profile_cache[cache_key] = profile
            return profile

        total_chunks = 0
        text_chunks = 0
        multimodal_chunks = 0
        table_chunks = 0
        image_chunks = 0
        equation_chunks = 0
        discarded_chunks = 0
        chunks_with_page_idx = 0
        unique_pages: set[int] = set()

        for chunk in self.text_chunks.values():
            if not isinstance(chunk, dict):
                continue
            if chunk.get("full_doc_id") != internal_doc_id:
                continue
            total_chunks += 1
            page_idx = chunk.get("page_idx")
            if isinstance(page_idx, int):
                chunks_with_page_idx += 1
                unique_pages.add(page_idx)
            if chunk.get("is_multimodal"):
                multimodal_chunks += 1
                original_type = str(chunk.get("original_type") or "unknown").lower()
                if original_type == "table":
                    table_chunks += 1
                elif original_type == "image":
                    image_chunks += 1
                elif original_type == "equation":
                    equation_chunks += 1
                elif original_type == "discarded":
                    discarded_chunks += 1
            else:
                text_chunks += 1

        searchable_chunks = max(0, total_chunks - discarded_chunks)
        text_ratio = text_chunks / searchable_chunks if searchable_chunks else 0.0
        multimodal_ratio = multimodal_chunks / searchable_chunks if searchable_chunks else 0.0
        page_coverage_ratio = (
            chunks_with_page_idx / searchable_chunks if searchable_chunks else 0.0
        )
        profile = {
            "doc_id": doc_id,
            "resolved_doc_id": internal_doc_id,
            "total_chunks": total_chunks,
            "searchable_chunks": searchable_chunks,
            "text_chunks": text_chunks,
            "multimodal_chunks": multimodal_chunks,
            "table_chunks": table_chunks,
            "image_chunks": image_chunks,
            "equation_chunks": equation_chunks,
            "discarded_chunks": discarded_chunks,
            "chunks_with_page_idx": chunks_with_page_idx,
            "unique_pages": len(unique_pages),
            "text_ratio": text_ratio,
            "multimodal_ratio": multimodal_ratio,
            "page_coverage_ratio": page_coverage_ratio,
            "sparse": searchable_chunks <= 10,
            "text_sparse": text_chunks <= 8,
            "visual_heavy": multimodal_chunks > max(text_chunks, 0),
        }
        self._document_profile_cache[cache_key] = profile
        return profile

    def doc_summary_text(self, doc_id: str) -> str:
        item = self.full_docs.get(doc_id)
        if isinstance(item, dict):
            return str(item.get("content") or "")
        return ""

    def docs_for_entity_name(self, entity_name: str, *, max_docs: int = 12) -> list[str]:
        payload = load_json(self.path("kv_store_entity_chunks.json")).get(entity_name)
        if not isinstance(payload, dict):
            return []
        chunk_ids = payload.get("chunk_ids")
        if not isinstance(chunk_ids, list):
            return []
        doc_ids: list[str] = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_by_id(str(chunk_id))
            if not isinstance(chunk, dict):
                continue
            doc_id = chunk.get("full_doc_id")
            if isinstance(doc_id, str):
                doc_ids.append(doc_id)
        return list(dict.fromkeys(doc_ids))[:max_docs]

    def discover_candidate_docs(
        self,
        query_text: str,
        *,
        top_k: int = 8,
        seed_entities: list[str] | None = None,
        exclude_doc_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_norm = " ".join(query_text.strip().split()).casefold()
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", query_norm)
            if token not in {"the", "and", "for", "with", "from", "what", "when", "where", "which"}
        }
        scores: dict[str, dict[str, Any]] = {}
        for doc_id, item in self.full_docs.items():
            if not isinstance(item, dict):
                continue
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue
            content = str(item.get("content") or "")
            content_norm = content.casefold()
            content_tokens = set(re.findall(r"[a-z0-9]{3,}", content_norm))
            overlap = len(query_tokens & content_tokens)
            title = str(item.get("file_path") or self.file_for_doc(doc_id) or "")
            title_bonus = 0.0
            if title and any(token in title.casefold() for token in query_tokens):
                title_bonus = 0.5
            if overlap <= 0 and title_bonus <= 0:
                continue
            scores[doc_id] = {
                "doc_id": doc_id,
                "file_path": title,
                "score": float(overlap) + title_bonus,
                "token_overlap": overlap,
                "entity_hits": 0,
            }
        for entity_name in seed_entities or []:
            for doc_id in self.docs_for_entity_name(entity_name):
                if exclude_doc_ids and doc_id in exclude_doc_ids:
                    continue
                record = scores.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "file_path": self.file_for_doc(doc_id) or "",
                        "score": 0.0,
                        "token_overlap": 0,
                        "entity_hits": 0,
                    },
                )
                record["score"] += 1.5
                record["entity_hits"] += 1
        ranked = sorted(scores.values(), key=lambda item: float(item["score"]), reverse=True)
        return ranked[:top_k]

    @staticmethod
    def is_noisy_entity(name: str) -> bool:
        normalized = name.strip().strip('"').lower()
        return any(term in normalized for term in NOISY_ENTITY_TERMS)
