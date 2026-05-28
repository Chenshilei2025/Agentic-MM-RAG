"""Adapter for processed VideoRAG storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from agentic_mm_rag.config import DEFAULT_PATHS
from agentic_mm_rag.tools.runtime.stores.graph import GraphMLStore
from agentic_mm_rag.tools.runtime.stores.json_utils import load_json, safe_count_pattern
from agentic_mm_rag.tools.runtime.stores.vector import JsonVectorStore


def seconds_to_hms(seconds: int | float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


@dataclass(slots=True)
class VideoRAGStore:
    root: Path = DEFAULT_PATHS.video_rag_dir
    chunks_vdb: JsonVectorStore = field(init=False)
    entities_vdb: JsonVectorStore = field(init=False)
    segment_feature_vdb: JsonVectorStore = field(init=False)
    graph: GraphMLStore = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.chunks_vdb = JsonVectorStore(self.root / "vdb_chunks.json")
        self.entities_vdb = JsonVectorStore(self.root / "vdb_entities.json")
        self.segment_feature_vdb = JsonVectorStore(
            self.root / "vdb_video_segment_feature.json"
        )
        self.graph = GraphMLStore(self.root / "graph_chunk_entity_relation.graphml")

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def video_paths(self) -> dict[str, str]:
        return load_json(self.path("kv_store_video_path.json"))

    @property
    def video_segments(self) -> dict[str, Any]:
        return load_json(self.path("kv_store_video_segments.json"))

    @property
    def text_chunks(self) -> dict[str, Any]:
        return load_json(self.path("kv_store_text_chunks.json"))

    def manifest(self) -> dict[str, Any]:
        segments = self.video_segments
        segment_count = sum(len(v) for v in segments.values() if isinstance(v, dict))
        return {
            "root": str(self.root),
            "videos": len(self.video_paths),
            "segments": segment_count,
            "text_chunks": len(self.text_chunks),
            "chunk_vectors": safe_count_pattern(self.path("vdb_chunks.json"), '"__id__"'),
            "entity_vectors_estimate": safe_count_pattern(
                self.path("vdb_entities.json"), '"__id__"'
            ),
            "video_segment_vectors": safe_count_pattern(
                self.path("vdb_video_segment_feature.json"), '"__id__"'
            ),
        }

    def chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        item = self.text_chunks.get(chunk_id)
        return item if isinstance(item, dict) else None

    def segment_by_id(self, segment_id: str) -> dict[str, Any] | None:
        if "_" not in segment_id:
            return None
        video_id, index = segment_id.rsplit("_", 1)
        video = self.video_segments.get(video_id)
        if not isinstance(video, dict):
            return None
        item = video.get(index)
        if not isinstance(item, dict):
            return None
        return item

    def segment_locator(self, segment_id: str) -> tuple[str | None, str | None, str | None]:
        item = self.segment_by_id(segment_id)
        if not item:
            return None, None, None
        raw_time = item.get("time")
        if not isinstance(raw_time, str) or "-" not in raw_time:
            return raw_time, None, None
        start, end = raw_time.split("-", 1)
        try:
            return raw_time, seconds_to_hms(float(start)), seconds_to_hms(float(end))
        except ValueError:
            return raw_time, None, None

    def expand_segment_ids(self, segment_id: str, window: int) -> list[str]:
        if "_" not in segment_id:
            return []
        video_id, index_s = segment_id.rsplit("_", 1)
        try:
            index = int(index_s)
        except ValueError:
            return []
        video = self.video_segments.get(video_id, {})
        if not isinstance(video, dict):
            return []
        result: list[str] = []
        for i in range(max(0, index - window), index + window + 1):
            key = str(i)
            if key in video:
                result.append(f"{video_id}_{key}")
        return result

    @staticmethod
    def _keywords(text: str) -> set[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
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
            "who",
            "their",
            "them",
            "they",
            "have",
            "has",
            "into",
            "more",
            "based",
            "process",
            "explore",
            "focus",
            "focusing",
            "mechanism",
            "mechanisms",
            "behavior",
            "behaviors",
            "interaction",
            "interactions",
        }
        canonical = {
            "drivers": "driver",
            "driving": "driver",
            "drive": "driver",
            "blockers": "blocker",
        }
        return {
            canonical.get(token, token)
            for token in re.findall(r"[A-Za-z0-9_]{3,}", text.casefold())
            if token not in stopwords
        }

    @staticmethod
    def _important_keywords(text: str) -> set[str]:
        generic = {
            "action",
            "animal",
            "animals",
            "cause",
            "cues",
            "designated",
            "dictates",
            "engage",
            "environment",
            "fluid",
            "individual",
            "leader",
            "process",
            "prompts",
            "roles",
            "strategy",
            "video",
        }
        return {token for token in VideoRAGStore._keywords(text) if token not in generic}

    @staticmethod
    def _exact_detail_terms(text: str) -> list[str]:
        """Terms that can be matched lexically for optional exact-detail retrieval."""
        terms: list[str] = []
        terms.extend(match.strip() for match in re.findall(r"['\"]([^'\"]{3,80})['\"]", text))
        terms.extend(match.strip() for match in re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,5}\b", text))
        terms.extend(match.strip() for match in re.findall(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b", text))
        cleaned: list[str] = []
        for term in terms:
            term = re.sub(r"\s+", " ", term).strip(" ?.,:;")
            if len(term) >= 3 and term.casefold() not in {"the video", "chapter"}:
                cleaned.append(term)
        return list(dict.fromkeys(cleaned))[:24]

    @classmethod
    def _exact_detail_bonus(cls, query_text: str, content: str) -> float:
        terms = cls._exact_detail_terms(query_text)
        if not terms:
            return 0.0
        lower = content.casefold()
        bonus = 0.0
        for term in terms:
            term_lower = term.casefold()
            if term_lower in lower:
                bonus += 0.75 if " " in term_lower or "-" in term_lower else 0.45
                continue
            term_tokens = cls._keywords(term)
            content_tokens = cls._keywords(content)
            if term_tokens and term_tokens <= content_tokens:
                bonus += 0.45
        return min(2.25, bonus)

    def lexical_chunk_scores(self, query_text: str, *, exact_detail: bool = False) -> dict[str, float]:
        query_tokens = self._keywords(query_text)
        if not query_tokens:
            return {}
        important_tokens = self._important_keywords(query_text)
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.text_chunks.items():
            if not isinstance(chunk, dict):
                continue
            content = str(chunk.get("content") or "")
            content_tokens = self._keywords(content)
            if not content_tokens:
                continue
            overlap = len(query_tokens & content_tokens)
            important_overlap = len(important_tokens & content_tokens)
            exact_bonus = self._exact_detail_bonus(query_text, content) if exact_detail else 0.0
            if overlap or important_overlap or exact_bonus:
                base = overlap / max(len(query_tokens), 1)
                scores[str(chunk_id)] = base + min(1.25, important_overlap * 0.35) + exact_bonus
        return scores

    def lexical_segment_scores(self, query_text: str, *, exact_detail: bool = False) -> dict[str, float]:
        query_tokens = self._keywords(query_text)
        if not query_tokens:
            return {}
        important_tokens = self._important_keywords(query_text)
        scores: dict[str, float] = {}
        for video_id, segments in self.video_segments.items():
            if not isinstance(segments, dict):
                continue
            for index, segment in segments.items():
                if not isinstance(segment, dict):
                    continue
                content = f"{segment.get('content') or ''}\n{segment.get('transcript') or ''}"
                content_tokens = self._keywords(content)
                if not content_tokens:
                    continue
                overlap = len(query_tokens & content_tokens)
                important_overlap = len(important_tokens & content_tokens)
                exact_bonus = self._exact_detail_bonus(query_text, content) if exact_detail else 0.0
                if overlap or important_overlap or exact_bonus:
                    base = overlap / max(len(query_tokens), 1)
                    scores[f"{video_id}_{index}"] = base + min(1.25, important_overlap * 0.35) + exact_bonus
        return scores
