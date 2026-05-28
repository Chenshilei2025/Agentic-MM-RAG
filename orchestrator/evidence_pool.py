"""Refreshable cross-query evidence pool.

The evidence board is a per-query collaboration surface.  This module keeps a
longer-lived candidate pool that can be refreshed, invalidated, and reused
across orchestration runs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from agentic_mm_rag.agent.types import QueryContext, SubagentResult
from agentic_mm_rag.tools.runtime.scoring import keywords


EvidencePoolStatus = Literal["active", "stale", "superseded", "rejected"]


@dataclass(slots=True)
class EvidencePoolItem:
    """Versioned evidence candidate stored across query runs."""

    evidence_id: str
    evidence: dict[str, Any]
    source_key: str
    query_key: str
    status: EvidencePoolStatus = "active"
    version: int = 1
    first_seen_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    last_refreshed_at: float = field(default_factory=time.time)
    ttl_seconds: int | None = None
    refresh_count: int = 0
    score: float = 0.0
    fused_score: float = 0.0
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return False
        current = time.time() if now is None else now
        return current - self.last_refreshed_at > self.ttl_seconds

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_evidence:
            data.pop("evidence", None)
        return data


class RefreshableEvidencePool:
    """Longer-lived evidence pool with explicit refresh and invalidation state."""

    def __init__(
        self,
        *,
        default_ttl_seconds: int | None = None,
        max_items: int | None = None,
    ) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.max_items = max_items
        self._items: dict[str, EvidencePoolItem] = {}
        self._source_index: dict[str, str] = {}

    def clear(self) -> None:
        self._items.clear()
        self._source_index.clear()

    def upsert_results(
        self,
        query: QueryContext,
        results: list[SubagentResult],
        *,
        ttl_seconds: int | None = None,
        now: float | None = None,
    ) -> list[EvidencePoolItem]:
        evidence_items: list[dict[str, Any]] = []
        for result in results:
            if not result.ok:
                continue
            evidence_items.extend(item for item in result.evidence if isinstance(item, dict))
            data_items = result.data.get("items") if isinstance(result.data, dict) else None
            if isinstance(data_items, list):
                evidence_items.extend(item for item in data_items if isinstance(item, dict))
        return self.upsert_many(
            evidence_items,
            query_key=self.query_key(query),
            ttl_seconds=ttl_seconds,
            now=now,
        )

    def upsert_many(
        self,
        evidence_items: list[dict[str, Any]],
        *,
        query_key: str,
        ttl_seconds: int | None = None,
        now: float | None = None,
    ) -> list[EvidencePoolItem]:
        current = time.time() if now is None else now
        updated: list[EvidencePoolItem] = []
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        for evidence in evidence_items:
            evidence_id = self.evidence_id(evidence)
            if not evidence_id:
                continue
            source_key = self.source_key(evidence)
            existing = self._items.get(evidence_id)
            if existing is None:
                prior_id = self._source_index.get(source_key)
                if prior_id and prior_id in self._items and prior_id != evidence_id:
                    self._items[prior_id].status = "superseded"
                item = EvidencePoolItem(
                    evidence_id=evidence_id,
                    evidence=dict(evidence),
                    source_key=source_key,
                    query_key=query_key,
                    first_seen_at=current,
                    last_seen_at=current,
                    last_refreshed_at=current,
                    ttl_seconds=ttl,
                    score=float(evidence.get("score", 0.0) or 0.0),
                    fused_score=float(evidence.get("fused_score", evidence.get("score", 0.0)) or 0.0),
                    quality=dict(evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}),
                    provenance=dict(
                        evidence.get("provenance", {}) if isinstance(evidence.get("provenance"), dict) else {}
                    ),
                    metadata=dict(evidence.get("metadata", {}) if isinstance(evidence.get("metadata"), dict) else {}),
                )
                self._items[evidence_id] = item
            else:
                if existing.evidence != evidence:
                    existing.version += 1
                existing.evidence = dict(evidence)
                existing.source_key = source_key
                existing.query_key = query_key
                existing.status = "active"
                existing.last_seen_at = current
                existing.last_refreshed_at = current
                existing.ttl_seconds = ttl
                existing.refresh_count += 1
                existing.score = float(evidence.get("score", 0.0) or 0.0)
                existing.fused_score = float(evidence.get("fused_score", evidence.get("score", 0.0)) or 0.0)
                existing.quality = dict(
                    evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}
                )
                existing.provenance = dict(
                    evidence.get("provenance", {}) if isinstance(evidence.get("provenance"), dict) else {}
                )
                existing.metadata = dict(
                    evidence.get("metadata", {}) if isinstance(evidence.get("metadata"), dict) else {}
                )
                item = existing
            self._source_index[source_key] = evidence_id
            updated.append(item)
        self.evict()
        return updated

    def mark_stale(
        self,
        predicate: Callable[[EvidencePoolItem], bool] | None = None,
        *,
        reason: str | None = None,
    ) -> int:
        count = 0
        for item in self._items.values():
            if item.status != "active":
                continue
            if predicate is not None and not predicate(item):
                continue
            item.status = "stale"
            item.metadata["stale_reason"] = reason or "manual"
            count += 1
        return count

    def mark_rejected(
        self,
        predicate: Callable[[EvidencePoolItem], bool],
        *,
        reason: str | None = None,
    ) -> int:
        count = 0
        for item in self._items.values():
            if not predicate(item):
                continue
            item.status = "rejected"
            item.metadata["rejected_reason"] = reason or "manual"
            count += 1
        return count

    def refresh_expired(self, *, now: float | None = None) -> int:
        count = 0
        current = time.time() if now is None else now
        for item in self._items.values():
            if item.status == "active" and item.is_expired(now=current):
                item.status = "stale"
                item.metadata["stale_reason"] = "ttl_expired"
                count += 1
        return count

    def candidates(
        self,
        query: QueryContext | str | None = None,
        *,
        include_stale: bool = False,
        limit: int | None = None,
        min_keyword_overlap: int = 0,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        self.refresh_expired(now=now)
        query_text = query.query_text if isinstance(query, QueryContext) else str(query or "")
        query_terms = keywords(query_text) if query_text else set()
        records: list[tuple[float, EvidencePoolItem]] = []
        for item in self._items.values():
            if item.status == "active":
                pass
            elif include_stale and item.status == "stale":
                pass
            else:
                continue
            overlap = 0
            if query_terms:
                content = str(item.evidence.get("content") or item.evidence.get("text") or "")
                overlap = len(query_terms & keywords(content))
                if overlap < min_keyword_overlap:
                    continue
            score = max(item.fused_score, item.score) + min(0.5, overlap * 0.08)
            if item.status == "stale":
                score *= 0.75
            records.append((score, item))
        records.sort(key=lambda record: record[0], reverse=True)
        selected = records[:limit] if limit is not None else records
        evidence_items: list[dict[str, Any]] = []
        for score, item in selected:
            evidence = dict(item.evidence)
            metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {}
            evidence["metadata"] = {
                **metadata,
                "evidence_pool": {
                    "status": item.status,
                    "version": item.version,
                    "source_key": item.source_key,
                    "query_key": item.query_key,
                    "refresh_count": item.refresh_count,
                },
            }
            evidence.setdefault("score", item.score)
            evidence["pool_score"] = score
            evidence_items.append(evidence)
        return evidence_items

    def evict(self) -> int:
        removed = 0
        if self.max_items is None or len(self._items) <= self.max_items:
            return removed
        ranked = sorted(
            self._items.values(),
            key=lambda item: (
                item.status == "active",
                max(item.fused_score, item.score),
                item.last_seen_at,
            ),
            reverse=True,
        )
        keep = {item.evidence_id for item in ranked[: self.max_items]}
        for evidence_id in list(self._items):
            if evidence_id in keep:
                continue
            item = self._items.pop(evidence_id)
            if self._source_index.get(item.source_key) == evidence_id:
                self._source_index.pop(item.source_key, None)
            removed += 1
        return removed

    def snapshot(self, *, include_evidence: bool = True, refresh_expired: bool = False, now: float | None = None) -> dict[str, Any]:
        if refresh_expired:
            self.refresh_expired(now=now)
        status_counts = {"active": 0, "stale": 0, "superseded": 0, "rejected": 0}
        for item in self._items.values():
            status_counts[item.status] += 1
        return {
            "count": len(self._items),
            "status_counts": status_counts,
            "default_ttl_seconds": self.default_ttl_seconds,
            "max_items": self.max_items,
            "items": [
                item.to_dict(include_evidence=include_evidence)
                for item in sorted(
                    self._items.values(),
                    key=lambda item: (item.status != "active", -max(item.fused_score, item.score), item.evidence_id),
                )
            ],
        }

    @staticmethod
    def query_key(query: QueryContext | str) -> str:
        text = query.query_text if isinstance(query, QueryContext) else str(query)
        return " ".join(text.casefold().split())

    @staticmethod
    def evidence_id(evidence: dict[str, Any]) -> str:
        return str(evidence.get("id") or evidence.get("evidence_id") or "").strip()

    @classmethod
    def source_key(cls, evidence: dict[str, Any]) -> str:
        locator = evidence.get("locator") if isinstance(evidence.get("locator"), dict) else {}
        source_type = str(evidence.get("source_type") or "")
        if source_type == "doc":
            doc_id = locator.get("doc_id") or evidence.get("source_id") or ""
            page = locator.get("page_idx")
            return f"doc:{doc_id}:page:{page}"
        if source_type == "video":
            video_id = locator.get("video_id") or evidence.get("source_id") or ""
            segment_id = locator.get("segment_id") or cls.evidence_id(evidence)
            start = locator.get("start_time") or locator.get("raw_time") or ""
            return f"video:{video_id}:segment:{segment_id}:time:{start}"
        return f"unknown:{cls.evidence_id(evidence)}"
