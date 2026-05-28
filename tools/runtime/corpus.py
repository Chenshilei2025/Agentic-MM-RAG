"""Shared corpus adapter interfaces for long documents and long videos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_mm_rag.config import DEFAULT_PATHS
from agentic_mm_rag.tools.runtime.stores.doc_store import DocRAGStore
from agentic_mm_rag.tools.runtime.stores.video_store import VideoRAGStore


_CORPUS_ALIASES = {
    "doc": "doc",
    "docs": "doc",
    "document": "doc",
    "documents": "doc",
    "video": "video",
    "videos": "video",
}


class CorpusAdapter(ABC):
    """Common interface implemented by document and video corpora."""

    @property
    @abstractmethod
    def corpus_type(self) -> str:
        ...

    @abstractmethod
    def manifest(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def iter_chunks(self) -> list[tuple[str, dict[str, Any]]]:
        ...


@dataclass(slots=True)
class DocumentCorpusAdapter(CorpusAdapter):
    """Corpus adapter for processed long-document storage."""

    store: DocRAGStore

    @property
    def corpus_type(self) -> str:
        return "doc"

    def manifest(self) -> dict[str, Any]:
        return self.store.manifest()

    def chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        return self.store.chunk_by_id(chunk_id)

    def iter_chunks(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (chunk_id, chunk)
            for chunk_id, chunk in self.store.text_chunks.items()
            if isinstance(chunk, dict)
        ]


@dataclass(slots=True)
class VideoCorpusAdapter(CorpusAdapter):
    """Corpus adapter for processed long-video storage."""

    store: VideoRAGStore

    @property
    def corpus_type(self) -> str:
        return "video"

    def manifest(self) -> dict[str, Any]:
        return self.store.manifest()

    def chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        return self.store.chunk_by_id(chunk_id)

    def iter_chunks(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (chunk_id, chunk)
            for chunk_id, chunk in self.store.text_chunks.items()
            if isinstance(chunk, dict)
        ]


def build_corpus_adapter(
    corpus_type: str,
    *,
    root: str | Path | None = None,
) -> CorpusAdapter:
    """Construct a corpus adapter from the target corpus type."""

    normalized = _CORPUS_ALIASES.get(corpus_type.lower(), corpus_type.lower())

    if normalized == "doc":
        store = DocRAGStore(Path(root) if root is not None else DEFAULT_PATHS.doc_rag_dir)  # type: ignore[arg-type]
        return DocumentCorpusAdapter(store=store)
    if normalized == "video":
        store = VideoRAGStore(Path(root) if root is not None else DEFAULT_PATHS.video_rag_dir)  # type: ignore[arg-type]
        return VideoCorpusAdapter(store=store)
    raise ValueError(f"unsupported corpus_type: {corpus_type}")
