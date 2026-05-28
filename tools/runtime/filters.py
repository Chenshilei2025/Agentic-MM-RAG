"""Document retrieval filters shared by retrieval backends."""

from __future__ import annotations

from typing import Any, Protocol


NOISY_DOC_CHUNK_TYPES = {"discarded", "unknown"}


class DocumentIdResolver(Protocol):
    def resolve_doc_id(self, doc_id: str | None) -> str | None:
        ...


def doc_chunk_allowed(
    chunk: dict[str, Any],
    *,
    allow_docs: set[str] | None = None,
    include_multimodal: bool = True,
    modalities: set[str] | None = None,
) -> bool:
    """Return whether a processed document chunk should be exposed as evidence."""

    if allow_docs is not None and chunk.get("full_doc_id") not in allow_docs:
        return False
    original_type = str(chunk.get("original_type") or "").lower()
    if original_type in NOISY_DOC_CHUNK_TYPES:
        return False
    if not include_multimodal and chunk.get("is_multimodal"):
        return False
    if modalities is not None:
        if not chunk.get("is_multimodal"):
            return False
        if original_type not in modalities:
            return False
    return True


def resolve_doc_allowlist(
    store: DocumentIdResolver,
    doc_ids: list[str] | None,
) -> set[str] | None:
    """Resolve external document ids or filenames to internal document ids."""

    if not doc_ids:
        return None
    resolved = {
        internal_doc_id
        for doc_id in doc_ids
        for internal_doc_id in [store.resolve_doc_id(doc_id)]
        if internal_doc_id
    }
    return resolved or set()
