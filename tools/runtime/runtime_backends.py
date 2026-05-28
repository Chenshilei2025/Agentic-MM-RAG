"""Runtime backend configuration and execution protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.agent.prompts import DOC_CHUNK_RUNTIME_PROMPT_TEMPLATE
from agentic_mm_rag.tools.runtime.backends import DocumentRetrievalBackend, build_doc_backend


RuntimeBackendKind = str


@dataclass(slots=True)
class RuntimeBackendConfig:
    """Runtime execution policy layered above a corpus backend."""

    backend_type: RuntimeBackendKind = "doc_chunks"
    top_k: int = 8
    chunk_window: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeQueryRequest:
    """Minimal query contract for batch runtime execution."""

    query_text: str
    sample_id: str | None = None
    source_id: str | None = None


class RuntimeQueryBackend(Protocol):
    """Pluggable runtime execution backend for batch evaluation."""

    backend_type: RuntimeBackendKind
    corpus_type: str

    async def answer(self, request: RuntimeQueryRequest) -> ToolResponse:
        ...


@dataclass(slots=True)
class DocChunkRuntimeBackend:
    """Stable emergency runtime backend over per-document text chunks only."""

    backend: DocumentRetrievalBackend
    config: RuntimeBackendConfig
    backend_type: RuntimeBackendKind = "doc_chunks"
    corpus_type: str = "doc"

    def _build_file_to_internal_doc_id(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for internal_doc_id, item in self.backend.store.doc_status.items():
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file_path") or "").strip()
            if not file_path:
                continue
            mapping[file_path] = str(internal_doc_id)
            mapping[Path(file_path).name] = str(internal_doc_id)
            normalized = file_path.replace("_origin", "")
            mapping[normalized] = str(internal_doc_id)
            mapping[Path(normalized).name] = str(internal_doc_id)
        return mapping

    def _resolve_internal_doc_id(self, sample_doc_id: str) -> str:
        mapping = self.config.metadata.get("file_to_internal_doc_id")
        if not isinstance(mapping, dict) or not mapping:
            mapping = self._build_file_to_internal_doc_id()
            self.config.metadata["file_to_internal_doc_id"] = mapping
        return str(mapping.get(sample_doc_id, sample_doc_id))

    def _doc_chunks_for_doc(self, doc_id: str) -> list[tuple[str, dict[str, Any]]]:
        items: list[tuple[str, dict[str, Any]]] = []
        for chunk_id, chunk in self.backend.store.text_chunks.items():
            if isinstance(chunk, dict) and chunk.get("full_doc_id") == doc_id:
                items.append((chunk_id, chunk))
        items.sort(key=lambda item: int(item[1].get("chunk_order_index", 0)))
        return items

    @staticmethod
    def _keywords(text: str) -> set[str]:
        import re

        stopwords = {
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
        }
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
            if token not in stopwords
        }

    def _select_doc_chunks(
        self,
        *,
        question: str,
        doc_chunks: list[tuple[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        query_terms = self._keywords(question)
        scored: list[tuple[float, int]] = []
        for idx, (_chunk_id, chunk) in enumerate(doc_chunks):
            content = str(chunk.get("content") or "")
            overlap = len(query_terms & self._keywords(content))
            if overlap > 0:
                scored.append((float(overlap), idx))
        if not scored:
            seed_indices = list(range(min(self.config.top_k, len(doc_chunks))))
        else:
            scored.sort(key=lambda item: (-item[0], item[1]))
            seed_indices = [idx for _score, idx in scored[: self.config.top_k]]
        selected: set[int] = set()
        for idx in seed_indices:
            start = max(0, idx - self.config.chunk_window)
            end = min(len(doc_chunks), idx + self.config.chunk_window + 1)
            selected.update(range(start, end))
        return [doc_chunks[idx][1] for idx in sorted(selected)]

    def _build_doc_prompt(
        self,
        *,
        question: str,
        doc_id: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        context_parts: list[str] = []
        for chunk in chunks:
            chunk_idx = chunk.get("chunk_order_index")
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            context_parts.append(f"[chunk {chunk_idx}] {content}")
        context = "\n\n".join(context_parts)
        return DOC_CHUNK_RUNTIME_PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            file_path=self.backend.store.file_for_doc(doc_id) or "",
            question=question,
            context=context,
        )

    async def answer(self, request: RuntimeQueryRequest) -> ToolResponse:
        if not request.source_id:
            return ToolResponse(
                ok=False,
                tool="doc_chunk_runtime_answer",
                error="source_id is required for doc_chunks runtime backend",
            )
        internal_doc_id = self._resolve_internal_doc_id(request.source_id)
        doc_chunks = self._doc_chunks_for_doc(internal_doc_id)
        if not doc_chunks:
            return ToolResponse(
                ok=True,
                tool="doc_chunk_runtime_answer",
                data={
                    "answer": "Not answerable",
                    "backend_type": self.backend_type,
                    "internal_doc_id": internal_doc_id,
                    "selected_chunk_count": 0,
                },
            )
        selected_chunks = self._select_doc_chunks(
            question=request.query_text,
            doc_chunks=doc_chunks,
        )
        prompt = self._build_doc_prompt(
            question=request.query_text,
            doc_id=internal_doc_id,
            chunks=selected_chunks,
        )
        answer = await self.config.metadata["llm_model_func"](prompt, temperature=0)
        return ToolResponse(
            ok=True,
            tool="doc_chunk_runtime_answer",
            data={
                "answer": str(answer),
                "backend_type": self.backend_type,
                "internal_doc_id": internal_doc_id,
                "selected_chunk_count": len(selected_chunks),
            },
        )


def build_runtime_query_backend(
    *,
    doc_root: str | Path,
    config: RuntimeBackendConfig,
    llm_model_func: Any | None = None,
) -> RuntimeQueryBackend:
    backend = build_doc_backend(root=doc_root)
    config.metadata.setdefault("doc_root", str(doc_root))
    if llm_model_func is not None:
        config.metadata["llm_model_func"] = llm_model_func
    if config.backend_type == "doc_chunks":
        if "llm_model_func" not in config.metadata:
            raise RuntimeError("doc_chunks runtime backend requires llm_model_func")
        return DocChunkRuntimeBackend(backend=backend, config=config)
    raise ValueError(f"unsupported runtime backend type: {config.backend_type}")
