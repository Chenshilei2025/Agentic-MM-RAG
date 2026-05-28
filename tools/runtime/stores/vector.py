"""Small vector-store wrapper around NanoVectorDB JSON exports."""

from __future__ import annotations

from array import array
import base64
import binascii
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from agentic_mm_rag.tools.runtime.stores.json_utils import load_json


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any]


class JsonVectorStore:
    """Read-only cosine search over a NanoVectorDB JSON file.

    This is intended for small/medium processed stores. Very large stores such
    as document entity/relationship VDBs should be accessed through a future
    approximate backend or prefiltered cache.
    """

    def __init__(self, path: Path):
        self.path = path
        self.cache_meta_path = path.with_suffix(path.suffix + ".cache.json")
        self.cache_matrix_path = path.with_suffix(path.suffix + ".matrix.f32")
        self.embedding_dim: int | None = None
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectors: list[list[float]] | None = None
        self._matrix: array | None = None
        self._matrix_rows: int = 0

    @property
    def loaded(self) -> bool:
        return self._vectors is not None or self._matrix is not None

    @staticmethod
    def _decode_matrix(raw_matrix: Any, rows: int, dim: int) -> array | list[list[float]] | None:
        if isinstance(raw_matrix, list):
            return raw_matrix
        if not isinstance(raw_matrix, str) or rows <= 0 or dim <= 0:
            return None
        try:
            values = array("f")
            values.frombytes(base64.b64decode(raw_matrix))
            if len(values) != rows * dim:
                return None
            return values
        except Exception:
            return None

    def load(self) -> None:
        if self.loaded:
            return
        if self._load_cache_if_fresh():
            return
        raw = load_json(self.path)
        self.embedding_dim = int(raw.get("embedding_dim", 0))
        data = raw.get("data", [])
        matrix = self._decode_matrix(raw.get("matrix"), len(data), self.embedding_dim or 0)
        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        if isinstance(matrix, array):
            for item in data:
                ids.append(item["__id__"])
                metadata.append({k: v for k, v in item.items() if k not in {"__vector__", "vector"}})
            self._ids = ids
            self._metadata = metadata
            self._matrix = matrix
            self._matrix_rows = len(ids)
            self._vectors = None
            return
        for idx, item in enumerate(data):
            vector = item.get("__vector__")
            if vector is None and isinstance(matrix, list) and idx < len(matrix):
                vector = matrix[idx]
            if vector is None:
                continue
            ids.append(item["__id__"])
            metadata.append({k: v for k, v in item.items() if k not in {"__vector__", "vector"}})
            values = [float(x) for x in vector]
            norm = math.sqrt(sum(x * x for x in values)) or 1.0
            vectors.append([x / norm for x in values])
        self._ids = ids
        self._metadata = metadata
        self._vectors = vectors

    def _load_cache_if_fresh(self) -> bool:
        if not self.cache_meta_path.exists() or not self.cache_matrix_path.exists():
            return False
        try:
            source_mtime = self.path.stat().st_mtime
            meta = json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
            if float(meta.get("source_mtime", -1)) != source_mtime:
                return False
            dim = int(meta["embedding_dim"])
            ids = list(meta["ids"])
            metadata = list(meta["metadata"])
            values = array("f")
            with self.cache_matrix_path.open("rb") as fh:
                values.fromfile(fh, self.cache_matrix_path.stat().st_size // values.itemsize)
            if len(values) != len(ids) * dim:
                return False
        except Exception:
            return False
        self.embedding_dim = dim
        self._ids = [str(x) for x in ids]
        self._metadata = [dict(x) for x in metadata]
        self._matrix = values
        self._matrix_rows = len(self._ids)
        self._vectors = None
        return True

    def build_cache(self, *, force: bool = False) -> dict[str, Any]:
        if not force and self._load_cache_if_fresh():
            return {
                "status": "exists",
                "path": str(self.path),
                "rows": len(self._ids),
                "embedding_dim": self.embedding_dim,
            }
        try:
            return self._build_cache_streaming()
        except ImportError:
            pass
        raw = load_json(self.path)
        dim = int(raw.get("embedding_dim", 0))
        data = raw.get("data", [])
        matrix = self._decode_matrix(raw.get("matrix"), len(data), dim)
        if not isinstance(matrix, array):
            values = array("f")
            ids_for_values = []
            metadata = []
            for item in data:
                vector = item.get("__vector__")
                if vector is None:
                    continue
                ids_for_values.append(item["__id__"])
                metadata.append({k: v for k, v in item.items() if k not in {"__vector__", "vector"}})
                values.extend(float(x) for x in vector)
            ids = ids_for_values
        else:
            values = matrix
            ids = [item["__id__"] for item in data]
            metadata = [
                {k: v for k, v in item.items() if k not in {"__vector__", "vector"}}
                for item in data
            ]
        if dim <= 0 or len(values) != len(ids) * dim:
            raise RuntimeError(
                f"invalid vector cache dimensions for {self.path}: "
                f"values={len(values)} ids={len(ids)} dim={dim}"
            )
        self.cache_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_matrix_path.open("wb") as fh:
            values.tofile(fh)
        meta = {
            "source_path": str(self.path),
            "source_mtime": self.path.stat().st_mtime,
            "embedding_dim": dim,
            "rows": len(ids),
            "ids": ids,
            "metadata": metadata,
        }
        self.cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.embedding_dim = dim
        self._ids = [str(x) for x in ids]
        self._metadata = [dict(x) for x in metadata]
        self._matrix = values
        self._matrix_rows = len(ids)
        self._vectors = None
        return {
            "status": "built",
            "path": str(self.path),
            "rows": len(ids),
            "embedding_dim": dim,
            "matrix_cache": str(self.cache_matrix_path),
            "meta_cache": str(self.cache_meta_path),
        }

    def _build_cache_streaming(self) -> dict[str, Any]:
        import ijson

        self.cache_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        dim: int | None = None
        rows = 0
        with self.path.open("rb") as fh:
            for prefix, event, value in ijson.parse(fh):
                if prefix == "embedding_dim" and event == "number":
                    dim = int(value)
                    break
        if dim is None or dim <= 0:
            raise RuntimeError(f"missing embedding_dim in {self.path}")
        has_matrix = False
        with self.path.open("rb") as fh:
            for prefix, event, _value in ijson.parse(fh):
                if prefix == "" and event == "map_key" and _value == "matrix":
                    has_matrix = True
                    break
        with self.path.open("rb") as fh:
            for item in ijson.items(fh, "data.item"):
                item_id = item["__id__"]
                ids.append(item_id)
                metadata.append(self._compact_metadata(item))
        if has_matrix:
            ids, metadata = self._stream_metadata_for_matrix()
            rows = len(ids)
            self._stream_decode_matrix(dim=dim, rows=rows)
            return self._write_cache_meta(dim=dim, ids=ids, metadata=metadata, rows=rows)

        ids = []
        metadata = []
        with self.path.open("rb") as fh, self.cache_matrix_path.open("wb") as matrix_out:
            for item in ijson.items(fh, "data.item"):
                vector = item.get("__vector__") or item.get("vector")
                if not isinstance(vector, list):
                    continue
                if vector is None:
                    continue
                item_id = item["__id__"]
                ids.append(item_id)
                metadata.append(self._compact_metadata(item))
                values = array("f", (float(x) for x in vector))
                if len(values) != dim:
                    raise RuntimeError(
                        f"vector dim {len(values)} != embedding_dim {dim} for {item_id}"
                    )
                values.tofile(matrix_out)
                rows += 1
        return self._write_cache_meta(dim=dim, ids=ids, metadata=metadata, rows=rows)

    def _stream_metadata_for_matrix(self) -> tuple[list[str], list[dict[str, Any]]]:
        import ijson

        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        keep = {
            "__id__",
            "__created_at__",
            "entity_name",
            "relation_name",
            "src_id",
            "tgt_id",
            "source_id",
            "full_doc_id",
            "file_path",
        }
        with self.path.open("rb") as fh:
            for prefix, event, value in ijson.parse(fh):
                if prefix == "data.item" and event == "start_map":
                    current = {}
                    continue
                if prefix == "data.item" and event == "end_map":
                    if current and "__id__" in current:
                        ids.append(str(current["__id__"]))
                        metadata.append(dict(current))
                    current = None
                    continue
                if current is None or not prefix.startswith("data.item."):
                    continue
                key = prefix.rsplit(".", 1)[-1]
                if key in keep and event in {"string", "number", "boolean", "null"}:
                    current[key] = value
        return ids, metadata

    def _write_cache_meta(
        self,
        *,
        dim: int,
        ids: list[str],
        metadata: list[dict[str, Any]],
        rows: int,
    ) -> dict[str, Any]:
        meta = {
            "source_path": str(self.path),
            "source_mtime": self.path.stat().st_mtime,
            "embedding_dim": dim,
            "rows": rows,
            "ids": ids,
            "metadata": metadata,
        }
        self.cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.embedding_dim = dim
        self._ids = [str(x) for x in ids]
        self._metadata = [dict(x) for x in metadata]
        self._matrix = None
        self._matrix_rows = rows
        self._vectors = None
        return {
            "status": "built",
            "path": str(self.path),
            "rows": rows,
            "embedding_dim": dim,
            "matrix_cache": str(self.cache_matrix_path),
            "meta_cache": str(self.cache_meta_path),
        }

    def _stream_decode_matrix(self, *, dim: int, rows: int) -> None:
        marker = b'"matrix"'
        chunk_size = 8 * 1024 * 1024
        tail = b""
        found = False
        in_string = False
        escape = False
        b64_buffer = bytearray()
        decoded_bytes = 0
        expected_bytes = rows * dim * 4
        with self.path.open("rb") as src, self.cache_matrix_path.open("wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                data = tail + chunk
                start = 0
                if not found:
                    idx = data.find(marker)
                    if idx < 0:
                        tail = data[-len(marker):]
                        continue
                    colon = data.find(b":", idx + len(marker))
                    quote = data.find(b'"', colon + 1)
                    if colon < 0 or quote < 0:
                        tail = data[idx:]
                        continue
                    found = True
                    in_string = True
                    start = quote + 1
                for byte in data[start:]:
                    if not in_string:
                        break
                    if escape:
                        b64_buffer.append(byte)
                        escape = False
                        continue
                    if byte == 92:  # backslash
                        escape = True
                        continue
                    if byte == 34:  # quote
                        in_string = False
                        break
                    b64_buffer.append(byte)
                    if len(b64_buffer) >= 4 * 1024 * 1024:
                        keep = len(b64_buffer) % 4
                        decode_part = b64_buffer[:-keep] if keep else b64_buffer
                        if decode_part:
                            raw = base64.b64decode(decode_part)
                            dst.write(raw)
                            decoded_bytes += len(raw)
                            b64_buffer = bytearray(b64_buffer[-keep:] if keep else b"")
                tail = b""
                if found and not in_string:
                    break
            if b64_buffer:
                raw = base64.b64decode(b64_buffer)
                dst.write(raw)
                decoded_bytes += len(raw)
        if decoded_bytes != expected_bytes:
            raise RuntimeError(
                f"decoded matrix bytes {decoded_bytes} != expected {expected_bytes} "
                f"for {self.path}"
            )

    @staticmethod
    def _compact_metadata(item: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "__id__",
            "__created_at__",
            "entity_name",
            "relation_name",
            "src_id",
            "tgt_id",
            "source_id",
            "full_doc_id",
            "file_path",
        }
        return {k: v for k, v in item.items() if k in keep}

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        min_score: float | None = None,
        ids_allowlist: set[str] | None = None,
    ) -> list[VectorHit]:
        self.load()
        if not self._vectors and self._matrix is None:
            return []

        q = [float(x) for x in query_vector]
        if self.embedding_dim and len(q) != self.embedding_dim:
            raise ValueError(
                f"query vector dim {len(q)} != store dim {self.embedding_dim}"
            )
        q_norm = math.sqrt(sum(x * x for x in q))
        if q_norm == 0:
            raise ValueError("query_vector must be non-zero")
        q = [x / q_norm for x in q]

        scored: list[tuple[float, int]] = []
        if self._matrix is not None:
            assert self.embedding_dim is not None
            dim = self.embedding_dim
            matrix = self._matrix
            for idx, item_id in enumerate(self._ids):
                if ids_allowlist is not None and item_id not in ids_allowlist:
                    continue
                offset = idx * dim
                dot = 0.0
                row_norm_sq = 0.0
                for j, qv in enumerate(q):
                    value = float(matrix[offset + j])
                    dot += qv * value
                    row_norm_sq += value * value
                row_norm = math.sqrt(row_norm_sq) or 1.0
                score = dot / row_norm
                if min_score is not None and score < min_score:
                    continue
                scored.append((score, idx))
        else:
            assert self._vectors is not None
            for idx, (item_id, vector) in enumerate(zip(self._ids, self._vectors)):
                if ids_allowlist is not None and item_id not in ids_allowlist:
                    continue
                score = sum(a * b for a, b in zip(q, vector))
                if min_score is not None and score < min_score:
                    continue
                scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)

        hits: list[VectorHit] = []
        for score, idx in scored[:top_k]:
            hits.append(
                VectorHit(
                    id=self._ids[idx],
                    score=float(score),
                    metadata=dict(self._metadata[idx]),
                )
            )
        return hits
