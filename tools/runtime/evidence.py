"""EvidenceCard construction helpers for retrieval backends."""

from __future__ import annotations

import re
from typing import Any, Protocol

from agentic_mm_rag.schemas import EvidenceCard, Locator, ScoreParts


class DocumentEvidenceStore(Protocol):
    def file_for_doc(self, doc_id: str | None) -> str | None:
        ...

    def resolve_visual_asset_path(self, chunk: dict[str, Any]) -> str | None:
        ...


_VISUAL_PATH_PATTERNS = (
    re.compile(r"Image Path:\s*(?P<path>\S+)", re.IGNORECASE),
    re.compile(
        r"(?:image|img|figure|table)[_-]?path['\"]?\s*[:=]\s*['\"]?(?P<path>[^'\"\s,}]+)",
        re.IGNORECASE,
    ),
)


def _first_present(chunk: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = chunk.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_visual_asset_path(chunk: dict[str, Any]) -> str | None:
    direct = _first_present(
        chunk,
        (
            "image_path",
            "img_path",
            "figure_path",
            "table_image_path",
            "visual_path",
            "asset_path",
            "source_image_path",
            "modal_path",
        ),
    )
    if direct is not None:
        return str(direct)
    content = str(chunk.get("content") or "")
    for pattern in _VISUAL_PATH_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group("path").strip().rstrip(".,;)")
    return None


def _extract_visual_trace(chunk_id: str, chunk: dict[str, Any]) -> dict[str, Any]:
    trace = {
        "visual_block_id": chunk_id,
        "source_doc_id": chunk.get("full_doc_id"),
        "page_idx": chunk.get("page_idx"),
        "original_type": chunk.get("original_type"),
        "modal_entity_name": chunk.get("modal_entity_name"),
        "asset_path": _extract_visual_asset_path(chunk),
        "bbox": _first_present(chunk, ("bbox", "bounding_box", "box", "coordinates")),
        "document_file_path": chunk.get("file_path"),
        "requires_image_inspection": bool(chunk.get("is_multimodal")),
    }
    return {key: value for key, value in trace.items() if value is not None}


def _structured_cells_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(chunk.get("content") or "")
    original_type = str(chunk.get("original_type") or "").lower()
    if not content and not chunk.get("is_multimodal"):
        return []
    cells: list[dict[str, Any]] = []
    visual_trace = _extract_visual_asset_path(chunk)
    if original_type in {"table", "chart"} and content:
        cells.append(
            {
                "kind": original_type,
                "row": "",
                "column": "source_text",
                "value": content[:1200],
                "raw_value": content[:1200],
                "asset_path": visual_trace,
            }
        )
    def add_cell(row: str, column: str, value: Any, raw_value: str, *, kind: str | None = None) -> None:
        cells.append(
            {
                "kind": kind or original_type or "text",
                "row": row,
                "column": column,
                "value": value,
                "raw_value": raw_value,
                "asset_path": visual_trace,
            }
        )

    color_names = (
        "red", "blue", "green", "gray", "grey", "yellow", "orange", "purple",
        "black", "white", "brown", "pink", "cyan", "magenta",
    )
    color_pattern = re.compile(
        r"(?P<label>[A-Za-z][A-Za-z0-9 /+_.-]{1,80}?)\s+"
        r"(?:is|are|appears?|shown|colored|coloured|represented)\s+"
        r"(?:in|as|by|with)?\s*(?P<color>"
        + "|".join(color_names)
        + r")\b",
        re.IGNORECASE,
    )
    paren_value_pattern = re.compile(
        r"(?P<label>[A-Za-z][A-Za-z0-9 /+_.-]{1,80}?)\s*"
        r"\((?P<value>-?\d[\d,]*(?:\.\d+)?%?)",
        re.IGNORECASE,
    )
    key_value_pattern = re.compile(
        r"(?P<label>[A-Za-z][A-Za-z0-9 /+_.-]{1,80}?)\s*[:=]\s*"
        r"(?P<value>-?\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9 /+_.-]{1,80})",
        re.IGNORECASE,
    )
    object_count_pattern = re.compile(
        r"\b(?P<count>\d+)\s+(?P<label>cars?|images?|boxes?|bounding boxes|bboxes|icons?|ways?|levels?|segments?)\b",
        re.IGNORECASE,
    )
    for line_idx, line in enumerate(re.split(r"\n+", content)):
        text = line.strip()
        if not text:
            continue
        if re.search(r"\b(?:Image|Table|Figure) Path:\s*/", text, re.IGNORECASE):
            add_cell(f"line_{line_idx + 1}", "asset_path", text, text)
            continue
        if text.startswith(("-", "•", "*")):
            text = text.lstrip("-•* ").strip()
        if not text:
            continue
        for match in color_pattern.finditer(text):
            label = re.sub(r"\s+", " ", match.group("label")).strip(" ,.;:")
            if len(label.split()) <= 10:
                add_cell(label, "color", match.group("color").title(), text, kind="visual_attribute")
        for match in paren_value_pattern.finditer(text):
            label = re.sub(r"\s+", " ", match.group("label")).strip(" ,.;:")
            if len(label.split()) <= 8:
                add_cell(label, "value", match.group("value").replace(",", ""), text)
        for match in key_value_pattern.finditer(text):
            label = re.sub(r"\s+", " ", match.group("label")).strip(" ,.;:")
            value = match.group("value").strip().replace(",", "")
            if len(label.split()) <= 8:
                add_cell(label, "value", value, text)
        for match in object_count_pattern.finditer(text):
            add_cell(match.group("label").lower(), "count", int(match.group("count")), text, kind="object_count")
        numeric_match = re.search(r"-?\d[\d,]*(?:\.\d+)?%?", text)
        if numeric_match:
            add_cell(f"line_{line_idx + 1}", "value", numeric_match.group(0).replace(",", ""), text)
        if any(sep in text for sep in ("|", "\t", ",")) and len(text.split()) <= 16:
            add_cell(f"line_{line_idx + 1}", "text", text, text)
        if re.search(r"https?://\S+", text):
            add_cell(f"line_{line_idx + 1}", "url", text, text)
    return cells


class VideoEvidenceStore(Protocol):
    @property
    def video_paths(self) -> dict[str, str]:
        ...

    def segment_locator(self, segment_id: str) -> tuple[str | None, str | None, str | None]:
        ...


def doc_evidence_from_chunk(
    store: DocumentEvidenceStore,
    chunk_id: str,
    chunk: dict[str, Any],
    *,
    score: float,
    retriever: str,
    score_parts: ScoreParts,
) -> EvidenceCard:
    """Build a document EvidenceCard from a processed text or multimodal chunk."""

    doc_id = str(chunk.get("full_doc_id", ""))
    modality = "text"
    if chunk.get("is_multimodal"):
        original_type = str(chunk.get("original_type", "unknown")).lower()
        modality = (
            original_type
            if original_type
            in {"image", "chart", "table", "equation", "page_footnote", "aside_text", "code"}
            else "unknown"
        )
    visual_trace = _extract_visual_trace(chunk_id, chunk) if chunk.get("is_multimodal") else {}
    if visual_trace and not visual_trace.get("asset_path"):
        resolver = getattr(store, "resolve_visual_asset_path", None)
        if callable(resolver):
            asset_path = resolver(chunk)
            if asset_path:
                visual_trace["asset_path"] = asset_path
    provenance: dict[str, Any] = {"retriever": retriever, "raw_ids": [chunk_id]}
    if visual_trace:
        provenance["visual_block_id"] = chunk_id
        if visual_trace.get("asset_path"):
            provenance["asset_path"] = visual_trace["asset_path"]
    metadata = {
        "tokens": chunk.get("tokens"),
        "chunk_order_index": chunk.get("chunk_order_index"),
        "is_multimodal": chunk.get("is_multimodal", False),
        "original_type": chunk.get("original_type"),
        "modal_entity_name": chunk.get("modal_entity_name"),
    }
    if visual_trace:
        metadata["visual_trace"] = visual_trace
        metadata["visual_asset_path"] = visual_trace.get("asset_path")
        metadata["visual_block_id"] = chunk_id
    structured_cells = _structured_cells_from_chunk(chunk)
    if structured_cells:
        metadata["structured_cells"] = structured_cells
        if chunk.get("is_multimodal"):
            metadata["page_scoped_structured_cells"] = structured_cells
    return EvidenceCard(
        id=chunk_id,
        source_type="doc",
        modality=modality,  # type: ignore[arg-type]
        source_id=doc_id,
        locator=Locator(
            doc_id=doc_id,
            file_path=chunk.get("file_path") or store.file_for_doc(doc_id),
            page_idx=chunk.get("page_idx"),
        ),
        content=str(chunk.get("content", "")),
        score=score,
        score_parts=score_parts,
        provenance=provenance,
        metadata=metadata,
    )


def video_evidence_from_segment(
    store: VideoEvidenceStore,
    segment_id: str,
    segment: dict[str, Any],
    *,
    score: float,
    retriever: str,
    score_parts: ScoreParts,
) -> EvidenceCard:
    """Build a video EvidenceCard from a processed VideoRAG segment."""

    video_id = segment_id.rsplit("_", 1)[0] if "_" in segment_id else segment_id
    raw_time, start_time, end_time = store.segment_locator(segment_id)
    return EvidenceCard(
        id=segment_id,
        source_type="video",
        modality="video_segment",
        source_id=video_id,
        locator=Locator(
            video_id=video_id,
            segment_id=segment_id,
            raw_time=raw_time,
            start_time=start_time,
            end_time=end_time,
            file_path=store.video_paths.get(video_id),
        ),
        content=str(segment.get("content", "")),
        score=score,
        score_parts=score_parts,
        provenance={"retriever": retriever, "raw_ids": [segment_id]},
        metadata={
            "transcript": segment.get("transcript"),
            "frame_times": segment.get("frame_times"),
        },
    )


def video_text_evidence_from_chunk(
    store: VideoEvidenceStore,
    chunk_id: str,
    chunk: dict[str, Any],
    *,
    score: float,
    score_parts: ScoreParts,
    include_mapped_segment_details: bool = False,
) -> EvidenceCard:
    """Build video text EvidenceCard from a VideoRAG text chunk mapped to segments."""

    segment_ids = list(chunk.get("video_segment_id", []))
    first_segment = segment_ids[0] if segment_ids else None
    video_id = first_segment.rsplit("_", 1)[0] if first_segment else ""
    raw_time = start_time = end_time = None
    if first_segment:
        raw_time, start_time, _ = store.segment_locator(first_segment)
        if segment_ids:
            _, _, end_time = store.segment_locator(segment_ids[-1])
    segment_context: list[str] = []
    segment_lookup = getattr(store, "segment_by_id", None)
    if include_mapped_segment_details and callable(segment_lookup):
        for segment_id in segment_ids[:4]:
            segment = segment_lookup(str(segment_id))
            if not isinstance(segment, dict):
                continue
            transcript = str(segment.get("transcript") or "").strip()
            content = str(segment.get("content") or "").strip()
            if transcript:
                segment_context.append(f"{segment_id} transcript: {transcript}")
            elif content:
                segment_context.append(f"{segment_id} caption: {content}")
    content = str(chunk.get("content", ""))
    if segment_context:
        content = content.rstrip() + "\n\nMapped segment details:\n" + "\n".join(segment_context)
    return EvidenceCard(
        id=chunk_id,
        source_type="video",
        modality="text",
        source_id=video_id,
        locator=Locator(
            video_id=video_id,
            segment_id=",".join(segment_ids),
            raw_time=raw_time,
            start_time=start_time,
            end_time=end_time,
            file_path=store.video_paths.get(video_id),
        ),
        content=content,
        score=score,
        score_parts=score_parts,
        provenance={"retriever": "video_text_seek", "raw_ids": [chunk_id]},
        metadata={
            "tokens": chunk.get("tokens"),
            "chunk_order_index": chunk.get("chunk_order_index"),
            "video_segment_id": segment_ids,
            "segment_count": len(segment_ids),
            "mapped_segment_details": segment_context,
        },
    )


def csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", "\n", '"']):
        text = '"' + text.replace('"', '""') + '"'
    return text


def video_caption_context(store: VideoEvidenceStore, items: list[dict[str, Any]]) -> str:
    """Render retrieved video evidence as the CSV context expected by VideoRAG prompts."""

    rows = [["video_name", "start_time", "end_time", "content"]]
    for item in items:
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        segment_id = locator.get("segment_id") or item.get("id")
        if not segment_id:
            continue
        first_segment = str(segment_id).split(",", 1)[0]
        video_id = first_segment.rsplit("_", 1)[0] if "_" in first_segment else str(
            item.get("source_id") or ""
        )
        _raw, start_time, end_time = store.segment_locator(first_segment)
        content = str(item.get("content") or "").replace("\n", " ").strip()
        rows.append([video_id, start_time or "", end_time or "", content])
    if len(rows) == 1:
        return ""
    csv_rows = [",".join(csv_cell(cell) for cell in row) for row in rows]
    return "\n-----Retrieved Knowledge From Videos-----\n```csv\n" + "\n".join(csv_rows) + "\n```\n"
