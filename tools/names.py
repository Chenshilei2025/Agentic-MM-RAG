"""Canonical names for the agent-visible tool surface."""

QUERY_TOOL_NAMES: tuple[str, ...] = ()
DOC_TOOL_NAMES = ("doc_text_seek", "doc_visual_seek", "doc_graph_seek")
VIDEO_TOOL_NAMES = ("video_text_seek", "video_visual_seek", "video_graph_seek")
EVIDENCE_TOOL_NAMES = ("read_evidence", "write_evidence")

PUBLIC_TOOL_NAMES = (
    *QUERY_TOOL_NAMES,
    *DOC_TOOL_NAMES,
    *VIDEO_TOOL_NAMES,
    *EVIDENCE_TOOL_NAMES,
)
