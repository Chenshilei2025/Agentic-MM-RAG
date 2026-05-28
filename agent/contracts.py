"""Agent contracts for the final seek/read/write tool protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AgentProfile = Literal[
    "decision_agent",
    "doc_text_subagent",
    "doc_visual_subagent",
    "doc_graph_subagent",
    "video_text_subagent",
    "video_visual_subagent",
    "video_graph_subagent",
]

ToolIntent = Literal[
    "doc_text_seek",
    "doc_visual_seek",
    "doc_graph_seek",
    "video_text_seek",
    "video_visual_seek",
    "video_graph_seek",
    "read_evidence",
    "write_evidence",
]

PlannerStrategy = Literal[
    "decision_agent",
    "doc_text_first",
    "doc_visual_first",
    "doc_graph_first",
    "video_text_first",
    "video_visual_first",
    "video_graph_first",
    "multi_corpus_parallel",
]

PROFILE_TOOLS: dict[str, list[str]] = {
    "decision_agent": ["read_evidence"],
    "doc_text_subagent": ["doc_text_seek", "write_evidence"],
    "doc_visual_subagent": ["doc_visual_seek", "write_evidence"],
    "doc_graph_subagent": ["doc_graph_seek", "write_evidence"],
    "video_text_subagent": ["video_text_seek", "write_evidence"],
    "video_visual_subagent": ["video_visual_seek", "write_evidence"],
    "video_graph_subagent": ["video_graph_seek", "write_evidence"],
}

PROFILE_SEEK_TOOL: dict[str, str] = {
    profile: tools[0] for profile, tools in PROFILE_TOOLS.items() if profile != "decision_agent"
}

TOOL_TO_PROFILE: dict[str, str] = {tool: profile for profile, tool in PROFILE_SEEK_TOOL.items()}


@dataclass(slots=True)
class ToolContract:
    """Stable semantic contract mapped directly to final tool names."""

    intent: ToolIntent
    corpus: Literal["doc", "video", "shared"]
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "corpus": self.corpus,
            "params": dict(self.params),
        }


@dataclass(slots=True)
class PlannerTaskContract:
    """Planner-facing task contract for expert dispatch."""

    subagent_name: AgentProfile
    query: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)
    stop_condition: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent_name": self.subagent_name,
            "query": self.query,
            "allowed_tools": list(self.allowed_tools),
            "expected_evidence": list(self.expected_evidence),
            "stop_condition": self.stop_condition,
            "params": dict(self.params),
            "rationale": self.rationale,
            "depends_on": list(self.depends_on),
        }


def allowed_tools_for_contract(profile: str, corpus: str | None = None) -> list[str]:
    try:
        return list(PROFILE_TOOLS[profile])
    except KeyError as exc:
        raise ValueError(f"unknown agent profile: {profile}") from exc


def resolve_tool_name(intent: str, corpus: str | None = None) -> str:
    if intent in PROFILE_TOOLS:
        tools = PROFILE_TOOLS[intent]
        return tools[0]
    if intent in PROFILE_SEEK_TOOL.values() or intent in {"read_evidence", "write_evidence"}:
        return intent
    if corpus in {"doc", "video"} and intent in {"text", "visual", "graph"}:
        return f"{corpus}_{intent}_seek"
    raise ValueError(f"no final tool mapping for intent={intent!r} corpus={corpus!r}")


def profile_for_tool(tool_name: str) -> str:
    try:
        return TOOL_TO_PROFILE[tool_name]
    except KeyError as exc:
        raise ValueError(f"no subagent profile for tool: {tool_name}") from exc


def infer_doc_query_profile(query_text: str) -> dict[str, bool]:
    text = query_text.lower()
    return {
        "table_like": any(token in text for token in ("table", "tabular", "row", "column", "population", "survey")),
        "temporal_like": any(token in text for token in ("before", "after", "when", "date", "year", "time", "timeline")),
        "visual_like": any(token in text for token in ("figure", "image", "diagram", "visual", "chart", "axis", "page")),
        "count_like": any(token in text for token in ("how many", "count", "number of", "sum", "total")),
        "compare_like": any(token in text for token in ("greater", "larger", "higher", "more than", "less than")),
    }


def choose_doc_planner_route(
    doc_profile: dict[str, Any] | None,
    query_profile: dict[str, bool],
) -> str:
    if query_profile.get("visual_like") or query_profile.get("table_like"):
        return "doc_visual_first"
    if doc_profile and (
        doc_profile.get("text_sparse")
        or doc_profile.get("visual_heavy")
        or doc_profile.get("page_coverage_ratio", 1.0) < 0.25
    ):
        return "doc_graph_first"
    return "doc_text_first"


def preferred_doc_modalities(query_profile: dict[str, bool]) -> list[str] | None:
    modalities: list[str] = []
    if query_profile.get("table_like") or query_profile.get("count_like") or query_profile.get("compare_like"):
        modalities.append("table")
    if query_profile.get("visual_like"):
        modalities.append("image")
        modalities.append("chart")
    if query_profile.get("temporal_like"):
        modalities.append("page_footnote")
    return list(dict.fromkeys(modalities)) or None


def infer_doc_edge_type_filter(query_text: str) -> list[str]:
    text = query_text.lower()
    edge_types: list[str] = []
    if any(token in text for token in ("table", "tabular", "row", "column")):
        edge_types.append("table")
    if any(token in text for token in ("before", "after", "when", "date", "year", "time", "timeline")):
        edge_types.append("temporal")
    if any(token in text for token in ("figure", "image", "diagram", "visual", "chart")):
        edge_types.append("visual")
    return edge_types or ["semantic", "table", "visual", "temporal"]
