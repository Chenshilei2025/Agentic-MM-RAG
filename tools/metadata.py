"""Manifest metadata helpers for agent-visible tools."""

from __future__ import annotations

from typing import Any

from agentic_mm_rag.tools.base import tool_metadata


def seek_metadata(corpus: str, modality: str) -> dict[str, Any]:
    return tool_metadata(
        category="seek",
        corpus=corpus,
        role=f"{modality}_seek",
        stability="stable",
        recommended_usage=(
            f"Use from {corpus}_{modality}_subagent to recall and score top-k "
            "candidate evidence; semantic filtering happens in the subagent."
        ),
        tags=[corpus, modality, "seek", "public"],
    )
