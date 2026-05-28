"""Backward-compatible prompt role exports.

New code should import from agentic_mm_rag.agent.prompts.
"""

from agentic_mm_rag.agent.prompts import (
    DOC_GRAPH_SUBAGENT_ROLE,
    DOC_TEXT_SUBAGENT_ROLE,
    DOC_VISUAL_SUBAGENT_ROLE,
    VIDEO_GRAPH_SUBAGENT_ROLE,
    VIDEO_TEXT_SUBAGENT_ROLE,
    VIDEO_VISUAL_SUBAGENT_ROLE,
)

__all__ = [
    "DOC_GRAPH_SUBAGENT_ROLE",
    "DOC_TEXT_SUBAGENT_ROLE",
    "DOC_VISUAL_SUBAGENT_ROLE",
    "VIDEO_GRAPH_SUBAGENT_ROLE",
    "VIDEO_TEXT_SUBAGENT_ROLE",
    "VIDEO_VISUAL_SUBAGENT_ROLE",
]
