"""Project-local defaults and environment overrides.

The package should import cleanly in a fresh checkout. Defaults therefore point
at repository-relative locations, while deployments can override them through
environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: str | Path) -> Path:
    """Return an environment-controlled path with a repository-local default."""

    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


def env_path_list(name: str, default: tuple[str | Path, ...] = ()) -> tuple[Path, ...]:
    """Return environment-controlled paths split by the platform path separator."""

    value = os.environ.get(name)
    raw_paths: tuple[str | Path, ...]
    if value:
        raw_paths = tuple(item for item in value.split(os.pathsep) if item.strip())
    else:
        raw_paths = default
    return tuple(Path(item).expanduser() for item in raw_paths)


def env_str(name: str, default: str) -> str:
    """Return an environment-controlled string with a stable default."""

    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class CorpusPaths:
    """Locations of processed corpora used by retrieval tools."""

    doc_rag_dir: Path = env_path(
        "AGENTIC_MM_RAG_DOC_ROOT",
        REPO_ROOT / "data" / "doc_rag",
    )
    video_rag_dir: Path = env_path(
        "AGENTIC_MM_RAG_VIDEO_ROOT",
        REPO_ROOT / "data" / "video_rag",
    )
    doc_visual_asset_roots: tuple[Path, ...] = env_path_list(
        "AGENTIC_MM_RAG_DOC_VISUAL_ASSET_ROOTS",
    )


@dataclass(frozen=True)
class ModelDefaults:
    """Default model routing for decision and specialist agents."""

    decision: str = env_str("AGENTIC_MM_RAG_DECISION_MODEL", "gpt-4o")
    text_expert: str = env_str("AGENTIC_MM_RAG_TEXT_MODEL", "gpt-4o-mini")
    visual_expert: str = env_str("AGENTIC_MM_RAG_VISUAL_MODEL", "gpt-4.1")
    graph_expert: str = env_str("AGENTIC_MM_RAG_GRAPH_MODEL", "gpt-4o-mini")


DEFAULT_PATHS = CorpusPaths()
DEFAULT_MODELS = ModelDefaults()
