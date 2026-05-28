"""Open-source facing runtime builders for the tools runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.runtime.backends import build_backend
from agentic_mm_rag.tools.runtime.contracts import RetrievalBackend
from agentic_mm_rag.tools.runtime.corpus import CorpusAdapter, build_corpus_adapter
from agentic_mm_rag.orchestrator.tools import (
    RegistryBundle,
    ToolRegistryProfile,
    build_public_tool_registry,
    build_registry_bundle,
)
from agentic_mm_rag.tools.runtime.types import QueryRequest
from agentic_mm_rag.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgenticRuntime:
    """Convenience entrypoint for corpus-aware tool registries and manifests."""

    registry_bundle: RegistryBundle = field(default_factory=build_registry_bundle)

    @property
    def tools(self) -> ToolRegistry:
        return self.registry_bundle.internal_tools

    @property
    def public_tools(self) -> ToolRegistry:
        return self.registry_bundle.public_tools

    @property
    def default_profile(self) -> ToolRegistry:
        return self.registry_bundle.public_tools

    def tool_registry(
        self,
        profile: ToolRegistryProfile = "default",
    ) -> ToolRegistry:
        """Return a curated tool registry profile for external integrations."""

        return build_public_tool_registry(self.tools, profile=profile)

    def tool_manifest(
        self,
        profile: ToolRegistryProfile = "default",
    ) -> list[dict[str, Any]]:
        """Return the structured tool manifest for a curated registry profile."""

        return self.tool_registry(profile=profile).get_manifest()

    def registry_manifest(self) -> dict[str, list[dict[str, Any]]]:
        """Return all curated manifest profiles in one open-source friendly payload."""

        return {
            profile: registry.get_manifest()
            for profile, registry in self.registry_bundle.profile_tools.items()
        }

    def corpus(
        self,
        corpus_type: str,
        *,
        root: str | Path | None = None,
    ) -> CorpusAdapter:
        return build_corpus_adapter(corpus_type, root=root)

    def backend(
        self,
        corpus_type: str,
        *,
        root: str | Path | None = None,
    ) -> RetrievalBackend:
        return build_backend(corpus_type, root=root)

    def manifests(
        self,
        request: QueryRequest | None = None,
        *,
        corpora: tuple[str, ...] | None = None,
        roots: dict[str, str | Path | None] | None = None,
    ) -> dict[str, Any]:
        query = request or QueryRequest(query_text="")
        selected = corpora or query.corpora
        root_overrides = {
            "doc": query.doc_root,
            "video": query.video_root,
        }
        if roots:
            for corpus_name, root in roots.items():
                root_overrides[corpus_name] = root

        data: dict[str, Any] = {}
        for corpus_name in selected:
            adapter = self.corpus(corpus_name, root=root_overrides.get(corpus_name))
            data[adapter.corpus_type] = adapter.manifest()
        return data

    def manifest_response(
        self,
        request: QueryRequest | None = None,
        *,
        corpora: tuple[str, ...] | None = None,
        roots: dict[str, str | Path | None] | None = None,
    ) -> ToolResponse:
        return ToolResponse(
            ok=True,
            tool="runtime_manifest",
            data=self.manifests(request, corpora=corpora, roots=roots),
        )
