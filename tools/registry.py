"""Registry for retrieval tools."""

from __future__ import annotations

from typing import Any

from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.base import Tool


class ToolRegistry:
    """Register, describe, validate, and execute agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions
        definitions = [tool.to_schema() for tool in self._tools.values()]
        definitions.sort(key=self._schema_name)
        self._cached_definitions = definitions
        return definitions

    def get_manifest(self) -> list[dict[str, Any]]:
        manifest = [tool.manifest() for tool in self._tools.values()]
        manifest.sort(key=lambda item: str(item.get("name", "")))
        return manifest

    def prepare_call(
        self,
        name: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        raw_params: Any = {} if params is None else params
        tool = self._tools.get(name)
        if tool is None:
            return None, raw_params, (
                f"Error: Tool '{name}' not found. Available: {', '.join(self.names)}"
            )
        cast_params = tool.cast_params(raw_params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    async def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResponse:
        if params is not None and kwargs:
            return ToolResponse(
                ok=False,
                tool=name,
                error="pass either params or keyword arguments, not both",
            )
        call_params = params if params is not None else kwargs
        tool, cast_params, error = self.prepare_call(name, call_params)
        if error:
            return ToolResponse(ok=False, tool=name, error=error)
        try:
            assert tool is not None
            return await tool.execute(**cast_params)
        except Exception as exc:
            return ToolResponse(ok=False, tool=name, error=f"Error executing {name}: {exc}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
