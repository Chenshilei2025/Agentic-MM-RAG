"""Tool discovery and registration."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

from agentic_mm_rag.tools.base import Tool
from agentic_mm_rag.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

_SKIP_MODULES = frozenset({"base", "schema", "registry", "loader", "__init__"})


class ToolLoader:
    """Discover tools in a package."""

    def __init__(self, package: Any = None, *, test_classes: list[type[Tool]] | None = None):
        if package is None:
            import agentic_mm_rag.tools as package
        self._package = package
        self._test_classes = test_classes
        self._discovered: list[type[Tool]] | None = None

    def discover(self) -> list[type[Tool]]:
        if self._test_classes is not None:
            return list(self._test_classes)
        if self._discovered is not None:
            return self._discovered

        results: list[type[Tool]] = []
        seen: set[int] = set()
        for _importer, module_name, _ispkg in pkgutil.iter_modules(self._package.__path__):
            if module_name.startswith("_") or module_name in _SKIP_MODULES:
                continue
            try:
                module = importlib.import_module(f".{module_name}", self._package.__name__)
            except Exception:
                logger.exception("Failed to import tool module: %s", module_name)
                continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                    and attr.__module__ == module.__name__
                    and not attr_name.startswith("_")
                    and not getattr(attr, "__abstractmethods__", None)
                    and id(attr) not in seen
                ):
                    seen.add(id(attr))
                    results.append(attr)
        results.sort(key=lambda cls: cls.__name__)
        self._discovered = results
        return results

    def load(
        self,
        registry: ToolRegistry,
        ctx: Any | None = None,
        *,
        scope: str = "core",
    ) -> list[str]:
        registered: list[str] = []

        for tool_cls in self.discover():
            try:
                if scope not in getattr(tool_cls, "_scopes", {"core"}):
                    continue
                if not tool_cls.enabled(ctx):
                    continue
                tool = tool_cls.create(ctx)
                registry.register(tool)
                registered.append(tool.name)
            except Exception:
                logger.exception("Failed to register tool class: %s", tool_cls.__name__)

        for _importer, module_name, _ispkg in pkgutil.iter_modules(self._package.__path__):
            if module_name.startswith("_") or module_name in _SKIP_MODULES:
                continue
            try:
                module = importlib.import_module(f".{module_name}", self._package.__name__)
                register_tools = getattr(module, "register_tools", None)
                if not callable(register_tools):
                    continue
                before = set(registry.tool_names)
                signature = inspect.signature(register_tools)
                if len(signature.parameters) >= 2:
                    register_tools(registry, ctx)
                else:
                    register_tools(registry)
                for name in registry.tool_names:
                    if name not in before and name not in registered:
                        registered.append(name)
            except Exception:
                logger.exception("Failed to register tools from module: %s", module_name)

        return registered
