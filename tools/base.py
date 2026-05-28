"""Base classes for agentic_mm_rag tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from agentic_mm_rag.schemas import ToolResponse

if TYPE_CHECKING:
    from agentic_mm_rag.tools.context import ToolContext

_ToolT = TypeVar("_ToolT", bound="Tool")

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Schema(ABC):
    """JSON Schema fragment with shared validation helpers."""

    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t  # type: ignore[return-value]

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(
        val: Any,
        schema: dict[str, Any],
        path: str = "",
    ) -> list[str]:
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get(
            "nullable", False
        )
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(
            val, _JSON_TYPE_MAP[t]
        ):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            for k, v in val.items():
                if k in props:
                    errors.extend(
                        Schema.validate_json_schema_value(
                            v,
                            props[k],
                            Schema.subpath(path, k),
                        )
                    )
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(val):
                    errors.extend(
                        Schema.validate_json_schema_value(
                            item,
                            schema["items"],
                            prefix.format(i),
                        )
                    )
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        to_json_schema = getattr(value, "to_json_schema", None)
        if callable(to_json_schema):
            return to_json_schema()
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


@dataclass(slots=True)
class ToolMetadata:
    """Structured metadata used to document tool intent and visibility."""

    category: str = "general"
    corpus: str = "shared"
    role: str = "tool"
    stability: str = "stable"
    is_public: bool = True
    fallback: bool = False
    recommended_usage: str = ""
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "category": self.category,
            "corpus": self.corpus,
            "role": self.role,
            "stability": self.stability,
            "is_public": self.is_public,
            "fallback": self.fallback,
        }
        if self.recommended_usage:
            out["recommended_usage"] = self.recommended_usage
        if self.tags:
            out["tags"] = list(self.tags)
        if self.extra:
            out.update(self.extra)
        return out


def tool_metadata(
    *,
    category: str = "general",
    corpus: str = "shared",
    role: str = "tool",
    stability: str = "stable",
    is_public: bool = True,
    fallback: bool = False,
    recommended_usage: str = "",
    tags: tuple[str, ...] | list[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicit structured metadata for a tool manifest."""

    return ToolMetadata(
        category=category,
        corpus=corpus,
        role=role,
        stability=stability,
        is_public=is_public,
        fallback=fallback,
        recommended_usage=recommended_usage,
        tags=tuple(tags),
        extra=dict(extra or {}),
    ).to_dict()


def infer_tool_metadata(
    name: str,
    description: str,
    *,
    is_public: bool = True,
    fallback: bool = False,
) -> ToolMetadata:
    """Infer a reasonable manifest for a tool from its name and description."""

    lower_name = name.lower()
    lower_desc = description.lower()

    if lower_name.startswith("doc_"):
        corpus = "doc"
    elif lower_name.startswith("video_"):
        corpus = "video"
    else:
        corpus = "shared"

    if "inspect" in lower_name:
        category = "inspection"
        role = "inspect"
    elif "manifest" in lower_name:
        category = "inspection"
        role = "manifest"
    elif "expand" in lower_name:
        category = "context"
        role = "expand"
    elif "fuse" in lower_name:
        category = "fusion"
        role = "fusion"
    elif "retrieve" in lower_name or "seek" in lower_name or "hybrid" in lower_name:
        category = "retrieval"
        role = "strategy"
    else:
        category = "general"
        role = "tool"

    if lower_desc.startswith("internal:"):
        stability = "internal"
        fallback = fallback or "compatibility" in lower_desc
    elif "experimental" in lower_desc:
        stability = "experimental"
    else:
        stability = "stable"

    if role == "strategy":
        if corpus == "doc":
            recommended_usage = "Use for explicit document retrieval strategy selection."
        elif corpus == "video":
            recommended_usage = "Use for explicit video retrieval strategy selection."
        else:
            recommended_usage = "Use for general retrieval routing."
    elif role == "inspect":
        recommended_usage = "Use for second-pass evidence inspection and reflection."
    elif role == "manifest":
        recommended_usage = "Use to inspect corpus availability before retrieval."
    elif role == "expand":
        recommended_usage = "Use to expand local chunk or segment context."
    else:
        recommended_usage = ""

    tags = tuple(
        tag
        for tag in (
            corpus,
            category,
            role,
            "public" if is_public else "internal",
            "fallback" if fallback else "",
        )
        if tag
    )
    return ToolMetadata(
        category=category,
        corpus=corpus,
        role=role,
        stability=stability,
        is_public=is_public,
        fallback=fallback,
        recommended_usage=recommended_usage,
        tags=tags,
        extra={},
    )


class Tool(ABC):
    """Agent tool with OpenAI-compatible schema and schema-driven validation."""

    _TYPE_MAP = _JSON_TYPE_MAP
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))
    config_key: str = ""
    _plugin_discoverable: bool = True
    _scopes: set[str] = {"core"}

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    @property
    def read_only(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    @property
    def metadata(self) -> dict[str, Any]:
        return infer_tool_metadata(self.name, self.description).to_dict()

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "read_only": self.read_only,
            "exclusive": self.exclusive,
            "concurrency_safe": self.concurrency_safe,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @classmethod
    def config_cls(cls) -> type[Any] | None:
        return None

    @classmethod
    def enabled(cls, ctx: ToolContext | None) -> bool:
        return True

    @classmethod
    def create(cls: type[_ToolT], ctx: ToolContext | None) -> _ToolT:
        return cls()  # type: ignore[call-arg]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResponse:
        ...

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        return Schema.resolve_json_schema_type(t)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {k: self._cast_value(v, props[k]) if k in props else v for k, v in obj.items()}

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        t = self._resolve_type(schema.get("type"))

        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[t]
            if isinstance(val, expected):
                return val

        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val
        if t == "string":
            return val if val is None else str(val)
        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                return True
            if low in self._BOOL_FALSE:
                return False
            return val
        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            return [self._cast_value(x, items) for x in val] if items else val
        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)
        return val

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def __call__(self, **kwargs: Any) -> ToolResponse:
        return await self.execute(**kwargs)


@dataclass(slots=True)
class FunctionTool(Tool):
    """Wrap an async function as a schema-aware tool."""

    _name: str
    _description: str
    func: Any
    _parameters: dict[str, Any]
    _read_only: bool = True
    _manifest_metadata: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._parameters)

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = infer_tool_metadata(
            self.name,
            self.description,
            is_public=True,
            fallback=False,
        ).to_dict()
        if self._manifest_metadata:
            metadata.update(deepcopy(self._manifest_metadata))
        return metadata

    async def execute(self, **kwargs: Any) -> ToolResponse:
        return await self.func(**kwargs)


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """Class decorator for attaching JSON schema parameters to a tool."""

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return deepcopy(frozen)

        cls.parameters = parameters  # type: ignore[assignment]
        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]
        return cls

    return decorator
