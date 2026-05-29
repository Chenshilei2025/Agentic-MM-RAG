"""Agent roles, prompts, contracts, runners, and subagent implementations."""

from agentic_mm_rag.agent.contracts import PlannerTaskContract, ToolContract, resolve_tool_name
from agentic_mm_rag.agent.hook import AgentHook, AgentHookContext, CompositeHook
from agentic_mm_rag.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from agentic_mm_rag.orchestrator.types import (
    AgentPlan,
    OrchestrationResult,
    QueryContext,
    ReflectionResult,
    RetrievalTask,
    SubagentResult,
)


def __getattr__(name: str):
    if name in {"AgenticRunResult", "ReflectionDecision"}:
        from agentic_mm_rag.results import AgenticRunResult, ReflectionDecision

        return {
            "AgenticRunResult": AgenticRunResult,
            "ReflectionDecision": ReflectionDecision,
        }[name]
    if name == "DecisionAgent":
        from agentic_mm_rag.agent.decision import DecisionAgent

        return DecisionAgent
    if name in {
        "DocGraphSubagent",
        "DocTextSubagent",
        "DocVisualSubagent",
        "ExpertSubagent",
        "VideoGraphSubagent",
        "VideoTextSubagent",
        "VideoVisualSubagent",
    }:
        from agentic_mm_rag.agent.subagent import (
            DocGraphSubagent,
            DocTextSubagent,
            DocVisualSubagent,
            ExpertSubagent,
            VideoGraphSubagent,
            VideoTextSubagent,
            VideoVisualSubagent,
        )

        return {
            "DocGraphSubagent": DocGraphSubagent,
            "DocTextSubagent": DocTextSubagent,
            "DocVisualSubagent": DocVisualSubagent,
            "ExpertSubagent": ExpertSubagent,
            "VideoGraphSubagent": VideoGraphSubagent,
            "VideoTextSubagent": VideoTextSubagent,
            "VideoVisualSubagent": VideoVisualSubagent,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AgentHook",
    "AgentHookContext",
    "AgentPlan",
    "AgentRunResult",
    "AgentRunSpec",
    "AgentRunner",
    "AgenticRunResult",
    "CompositeHook",
    "DecisionAgent",
    "DocGraphSubagent",
    "DocTextSubagent",
    "DocVisualSubagent",
    "ExpertSubagent",
    "OrchestrationResult",
    "PlannerTaskContract",
    "QueryContext",
    "ReflectionDecision",
    "ReflectionResult",
    "RetrievalTask",
    "SubagentResult",
    "ToolContract",
    "VideoGraphSubagent",
    "VideoTextSubagent",
    "VideoVisualSubagent",
    "resolve_tool_name",
]
