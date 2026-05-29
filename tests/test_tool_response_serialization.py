from __future__ import annotations

import asyncio

from agentic_mm_rag.agent.runner import AgentRunSpec, AgentRunner
from agentic_mm_rag.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agentic_mm_rag.schemas import ToolResponse
from agentic_mm_rag.tools.base import FunctionTool
from agentic_mm_rag.tools.registry import ToolRegistry
from agentic_mm_rag.tools.schema import ArraySchema, NumberSchema, tool_parameters_schema


class ToolErrorProvider(LLMProvider):
    async def chat(
        self,
        *,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
        response_format=None,
    ) -> LLMResponse:
        if len(messages) <= 1:
            return LLMResponse(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call-1",
                        name="broken_tool",
                        arguments={"x": 1},
                    )
                ],
            )
        return LLMResponse(content="done")


async def broken_tool(x: int) -> ToolResponse:
    return ToolResponse(ok=False, tool="broken_tool", error="boom")


def test_runner_serializes_tool_response_errors_as_json():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "broken_tool",
            "Broken tool.",
            broken_tool,
            tool_parameters_schema(
                x=ArraySchema(NumberSchema("value"), min_items=1),
                required=["x"],
            ),
        )
    )
    runner = AgentRunner(ToolErrorProvider())

    result = asyncio.run(
        runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "user", "content": "test"}],
                tools=registry,
                model="dummy",
                max_iterations=2,
            )
        )
    )

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_messages
    assert tool_messages[0]["content"].startswith("{")
    assert '"ok": false' in tool_messages[0]["content"]
    assert result.error is None
