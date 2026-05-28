"""Small tool-calling agent runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any

from agentic_mm_rag.agent.hook import AgentHook, AgentHookContext
from agentic_mm_rag.providers.base import LLMProvider, ToolCallRequest
from agentic_mm_rag.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for one model role session."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int = 4
    max_tokens: int = 4096
    temperature: float = 0.2
    reasoning_effort: str | None = None
    concurrent_tools: bool = True
    fail_on_tool_error: bool = False
    session_key: str | None = None
    hook: AgentHook | None = None
    checkpoint_callback: Any | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Result of a tool-capable model loop."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None


class AgentRunner:
    """Execute a chat model until it stops calling tools or hits max iterations."""

    def __init__(self, provider: LLMProvider) -> None:
        if provider is None:
            raise ValueError("AgentRunner requires an LLMProvider.")
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        messages = list(spec.initial_messages)
        tools_used: list[str] = []
        usage: dict[str, int] = {}
        final_content: str | None = None
        error: str | None = None
        stop_reason = "completed"
        hook = spec.hook or AgentHook()

        for iteration in range(spec.max_iterations):
            hook_context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                usage=usage,
            )
            await hook.before_iteration(hook_context)
            response = await self.provider.chat(
                messages=messages,
                tools=spec.tools.get_definitions(),
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                reasoning_effort=spec.reasoning_effort,
            )
            self._accumulate_usage(usage, response.usage)
            hook_context.response = response
            hook_context.usage = dict(usage)
            if response.error:
                error = response.error
                final_content = response.content or response.error
                stop_reason = "error"
                final_content = hook.finalize_content(hook_context, final_content)
                messages.append({"role": "assistant", "content": final_content})
                hook_context.final_content = final_content
                hook_context.stop_reason = stop_reason
                hook_context.error = error
                await self._emit_checkpoint(spec, hook_context)
                await hook.after_iteration(hook_context)
                break

            if response.should_execute_tools:
                assistant_message = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [call.to_openai_tool_call() for call in response.tool_calls],
                }
                messages.append(assistant_message)
                hook_context.tool_calls = list(response.tool_calls)
                await hook.before_execute_tools(hook_context)
                tool_results = await self._execute_tool_calls(spec, response.tool_calls)
                hook_context.tool_results = list(tool_results)
                for call, result in zip(response.tool_calls, tool_results):
                    tools_used.append(call.name)
                    if isinstance(result, str) and result.startswith("Error:"):
                        if spec.fail_on_tool_error:
                            error = result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": self._stringify_tool_result(result),
                        }
                    )
                if error:
                    final_content = error
                    stop_reason = "tool_error"
                    final_content = hook.finalize_content(hook_context, final_content)
                    messages.append({"role": "assistant", "content": final_content})
                    hook_context.final_content = final_content
                    hook_context.stop_reason = stop_reason
                    hook_context.error = error
                    await self._emit_checkpoint(spec, hook_context)
                    await hook.after_iteration(hook_context)
                    break
                hook_context.stop_reason = "tool_calls"
                await self._emit_checkpoint(spec, hook_context)
                await hook.after_iteration(hook_context)
                continue

            final_content = response.content or ""
            final_content = hook.finalize_content(hook_context, final_content)
            messages.append({"role": "assistant", "content": final_content})
            hook_context.final_content = final_content
            hook_context.stop_reason = stop_reason
            await self._emit_checkpoint(spec, hook_context)
            await hook.after_iteration(hook_context)
            break
        else:
            stop_reason = "max_iterations"
            final_content = (
                "Agent stopped because it reached max_iterations before finalizing."
            )
            messages.append({"role": "assistant", "content": final_content})
            hook_context = AgentHookContext(
                iteration=spec.max_iterations,
                messages=messages,
                usage=dict(usage),
                final_content=final_content,
                stop_reason=stop_reason,
            )
            await self._emit_checkpoint(spec, hook_context)
            await hook.after_iteration(hook_context)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
        )

    async def _execute_tool_calls(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[Any]:
        if spec.concurrent_tools and len(tool_calls) > 1:
            return list(await asyncio.gather(*(self._run_tool(spec, call) for call in tool_calls)))
        results: list[Any] = []
        for call in tool_calls:
            results.append(await self._run_tool(spec, call))
        return results

    @staticmethod
    async def _run_tool(spec: AgentRunSpec, call: ToolCallRequest) -> Any:
        response = await spec.tools.execute(call.name, call.arguments)
        if not response.ok:
            return f"Error: {response.error or 'tool failed'}"
        return response.to_dict()

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + int(value or 0)

    @staticmethod
    async def _emit_checkpoint(
        spec: AgentRunSpec,
        context: AgentHookContext,
    ) -> None:
        if spec.checkpoint_callback is None:
            return
        await spec.checkpoint_callback(
            {
                "iteration": context.iteration,
                "stop_reason": context.stop_reason,
                "usage": dict(context.usage),
                "tool_calls": [call.name for call in context.tool_calls],
                "final_content": context.final_content,
                "error": context.error,
            }
        )
