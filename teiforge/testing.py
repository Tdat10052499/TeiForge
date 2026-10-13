"""Helpers for testing agents without network access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

from .llm import LLMError, LLMResponse, Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from .tools import Tool

ScriptStep = Union[str, LLMResponse, Dict[str, Any]]


class ScriptedProvider:
    """An :class:`~teiforge.llm.LLMProvider` that replays a fixed script.

    Each step is a string (a final text answer), an :class:`LLMResponse`, or a dict
    ``{"tool": name, "args": {...}}`` that requests one tool call::

        provider = ScriptedProvider([
            {"tool": "web_search", "args": {"query": "python"}},
            "Python is a programming language.",
        ])

    Every request is recorded in :attr:`requests` for assertions.
    """

    name = "scripted"

    def __init__(self, script: Sequence[ScriptStep], model: str = "scripted-model") -> None:
        self.model = model
        self._script = list(script)
        self.requests: List[Dict[str, Any]] = []

    def complete(self, messages: List[Message], tools: Sequence[Tool] = ()) -> LLMResponse:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": [t.name for t in tools]})
        if not self._script:
            raise LLMError("ScriptedProvider has no more steps.")
        step = self._script.pop(0)
        if isinstance(step, LLMResponse):
            return step
        if isinstance(step, str):
            return LLMResponse(content=step)
        call_id = f"call_{len(self.requests)}"
        return LLMResponse(
            content=step.get("content"),
            tool_calls=[ToolCall(id=call_id, name=step["tool"], arguments=dict(step.get("args", {})))],
        )


__all__ = ["ScriptedProvider"]
