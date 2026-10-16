"""Plain helpers for the terminal UI: how events are shown and which tools need approval.

Kept free of Textual so they can be tested and reused without the ``tui`` extra.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from rich.markup import escape

from ..events import Event
from ..llm import ToolCall

#: Tools that change something outside the conversation; the UI asks before running them.
APPROVAL_TOOLS = frozenset({"file_writer", "command_runner"})

#: Built-in tools the UI gives the agent by default.
DEFAULT_UI_TOOLS = ("web_search", "file_reader", "file_writer")


def default_tools() -> List[str]:
    """Return the default tools, adding ``command_runner`` only when commands are enabled."""
    tools = list(DEFAULT_UI_TOOLS)
    if os.getenv("TEIFORGE_ENABLE_COMMANDS") == "1":
        tools.append("command_runner")
    return tools


def needs_approval(call: ToolCall) -> bool:
    return call.name in APPROVAL_TOOLS


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_step(event: Event) -> Optional[str]:
    """Return one line of Rich markup describing ``event``, or ``None`` to skip it."""
    data = event.data
    time = event.timestamp.astimezone().strftime("%H:%M:%S")
    if event.type == "task_start":
        return f"[dim]{time}[/] [b]▶[/] {escape(_short(data.get('task', '')))}"
    if event.type == "llm_response":
        names = data.get("tool_calls") or []
        detail = f"asks for {escape(', '.join(names))}" if names else "answered"
        return f"[dim]{time}[/]   model {detail}"
    if event.type == "tool_call":
        arguments = data.get("arguments") or {}
        shown = ", ".join(f"{key}={_short(value, 30)}" for key, value in arguments.items())
        note = " [i](recovered from text)[/]" if data.get("recovered") else ""
        return f"[dim]{time}[/]   [cyan]▸ {escape(str(data.get('name')))}[/] {escape(shown)}{note}"
    if event.type == "tool_result":
        if data.get("declined"):
            return f"[dim]{time}[/]     [yellow]⊘ declined by you[/]"
        if data.get("ok"):
            return f"[dim]{time}[/]     [green]✓ ok[/] · {len(str(data.get('output', '')))} chars"
        return f"[dim]{time}[/]     [red]✗ {escape(_short(data.get('output', ''), 70))}[/]"
    if event.type == "self_check":
        return f"[dim]{time}[/]   [magenta]↻ asked the model to check its answer again[/]"
    if event.type == "error":
        return f"[dim]{time}[/]   [red]! {escape(_short(data.get('message', ''), 70))}[/]"
    if event.type == "final":
        status = "[green]done[/]" if data.get("ok") else "[red]failed[/]"
        return f"[dim]{time}[/] [b]■[/] {status}"
    return None


def describe_call(call: ToolCall) -> str:
    """Return a readable description of a tool call for the approval dialog (plain text)."""
    arguments = dict(call.arguments)
    if call.name == "file_writer":
        content = str(arguments.get("content", ""))
        preview = content if len(content) <= 600 else content[:600] + "\n..."
        mode = "overwrite" if arguments.get("overwrite") else "create"
        return f"{mode} the file {arguments.get('path')} ({len(content)} characters):\n\n{preview}"
    if call.name == "command_runner":
        return f"run the command:\n\n{arguments.get('command')}"
    return f"run {call.name} with {json.dumps(arguments, ensure_ascii=False)}"


__all__ = [
    "APPROVAL_TOOLS",
    "DEFAULT_UI_TOOLS",
    "default_tools",
    "describe_call",
    "format_step",
    "needs_approval",
]
