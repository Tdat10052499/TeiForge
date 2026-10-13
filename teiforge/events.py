"""Structured events emitted while an agent runs.

Every run produces a list of :class:`Event` objects. Front ends (the CLI today, a
web UI later) and logging use the same stream, so they never have to parse the
agent's text output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal

EventType = Literal["task_start", "llm_response", "tool_call", "tool_result", "final", "error"]

EVENT_TYPES = ("task_start", "llm_response", "tool_call", "tool_result", "final", "error")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Event:
    """One step of an agent run."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp.isoformat()}


EventHandler = Callable[[Event], None]


@dataclass
class RunResult:
    """The full outcome of :meth:`teiforge.Agent.run_detailed`."""

    task: str
    output: str
    ok: bool
    mode: str
    steps: int = 0
    events: List[Event] = field(default_factory=list)

    def tool_calls(self) -> List[Event]:
        """Return only the ``tool_call`` events."""
        return [event for event in self.events if event.type == "tool_call"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "output": self.output,
            "ok": self.ok,
            "mode": self.mode,
            "steps": self.steps,
            "events": [event.to_dict() for event in self.events],
        }

    def __str__(self) -> str:
        return self.output


__all__ = ["Event", "EventType", "EventHandler", "RunResult", "EVENT_TYPES"]
