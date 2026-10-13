"""Conversation memory for agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class MemoryEntry:
    role: str
    content: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class Memory:
    """A bounded list of conversation turns that can be saved to and loaded from JSON."""

    def __init__(self, max_entries: int = 100) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")
        self.history: List[MemoryEntry] = []
        self.max_entries = max_entries

    def add(self, message: str, role: str = "assistant") -> None:
        self.history.append(MemoryEntry(role=role, content=message, timestamp=datetime.now()))
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries :]

    def get(self) -> List[MemoryEntry]:
        return self.history.copy()

    def get_context(self, limit: int = 10) -> str:
        if not self.history:
            return "No previous context."
        return "\n".join(f"{entry.role.capitalize()}: {entry.content}" for entry in self.history[-limit:])

    def to_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Return user and assistant turns as chat messages for an LLM prompt."""
        entries = self.history if limit is None else self.history[-limit:]
        return [
            {"role": entry.role, "content": entry.content}
            for entry in entries
            if entry.role in {"user", "assistant"}
        ]

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Write the history to ``path`` as JSON."""
        payload = {
            "version": 1,
            "max_entries": self.max_entries,
            "entries": [entry.to_dict() for entry in self.history],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, os.PathLike[str]], max_entries: Optional[int] = None) -> Memory:
        """Read a history written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls(max_entries=max_entries or int(payload.get("max_entries", 100)))
        memory.history = [MemoryEntry.from_dict(item) for item in payload.get("entries", [])]
        memory.history = memory.history[-memory.max_entries :]
        return memory

    def clear(self) -> None:
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)

    def __str__(self) -> str:
        return f"Memory({len(self.history)} entries)"


__all__ = ["Memory", "MemoryEntry"]
