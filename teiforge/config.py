"""Agent configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_MODEL = "gpt-4o-mini"
PROVIDERS = ("openai", "anthropic", "none")
MODES = ("auto", "llm", "router")


@dataclass
class AgentConfig:
    """Configuration for an :class:`~teiforge.Agent`.

    ``provider`` is ``None`` (auto-detect from the environment), ``"openai"``,
    ``"anthropic"`` or ``"none"``. ``mode`` is ``"auto"`` (use the LLM when one is
    configured, otherwise the keyword router), ``"llm"`` or ``"router"``.
    ``recover_text_tool_calls`` runs tool calls that the model wrote as plain text instead
    of through the tool-calling interface (see :func:`teiforge.llm.parse_text_tool_calls`).
    ``self_check`` asks the model once to answer again when every tool call succeeded but
    its answer claims that a tool failed.
    """

    name: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2000
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    memory_size: int = 100
    provider: Optional[str] = None
    base_url: Optional[str] = None
    max_steps: int = 6
    workspace: Optional[str] = None
    mode: str = "auto"
    recover_text_tool_calls: bool = True
    self_check: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty.")
        if not self.model.strip():
            raise ValueError("Agent model must not be empty.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Agent temperature must be between 0 and 2.")
        if self.max_tokens <= 0:
            raise ValueError("Agent max_tokens must be greater than 0.")
        if self.memory_size <= 0:
            raise ValueError("Agent memory_size must be greater than 0.")
        if self.max_steps <= 0:
            raise ValueError("Agent max_steps must be greater than 0.")
        if self.provider is not None and self.provider not in PROVIDERS:
            raise ValueError(f"Agent provider must be one of {', '.join(PROVIDERS)}.")
        if self.mode not in MODES:
            raise ValueError(f"Agent mode must be one of {', '.join(MODES)}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> AgentConfig:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(config_dict) - known)
        if unknown:
            raise ValueError(f"Unknown AgentConfig field(s): {', '.join(unknown)}")
        return cls(**config_dict)

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike[str]]) -> AgentConfig:
        """Load a configuration from a ``.toml`` or ``.json`` file.

        The settings may sit at the top level or under an ``[agent]`` table.
        """
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            data = json.loads(text)
        elif file_path.suffix.lower() == ".toml":
            data = _load_toml(text)
        else:
            raise ValueError("Configuration files must end in .toml or .json.")
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a table/object.")
        section = data.get("agent", data)
        if not isinstance(section, dict):
            raise ValueError("The [agent] section must be a table/object.")
        return cls.from_dict(section)


def _load_toml(text: str) -> Dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found,unused-ignore]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Reading TOML on Python 3.10 needs the 'tomli' package; use JSON or install tomli."
            ) from exc
    result: Dict[str, Any] = tomllib.loads(text)
    return result


__all__ = ["AgentConfig", "DEFAULT_MODEL", "PROVIDERS", "MODES"]
