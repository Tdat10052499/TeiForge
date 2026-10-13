"""Configuration management for AgentForge."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    memory_size: int = 100

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "memory_size": self.memory_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        return cls(**config_dict)


__all__ = ["AgentConfig"]
