"""The per-user configuration that the terminal UI reads and writes.

It lives in ``~/.teiforge/config.toml`` (or ``$TEIFORGE_HOME/config.toml``) and holds
an ``[agent]`` table in the format of :meth:`teiforge.AgentConfig.from_file`. API keys
are never stored here; providers keep reading them from environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from .config import AgentConfig

HOME_ENV_VAR = "TEIFORGE_HOME"

#: Settings written by :func:`save_user_config`, in this order.
SAVED_FIELDS = ("name", "provider", "model", "base_url", "temperature", "workspace", "tools", "max_steps")


def config_dir() -> Path:
    """Return the TeiForge settings folder: ``$TEIFORGE_HOME`` or ``~/.teiforge``."""
    override = os.getenv(HOME_ENV_VAR)
    return Path(override) if override else Path.home() / ".teiforge"


def config_path() -> Path:
    return config_dir() / "config.toml"


def chat_path() -> Path:
    """Where the terminal UI keeps the conversation between sessions."""
    return config_dir() / "chat.json"


def load_user_config(path: Union[str, os.PathLike[str], None] = None) -> Optional[AgentConfig]:
    """Load the user configuration, or return ``None`` when there is none yet."""
    target = Path(path) if path is not None else config_path()
    if not target.is_file():
        return None
    return AgentConfig.from_file(target)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # JSON string escapes are valid TOML basic-string escapes.
    return json.dumps(str(value), ensure_ascii=False)


def save_user_config(config: AgentConfig, path: Union[str, os.PathLike[str], None] = None) -> Path:
    """Write the settings in :data:`SAVED_FIELDS` to the user configuration file."""
    target = Path(path) if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TeiForge settings, written by the terminal UI. API keys are never stored here.", "[agent]"]
    for name in SAVED_FIELDS:
        value = getattr(config, name)
        if value is not None:
            lines.append(f"{name} = {_toml_value(value)}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "HOME_ENV_VAR",
    "SAVED_FIELDS",
    "chat_path",
    "config_dir",
    "config_path",
    "load_user_config",
    "save_user_config",
]
