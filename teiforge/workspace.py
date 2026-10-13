"""Workspace sandbox that keeps file tools inside one directory tree."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

WORKSPACE_ENV_VAR = "TEIFORGE_WORKSPACE"


class WorkspaceError(ValueError):
    """Raised when a path resolves outside the workspace."""


class Workspace:
    """A directory that file tools are confined to.

    Relative paths are resolved against ``root``. Absolute paths are accepted only
    when they point inside ``root``. ``..`` segments and symbolic links that would
    escape the root are rejected.

    When ``root`` is ``None`` the workspace is resolved at call time from the
    ``TEIFORGE_WORKSPACE`` environment variable, falling back to the current
    working directory.
    """

    def __init__(self, root: Optional[PathLike] = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root.expanduser().resolve()
        env_root = os.getenv(WORKSPACE_ENV_VAR)
        if env_root:
            return Path(env_root).expanduser().resolve()
        return Path.cwd().resolve()

    def resolve(self, path: PathLike) -> Path:
        """Return the absolute path for ``path`` or raise ``WorkspaceError``."""
        raw = str(path).strip()
        if not raw:
            raise WorkspaceError("No file path was provided.")

        root = self.root
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()

        if resolved != root and root not in resolved.parents:
            raise WorkspaceError(
                f"Path '{raw}' is outside the workspace '{root}'. "
                f"Set {WORKSPACE_ENV_VAR} or pass workspace=... to change it."
            )
        return resolved

    def relative(self, path: Path) -> str:
        """Return ``path`` relative to the root, for display."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def __repr__(self) -> str:
        return f"Workspace(root={self._root!r})"


__all__ = ["Workspace", "WorkspaceError", "WORKSPACE_ENV_VAR"]
