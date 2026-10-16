"""Full-screen terminal UI for TeiForge. Needs the ``tui`` extra: ``pip install "teiforge[tui]"``."""

from __future__ import annotations

INSTALL_HINT = 'The terminal UI needs Textual. Install it with:  python -m pip install "teiforge[tui]"'


def textual_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def run_tui() -> int:
    """Start the terminal UI; print an install hint and return 2 when Textual is missing."""
    if not textual_available():
        print(INSTALL_HINT)
        return 2
    from .app import run_tui as run

    return run()


__all__ = ["INSTALL_HINT", "run_tui", "textual_available"]
