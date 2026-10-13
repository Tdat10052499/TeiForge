"""Shared fixtures: keep every test offline, deterministic and out of the repository."""

import pytest

_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TEIFORGE_PROVIDER",
    "TEIFORGE_MODEL",
    "TEIFORGE_BASE_URL",
    "TEIFORGE_WORKSPACE",
    "TEIFORGE_ENABLE_COMMANDS",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path
