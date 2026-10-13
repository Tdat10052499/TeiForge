# Contributing to TeiForge

Thank you for your interest in contributing. By taking part you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and suggesting features

Check the existing [issues](https://github.com/Tdat10052499/teiforge/issues) first, then
open a new one with the bug report or feature request template. Security problems should
be reported privately, as described in [SECURITY.md](SECURITY.md).

## Development setup

```shell
git clone https://github.com/Tdat10052499/teiforge.git
cd teiforge
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux), then install TeiForge with the development tools:

```shell
python -m pip install -e ".[dev]"
```

## Checks

Run the same checks as CI before opening a pull request:

```shell
ruff check teiforge tests examples
ruff format teiforge tests examples
mypy teiforge
python -m pytest --cov=teiforge
```

Tests never touch the network or a real LLM. Use `unittest.mock` for HTTP calls and
`teiforge.testing.ScriptedProvider` for model responses. Coverage must stay at or above 85%.

## Pull requests

1. Create a focused branch from `main`.
2. Add or update tests for every behaviour change.
3. Update `README.md` and the `Unreleased` section of `CHANGELOG.md` for user-facing changes.
4. Open the pull request and fill in the template.

Use type hints, keep public functions documented, and keep changes small enough to review.

## Releasing (maintainer)

1. Update `teiforge/_version.py` and move the `Unreleased` notes in `CHANGELOG.md` under the
   new version.
2. Merge to `main`, then push a tag such as `v0.3.0`. The `release.yml` workflow builds the
   package, publishes it to PyPI and creates a GitHub release.
