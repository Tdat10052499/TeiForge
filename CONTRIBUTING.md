# Contributing to AgentForge

Thank you for your interest in contributing.

## Reporting bugs and suggesting features

Before opening an issue, check the existing [issues](https://github.com/Tdat10052499/agentforge/issues) and [discussions](https://github.com/Tdat10052499/agentforge/discussions). Include a clear description, steps to reproduce or use cases, expected and actual behavior, and your Python version and operating system where relevant.

## Development setup

```shell
git clone https://github.com/Tdat10052499/agentforge.git
cd agentforge
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\Activate.ps1` in PowerShell on Windows, or `source .venv/bin/activate` on macOS/Linux), then install the development dependencies:

```shell
python -m pip install -e ".[dev]"
```

Run the tests:

```shell
python -m pytest
```

## Pull requests

1. Create a focused feature branch.
2. Add or update tests for behavior changes.
3. Run the test suite and relevant formatting or lint checks.
4. Update user-facing documentation and `CHANGELOG.md` as needed.
5. Open a pull request describing the change and linking any related issue.

Follow the existing code style, use type hints where practical, and keep changes focused. Be respectful and constructive in project discussions.
