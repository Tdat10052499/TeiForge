# TeiForge - instructions for Claude Code

TeiForge is a lightweight Python toolkit for building AI agents (tools, memory, an LLM
tool-calling loop, and a keyword router fallback). Package: `teiforge/`. Python 3.10-3.13.

## Communication

- The maintainer prefers **Vietnamese** in conversation. Code, identifiers, commit messages,
  docstrings, README, CHANGELOG and other public docs stay in **English**.
- Planning documents in `docs/plans/` are written in Vietnamese.
- Separate verified facts (a command you ran, a file you read) from assumptions. If something
  was not run or checked, say so explicitly.

## Layout

| Path | Purpose |
| --- | --- |
| `teiforge/agent.py` | `Agent`: LLM loop (`_run_llm`), keyword router (`route`, `_run_router`), `run_detailed` |
| `teiforge/tools.py` | `Tool`, `@tool`, `ToolResult`, `ToolError`, built-in tools |
| `teiforge/llm.py` | `OpenAIProvider`, `AnthropicProvider`, `provider_from_config` |
| `teiforge/workspace.py` | Sandbox that confines file tools to one directory |
| `teiforge/events.py` | `Event`, `RunResult` (the stream a UI will consume) |
| `teiforge/memory.py`, `config.py`, `cli.py` | Memory, `AgentConfig`, the `teiforge` CLI |
| `teiforge/testing.py` | `ScriptedProvider` for offline tests |
| `teiforge/_version.py` | The only place the version is defined |
| `docs/plans/` | Versioned plans (`v0.3-plan.md`, ...) with acceptance criteria |

## Commands

Set up once (Windows PowerShell shown; use `source .venv/bin/activate` on macOS/Linux):

```shell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,llm]"
```

Quality gate, the same as CI. All four must pass before a commit:

```shell
ruff check teiforge tests examples
ruff format --check teiforge tests examples
mypy teiforge
python -m pytest --cov=teiforge --cov-fail-under=85
```

## Rules

- **Tests never use the network or a real LLM.** Mock `requests.get`; use
  `teiforge.testing.ScriptedProvider` for model responses. Real-API checks live only in
  `/tf-smoke-llm` and are never committed as tests.
- Every behaviour change needs a test; every bug fix needs a regression test that fails first.
- Keep coverage at or above 85% and `mypy teiforge` clean (`disallow_untyped_defs` is on).
- Public API changes need a README update and an entry under `## Unreleased` in `CHANGELOG.md`.
- Keep the core dependency list minimal (`requests`, plus `tomli` on 3.10). New providers or
  UI dependencies go into optional extras in `pyproject.toml`.
- Security model (see `SECURITY.md`): file tools must stay inside the `Workspace`;
  `file_writer` must not overwrite without `overwrite=True`; `command_runner` stays off unless
  `TEIFORGE_ENABLE_COMMANDS=1`, never uses a shell, and keeps its allow-list.
- Never print, log, commit or write API keys or tokens to any file, including `.mcp.json`.
  Read them from environment variables only.
- Do not delete or rewrite files under `docs/plans/` for past versions; add new plan files.

## Git workflow

- Work on a branch, never directly on `main`: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`.
- Small, focused commits with imperative English messages.
- Ask before any irreversible or outward-facing action: merging a PR, pushing a tag,
  publishing a release, deleting a branch, force-pushing.
- Open pull requests as drafts unless asked otherwise, and fill in
  `.github/pull_request_template.md`.

## Windows notes

- The maintainer develops on Windows with PowerShell. Use PowerShell syntax in instructions,
  and `pathlib` (never hard-coded `/` or `\`) in code.
- Tests must pass on Windows: CI runs `windows-latest`.
