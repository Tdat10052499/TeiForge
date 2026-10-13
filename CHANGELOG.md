# Changelog

Notable changes to TeiForge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x releases may contain breaking changes).

## Unreleased

## 0.3.0 - 2026-10-13

The project is renamed from **AgentForge** to **TeiForge** because the `agentforge` name is
already used on PyPI by an unrelated project. See "Upgrading from AgentForge 0.2" in the README.

### Added

- LLM tool-calling loop: the model chooses tools, sees their results and writes the final answer,
  limited by `max_steps`.
- Provider layer with `OpenAIProvider` (including OpenAI-compatible servers such as Ollama through
  `base_url`) and `AnthropicProvider`; `teiforge.testing.ScriptedProvider` for offline tests.
- `@tool` decorator that builds a JSON Schema from type hints and Google-style docstrings.
- `ToolResult` and `ToolError` for explicit success and failure.
- `Agent.run_detailed()` returning a `RunResult` with structured events, and an `on_event` callback.
- Workspace sandbox for file tools (`workspace` option and `TEIFORGE_WORKSPACE`).
- `Memory.save()`, `Memory.load()` and `Memory.to_messages()`; memory is now part of the prompt.
- `AgentConfig.from_file()` for TOML and JSON, plus the `provider`, `base_url`, `max_steps`,
  `workspace` and `mode` settings.
- CLI subcommands `run`, `search`, `chat`, `tools` and `version`, the `teiforge` console script,
  and `python -m teiforge`.
- `py.typed`, security policy, code of conduct, issue and pull request templates, Dependabot,
  and a release workflow that publishes to PyPI on version tags.

### Changed

- The keyword router matches whole words only, so "findings" no longer triggers web search,
  "return" no longer triggers the command runner, and "profile" no longer triggers the file reader.
- The router finds file paths inside sentences ("Read demo.txt if it exists") and accepts
  `read:`/`write:`/`search:` anywhere in the task.
- `Tool.run()` returns a `ToolResult` instead of a string.
- `file_writer` refuses to overwrite existing files unless `overwrite=true`, and keeps
  leading and trailing whitespace in the content.
- `command_runner` resolves `ls` and `pwd` against the workspace.
- Environment variables use the `TEIFORGE_` prefix.
- CI runs ruff (lint and format), mypy and tests with an 85% coverage threshold on
  Python 3.10-3.13, plus Windows and macOS.
- The version is defined once, in `teiforge/_version.py`.

### Fixed

- `write:C:\path\file.txt:content` wrote to a file named `C` on Windows.
- File tools could read and write any path on the machine.
- `search for` with nothing after it searched for the word "for".

### Removed

- Python 3.9 support (end of life since October 2025).
- Generated `results.txt` and `demo.txt` files from the repository.

## 0.2.0 - 2026-10-13

Released as AgentForge.

### Added

- OpenAI-powered summaries for web search results when the optional client and API key are configured.
- DuckDuckGo HTML result parsing as a fallback when the Instant Answer API has no results.
- CLI options to save search results and disable LLM summarization.

### Changed

- Restricted command execution to a small set of built-in operations and Python version reporting;
  commands are never passed through a shell.
- Made search-and-save failures explicit and kept task history consistent.
- Connected `AgentConfig` to agent construction, memory sizing, built-in tool selection, and
  OpenAI summary settings.
