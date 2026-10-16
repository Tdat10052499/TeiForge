# Changelog

Notable changes to TeiForge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x releases may contain breaking changes).

## Unreleased

### Added

- `teiforge bench` and `teiforge.bench.run_bench()`: six fixed scenarios (direct answer, reading
  a file, extracting action items, a custom `@tool`, the workspace sandbox and chat history) run
  several times against the configured model. The report gives the pass rate with a 95% Wilson
  interval and classifies each failure (`text_tool_call`, `no_tool_call`, `wrong_answer`, ...).
- Tool calls written as text are recovered: when a model replies with only a tool call as plain
  text (bare JSON, a fenced code block, `<tool_call>` tags or Llama's `<|python_tag|>`) that names
  one of the agent's tools, the agent runs it instead of returning the JSON as the answer. The
  `tool_call` event carries `"recovered": true`. Controlled by the new
  `AgentConfig.recover_text_tool_calls` (default `True`) and `teiforge bench --no-recovery`.
  Text that only mentions a tool, or names a tool the agent does not have, is never run.
- Self-check: when every tool call in a run succeeded but the model's answer claims that a tool
  failed (for example "the tool call failed to extract relevant information"), the agent asks the
  model once to answer again from the tool results. A new `self_check` event records the first
  answer. Controlled by `AgentConfig.self_check` (default `True`) and `teiforge bench --no-self-check`.
  The check is a phrase match, so an answer that quotes a failure from the data itself can trigger
  one extra model call.

## 0.3.1 - 2026-10-14

Fixes found by installing 0.3.0 from PyPI and following the README on Windows with Ollama.

### Fixed

- When no LLM provider was configured, the agent silently fell back to the keyword router and
  answered "I don't have a direct tool for this task yet", even when a matching custom tool was
  registered. It now logs a one-time warning (`No LLM provider is configured ...`) and the answer
  explains that no model is configured, how to configure one, and that custom tools need an LLM.
  The warning is not shown with `mode="router"`, `--no-llm` or `provider="none"`.
- A task the keyword router cannot handle is now reported with `ok=False` (CLI exit code 1)
  instead of `ok=True`.
- `Agent.model`, `str(agent)` and the CLI showed the configured default (`gpt-4o-mini`) while the
  agent actually used `TEIFORGE_MODEL` (for example `llama3.1`). They now show the model in use.
- `file_reader` passed the UTF-8 byte-order mark written by Windows PowerShell 5 and some editors
  to the model. It is now removed.

### Changed

- Rewrote the start of the README for new users: what TeiForge is for, three short examples,
  a free local quickstart with Ollama, an offline fallback, and when to choose another framework.
- The README examples use `temperature=0`, and the quickstart explains that environment variables
  only last for the current terminal and how to check that TeiForge sees the model.
- Python 3.14 is tested in CI and listed as supported.
- Dependabot no longer opens version-update PRs for Python dependencies, which only raised
  TeiForge's lower bounds; GitHub Actions updates continue.

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
- The default system prompt told the model to cite web search results even when the agent had no
  `web_search` tool, which led small models to claim they were searching. The prompt now lists only
  the tools the agent has, says so when it has none, and asks the model to call tools through the
  tool-calling interface rather than writing the call as text.

### Known limitations

- Provider adapters are tested with mocked clients. A live smoke test passed with Ollama
  (`llama3.1` 8B) through `OpenAIProvider`; the OpenAI cloud API and `AnthropicProvider` have not
  been tested against the live services yet.
- Small local models sometimes write a tool call as plain JSON text instead of using tool calling.
  TeiForge then returns that text as the final answer. Use a low `temperature` (for example `0`)
  and a model with good tool-calling support.

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
