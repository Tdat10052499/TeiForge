# TeiForge

[![CI](https://github.com/Tdat10052499/teiforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Tdat10052499/teiforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

TeiForge is a lightweight Python toolkit for building AI agents that use tools and remember
the conversation. It works in two modes:

- **LLM mode**: a model (OpenAI, Anthropic, or any OpenAI-compatible server such as Ollama)
  decides which tools to call, reads their results and writes the answer.
- **Router mode**: with no model configured, a deterministic keyword router picks the tool,
  so the toolkit is still useful offline and in tests.

*"Tei" comes from Teichi, the author's nickname (Hồ Du Tuấn Đạt). TeiForge is not related to Hugging Face Text Embeddings Inference (TEI).*

## Features

- Turn any typed Python function into a tool with `@tool`; the JSON Schema is generated from
  type hints and the docstring.
- Built-in tools: web search (DuckDuckGo), file reader, file writer, and a restricted command runner.
- File tools are sandboxed to a workspace directory and never overwrite files unless asked.
- Every run returns structured events (`tool_call`, `tool_result`, `final`, ...) for logging or a UI.
- Conversation memory that is sent to the model and can be saved to and loaded from JSON.
- A CLI with one-shot runs, web search, and an interactive chat.
- Typed (`py.typed`), tested, and light: the core needs only `requests` (plus `tomli` on Python 3.10).

## Installation

```shell
python -m pip install teiforge                 # core, keyword router only
python -m pip install "teiforge[openai]"       # + OpenAI and OpenAI-compatible servers
python -m pip install "teiforge[anthropic]"    # + Anthropic
```

Until the first PyPI release, install from source:

```shell
git clone https://github.com/Tdat10052499/teiforge.git
cd teiforge
python -m pip install -e ".[openai]"
```

## Quick start

```python
from teiforge import Agent, AgentConfig

agent = Agent(config=AgentConfig(name="researcher", tools=["web_search", "file_reader"]))
print(agent.run("Search for open-source Python agent frameworks"))
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and the same code runs as an LLM agent. Without a
key it uses the keyword router.

### Your own tools

```python
from teiforge import Agent, tool

@tool
def convert_currency(amount: float, rate: float) -> str:
    """Convert an amount using an exchange rate.

    Args:
        amount: The amount to convert.
        rate: Units of the target currency per unit of the source currency.
    """
    return f"{amount * rate:.2f}"

agent = Agent(name="finance", tools=[convert_currency])
print(agent.run("How much is 120 USD in VND at a rate of 25400?"))  # needs an LLM provider
```

Tools report failures by raising `teiforge.ToolError`; the agent passes the message to the
model instead of crashing.

### Seeing what the agent does

```python
result = agent.run_detailed("Read notes.txt and summarize it", on_event=print)
result.ok           # True / False
result.mode         # "llm" or "router"
result.tool_calls() # the tool_call events
result.to_dict()    # JSON-serialisable record of the whole run
```

## LLM providers

| Provider | How to enable |
| --- | --- |
| OpenAI | `pip install "teiforge[openai]"` and set `OPENAI_API_KEY` |
| Anthropic | `pip install "teiforge[anthropic]"` and set `ANTHROPIC_API_KEY` |
| Ollama, LM Studio, vLLM, OpenRouter | `teiforge[openai]` plus `TEIFORGE_BASE_URL` (e.g. `http://localhost:11434/v1`) and `TEIFORGE_MODEL` |

You can also pass a provider object directly, which is how you plug in anything else:

```python
from teiforge import Agent, OpenAIProvider

local = OpenAIProvider("llama3.1", base_url="http://localhost:11434/v1")
agent = Agent(name="local", provider=local)
```

**Local models:** small models (around 8B parameters) sometimes write a tool call as plain JSON
text instead of using tool calling, and that text then comes back as the answer. Set a low
temperature (`AgentConfig(..., temperature=0)` or `OpenAIProvider(..., temperature=0)`) and pick
a model with good tool-calling support.

For tests, `teiforge.testing.ScriptedProvider` replays a fixed list of responses without any network.

## CLI

```shell
teiforge run "Search for AI agent frameworks"
teiforge run "Read README.md and list the features" --agent files --workspace .
teiforge run "search: python 3.13 release" --json          # full result with events
teiforge search "AI agent libraries" --out results.txt --no-llm
teiforge chat --agent all --memory chat.json             # interactive; type /help
teiforge tools                                           # list tools and parameters
```

Common options: `--agent research|files|all`, `--tools web_search,file_reader`,
`--config agent.toml`, `--workspace DIR`, `--provider`, `--model`, `--base-url`,
`--max-steps`, `--no-llm`, `--verbose`. `python -m teiforge` works too.

A configuration file looks like this (TOML or JSON):

```toml
[agent]
name = "researcher"
tools = ["web_search", "file_reader"]
provider = "openai"
model = "gpt-4o-mini"
max_steps = 6
workspace = "./data"
system_prompt = "You are a careful research assistant."
```

## Built-in tools

| Tool | Parameters | Notes |
| --- | --- | --- |
| `web_search` | `query`, `max_results` | DuckDuckGo Instant Answer API with an HTML fallback. |
| `file_reader` | `path` | UTF-8 text files inside the workspace. |
| `file_writer` | `path`, `content`, `overwrite` | Refuses to replace a file unless `overwrite` is true. |
| `command_runner` | `command` | Disabled unless `TEIFORGE_ENABLE_COMMANDS=1`; only `echo`, `pwd`, `ls [path]`, `python --version`; no shell. |

In router mode you can address tools explicitly: `search: <query>`, `read:<path>`,
`write:<path>:<content>` (Windows paths such as `C:\notes\a.txt` work).

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `TEIFORGE_PROVIDER` | `openai`, `anthropic` or `none` | Auto-detect from API keys |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Provider credentials | Unset |
| `TEIFORGE_MODEL` | Model name override | `gpt-4o-mini` (OpenAI), `claude-sonnet-5-5` (Anthropic) |
| `TEIFORGE_BASE_URL` | OpenAI-compatible endpoint | Unset |
| `TEIFORGE_WORKSPACE` | Directory file tools are confined to | Current directory |
| `TEIFORGE_ENABLE_COMMANDS` | Enable the restricted command runner | `0` |

See [`.env.example`](.env.example). TeiForge does not load `.env` files itself.

## Security

Agents act on model output, and web pages or files can contain prompt injections. Keep the
workspace narrow, enable only the tools a task needs, and read [SECURITY.md](SECURITY.md).

## Upgrading from AgentForge 0.2

- Install `teiforge` and change imports from `agentforge` to `teiforge`.
- `Tool.run()` now returns a `ToolResult` (`ok`, `output`, `error`) instead of a string;
  `str(result)` gives the old text. `Agent.run()` still returns a string.
- The CLI uses subcommands: `teiforge run "task"` replaces `--agent research --task "task"`
  (the old flags still work but print a deprecation note).
- Environment variables were renamed: `AGENTFORGE_MODEL` -> `TEIFORGE_MODEL`,
  `AGENTFORGE_ENABLE_COMMANDS` -> `TEIFORGE_ENABLE_COMMANDS`.
- File tools are now confined to the workspace, and `file_writer` no longer overwrites by default.
- Python 3.10 or newer is required.

## Development

```shell
python -m pip install -e ".[dev]"
python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md), the [changelog](CHANGELOG.md) and the
[v0.3 plan](docs/plans/v0.3-plan.md).

## License

TeiForge is distributed under the MIT License. See [LICENSE](LICENSE).
