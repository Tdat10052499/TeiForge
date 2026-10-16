# TeiForge

[![PyPI](https://img.shields.io/pypi/v/teiforge)](https://pypi.org/project/teiforge/)
[![CI](https://github.com/Tdat10052499/teiforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Tdat10052499/teiforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A small, readable Python toolkit for building AI agents that can safely work with your files.**

Give a language model a few tools (read a file, write a file, search the web, or any Python
function you write) and TeiForge runs the loop: the model picks a tool, TeiForge runs it inside a
sandbox, the model reads the result and answers.

- **Runs free and private on your machine** with [Ollama](https://ollama.com), or with OpenAI and Anthropic.
- **Safe by default**: file tools cannot leave the folder you choose and never overwrite files unless asked.
- **Small enough to read**: about 2,000 lines of typed, documented Python and one required dependency (`requests`).
- **Works without a model too**: a keyword router handles simple tasks offline.

*"Tei" comes from Teichi, the author's nickname (Hồ Du Tuấn Đạt). TeiForge is not related to
Hugging Face Text Embeddings Inference (TEI).*

> **Status: alpha (0.3).** The API may still change. See [Known limitations](CHANGELOG.md).

## What can you build with it?

**Ask questions about your own documents**, without the model being able to touch anything else:

```python
from teiforge import Agent, AgentConfig

agent = Agent(config=AgentConfig(name="reader", tools=["file_reader"], workspace="notes", temperature=0))
print(agent.run("Read meeting.txt and list the action items with their owners."))
```

**Give the model your own functions.** Type hints and the docstring become the tool description:

```python
import datetime

from teiforge import Agent, AgentConfig, ToolError, tool


@tool
def days_until(date: str) -> str:
    """Count the days from today until a date.

    Args:
        date: The target date in YYYY-MM-DD format.
    """
    try:
        target = datetime.date.fromisoformat(date)
    except ValueError:
        raise ToolError(f"'{date}' is not a YYYY-MM-DD date.") from None
    return f"{(target - datetime.date.today()).days} days"


agent = Agent(config=AgentConfig(name="planner", temperature=0), tools=[days_until])
print(agent.run("How many days are left until 2027-01-01?"))
```

**Chat with an assistant that remembers the conversation**, from the terminal:

```shell
teiforge chat --agent files --workspace notes --memory chat.json
```

Type `/help` inside the chat for commands. The conversation is saved to `chat.json` and picked
up again next time.

## Quickstart: free, on your own machine (5 minutes)

**1. Install TeiForge and a local model**

Install [Ollama](https://ollama.com/download), then:

```shell
python -m pip install "teiforge[openai]"
ollama pull llama3.1
```

`teiforge[openai]` installs the OpenAI client, which TeiForge also uses to talk to Ollama.
No API key or payment is needed.

**2. Point TeiForge at Ollama**

PowerShell (Windows):

```powershell
$env:TEIFORGE_PROVIDER = "openai"
$env:TEIFORGE_BASE_URL = "http://127.0.0.1:11434/v1"
$env:TEIFORGE_MODEL = "llama3.1"
```

macOS / Linux:

```shell
export TEIFORGE_PROVIDER=openai
export TEIFORGE_BASE_URL=http://127.0.0.1:11434/v1
export TEIFORGE_MODEL=llama3.1
```

These variables only last for the current terminal window. Run every command below in that same
window, or set them permanently (PowerShell: `setx TEIFORGE_PROVIDER openai`, and likewise for the
other two, then open a new window; macOS / Linux: add the `export` lines to `~/.bashrc` or `~/.zshrc`).

Check that TeiForge sees the model. This should print `OpenAIProvider llama3.1`:

```shell
python -c "from teiforge import Agent, AgentConfig; p = Agent(config=AgentConfig(name='check')).provider; print(type(p).__name__, getattr(p, 'model', None))"
```

If it prints `NoneType None`, the variables are not set in this window. Without a model, TeiForge
falls back to its keyword router and warns `No LLM provider is configured`.

**3. Try it**

Create a folder `notes` with a text file `meeting.txt` in it, then:

```shell
teiforge run "Read meeting.txt and summarise it in three bullet points." --agent files --workspace notes --verbose
```

`--verbose` shows each step: the tool the model called, whether it worked, and the final answer.
To see the sandbox at work, ask for a file outside `notes`; it is refused before anything is read:

```shell
teiforge run "Read ../secret.txt" --agent files --workspace notes --no-llm
```

**Tip for local models:** small models (around 8B parameters) are not always reliable: they may
misread a tool result or write a tool call as plain text instead of calling the tool. A
temperature of `0` makes them more consistent; if an answer looks wrong, run it again or try a
larger model. In Python, pass
`AgentConfig(..., temperature=0)`; for the CLI, put it in a config file (see [CLI](#cli)).

### No model at all?

Without any provider configured, TeiForge falls back to a keyword router, which is handy for
scripts and tests:

```shell
teiforge run "read:meeting.txt" --agent files --workspace notes --no-llm
```

### Using OpenAI or Anthropic instead

```shell
python -m pip install "teiforge[openai]"       # or "teiforge[anthropic]"
```

Then set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) in your environment instead of the Ollama
variables above. These services bill your own API key; see [Cost](#cost).

## Why TeiForge?

| If you want... | TeiForge |
| --- | --- |
| To understand how an agent loop works by reading the code | Yes: the LLM loop is one function of about 50 lines (`Agent._run_llm` in `teiforge/agent.py`) |
| An assistant that works on your files without risking the rest of your disk | Yes: every file path is checked against the workspace |
| To run on a laptop with a local model | Yes: Ollama and other OpenAI-compatible servers work out of the box |
| Hundreds of integrations, vector stores, multi-agent orchestration | No. Larger frameworks such as LangChain, LlamaIndex or smolagents cover that |

---

## Guide

### Your own tools

Any typed function becomes a tool with `@tool`. Supported parameter types are `str`, `int`,
`float`, `bool`, `list[...]`, `dict`, `Optional[...]` and `Literal[...]`. Report a problem by
raising `teiforge.ToolError`; the model receives the message and can react to it instead of the
agent crashing. You can pass functions directly with `Agent(..., tools=[my_function])` or
`agent.add_tool(my_function)`.

### Seeing what the agent does

```python
result = agent.run_detailed("Read notes.txt and summarize it", on_event=print)
result.ok           # True / False
result.mode         # "llm" or "router"
result.tool_calls() # the tool_call events
result.to_dict()    # JSON-serialisable record of the whole run
```

Event types: `task_start`, `llm_response`, `tool_call`, `tool_result`, `final`, `error`.

### LLM providers

| Provider | How to enable |
| --- | --- |
| Ollama, LM Studio, vLLM, OpenRouter | `teiforge[openai]`, `TEIFORGE_PROVIDER=openai`, `TEIFORGE_BASE_URL` and `TEIFORGE_MODEL` |
| OpenAI | `teiforge[openai]` and `OPENAI_API_KEY` |
| Anthropic | `teiforge[anthropic]` and `ANTHROPIC_API_KEY` |

You can also pass a provider object directly:

```python
from teiforge import Agent, AgentConfig, OpenAIProvider

local = OpenAIProvider("llama3.1", base_url="http://127.0.0.1:11434/v1", temperature=0)
agent = Agent(config=AgentConfig(name="local", tools=["file_reader"]), provider=local)
```

For tests, `teiforge.testing.ScriptedProvider` replays a fixed list of responses without any network.

### Measuring how well your model works

Small local models do not always call tools correctly. `teiforge bench` runs six fixed tasks
several times against the model you configured and reports how often each one passed, with a 95%
confidence interval and the reason for every failure:

```shell
teiforge bench --runs 10                 # all scenarios, 10 runs each
teiforge bench --list                    # scenarios and failure reasons
teiforge bench --scenario custom_tool --temperature 0 --json
teiforge bench --runs 10 --out bench.json   # table on screen, full report with answers in a file
```

```text
scenario         passed       95% CI  failures
answer_only       10/10     72%-100%  -
read_summary       8/10      49%-94%  text_tool_call x2
...
```

Every run uses a fresh temporary workspace and calls the model for real, so it takes a few minutes
on a laptop (and costs money with a paid API). Compare results only between runs that use the same
model, provider and temperature. The table above is an illustration, not a measurement.

### CLI

```shell
teiforge run "Read README.md and list the features" --agent files --workspace .
teiforge run "search: python 3.13 release" --json          # full result with events
teiforge search "AI agent libraries" --out results.txt --no-llm
teiforge chat --agent all --memory chat.json             # interactive; type /help
teiforge tools                                           # list tools and parameters
teiforge bench --runs 10                                 # measure the configured model
```

Common options: `--agent research|files|all`, `--tools web_search,file_reader`,
`--config agent.toml`, `--workspace DIR`, `--provider`, `--model`, `--base-url`,
`--max-steps`, `--no-llm`, `--verbose`. `python -m teiforge` works too.

A configuration file (TOML or JSON) sets everything at once, including `temperature`:

```toml
[agent]
name = "notes-assistant"
tools = ["file_reader", "file_writer"]
provider = "openai"
base_url = "http://127.0.0.1:11434/v1"
model = "llama3.1"
temperature = 0
max_steps = 6
workspace = "notes"
system_prompt = "You are a careful assistant that answers from the user's notes."
```

```shell
teiforge chat --config agent.toml --memory chat.json
```

### Built-in tools

| Tool | Parameters | Notes |
| --- | --- | --- |
| `file_reader` | `path` | UTF-8 text files inside the workspace. |
| `file_writer` | `path`, `content`, `overwrite` | Refuses to replace a file unless `overwrite` is true. |
| `web_search` | `query`, `max_results` | DuckDuckGo, no API key. Results come from DuckDuckGo's public pages and may be rate-limited. |
| `command_runner` | `command` | Off unless `TEIFORGE_ENABLE_COMMANDS=1`; only `echo`, `pwd`, `ls [path]`, `python --version`; no shell. |

In router mode you can address tools explicitly: `search: <query>`, `read:<path>`,
`write:<path>:<content>` (Windows paths such as `C:\notes\a.txt` work).

### Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `TEIFORGE_PROVIDER` | `openai`, `anthropic` or `none` | Auto-detect from API keys |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Provider credentials | Unset |
| `TEIFORGE_MODEL` | Model name override | `gpt-4o-mini` (OpenAI), `claude-sonnet-5-5` (Anthropic) |
| `TEIFORGE_BASE_URL` | OpenAI-compatible endpoint, e.g. Ollama | Unset |
| `TEIFORGE_WORKSPACE` | Directory file tools are confined to | Current directory |
| `TEIFORGE_ENABLE_COMMANDS` | Enable the restricted command runner | `0` |

See [`.env.example`](.env.example). TeiForge does not load `.env` files itself.

### Cost

TeiForge is free. Local models through Ollama cost nothing beyond your own hardware. With OpenAI
or Anthropic, every step of the agent loop is one API call billed to your key, and each call
resends the conversation, tool descriptions and tool results, so long files and long chats cost
more. Keep `max_steps` low, use a narrow workspace, and set a spending limit in your provider's
console. TeiForge does not yet report token usage.

### Security

Agents act on model output, and web pages or files can contain prompt injections. Keep the
workspace narrow, enable only the tools a task needs, and read [SECURITY.md](SECURITY.md).

### Upgrading from AgentForge 0.2

- Install `teiforge` and change imports from `agentforge` to `teiforge`.
- `Tool.run()` now returns a `ToolResult` (`ok`, `output`, `error`) instead of a string;
  `str(result)` gives the old text. `Agent.run()` still returns a string.
- The CLI uses subcommands: `teiforge run "task"` replaces `--agent research --task "task"`
  (the old flags still work but print a deprecation note).
- Environment variables were renamed: `AGENTFORGE_MODEL` -> `TEIFORGE_MODEL`,
  `AGENTFORGE_ENABLE_COMMANDS` -> `TEIFORGE_ENABLE_COMMANDS`.
- File tools are now confined to the workspace, and `file_writer` no longer overwrites by default.
- Python 3.10 or newer is required.

## Contributing

Bug reports, questions and pull requests are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md);
the [changelog](CHANGELOG.md) lists what changed in each release. For development:

```shell
git clone https://github.com/Tdat10052499/teiforge.git
cd teiforge
python -m pip install -e ".[dev]"
python -m pytest
```

## License

TeiForge is distributed under the MIT License. See [LICENSE](LICENSE).
