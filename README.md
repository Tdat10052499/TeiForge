# AgentForge

AgentForge is a lightweight Python toolkit for building agents with tools, short-term memory, web search, and simple workflows. It includes a Python API and a command-line interface.

## Features

- Register custom tools with a small, typed interface.
- Search the web with DuckDuckGo.
- Optionally summarize search results with OpenAI.
- Read and write local files.
- Keep a bounded in-memory conversation history.
- Run predefined workflows from Python or the CLI.

AgentForge is an early-stage project. Its agent currently routes tasks using keywords; it does not provide autonomous planning or general-purpose model-driven tool calling.

## Installation

```shell
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell on Windows, or `source .venv/bin/activate` on macOS/Linux), then install:

```shell
python -m pip install -e .
```

For development tools and tests:

```shell
python -m pip install -e ".[dev]"
```

To enable OpenAI-powered summaries, install the optional client and set an API key:

```shell
python -m pip install openai
```

Set `OPENAI_API_KEY` in your environment. `AGENTFORGE_MODEL` optionally selects the model and defaults to `gpt-4o-mini`. If OpenAI is unavailable or no key is configured, search results are returned without an LLM summary.

## Quick start

```python
from agentforge import Agent, WebSearchTool

agent = Agent(name="research-agent")
agent.add_tool(WebSearchTool)
print(agent.run("Search for open-source Python agent frameworks"))
```

See the [`examples/`](examples/) directory for research, file, and workflow examples.

## CLI

```shell
python -m agentforge.cli --agent research --task "Search for open-source Python agent frameworks"
python -m agentforge.cli --agent research --task "Search for agent frameworks" --out results.txt --no-llm
python -m agentforge.cli --agent file-analyzer --task "read:demo.txt"
python -m agentforge.cli --agent workflow --task "Search for AI agent libraries" --verbose
```

The `--out` option saves raw search results and, when available, an LLM summary. `--no-llm` disables summarization for that run.

## Built-in tools

`WebSearchTool`, `FileReaderTool`, and `FileWriterTool` are exported from the `agentforge` package. `CommandRunnerTool` is also available, but is disabled unless `AGENTFORGE_ENABLE_COMMANDS=1` is set.

When enabled, the command runner supports only `echo`, `pwd`, `ls [path]`, and `python`/`python3 --version` (or `-V`). It does not invoke a shell, execute arbitrary programs, or accept arbitrary Python code. Keep it disabled unless the restricted commands are needed.

```python
from agentforge import Agent, CommandRunnerTool

agent = Agent(name="local-agent")
agent.add_tool(CommandRunnerTool)
print(agent.run("run pwd"))
```

## Configuration

The project includes [`.env.example`](.env.example) as a reference. AgentForge does not load `.env` files automatically; export the variables in your shell or load them with your preferred environment-variable manager.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Enables OpenAI summaries when the `openai` package is installed. | Unset |
| `AGENTFORGE_MODEL` | OpenAI model used for summaries. | `gpt-4o-mini` |
| `AGENTFORGE_ENABLE_COMMANDS` | Enables the restricted command runner. | `0` |

## Development

Run the test suite with:

```shell
python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution guidelines. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

AgentForge is distributed under the MIT License. See [LICENSE](LICENSE).
