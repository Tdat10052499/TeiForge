"""Command-line interface: ``teiforge run | search | chat | tools | version``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TextIO

from ._version import __version__
from .agent import Agent
from .bench import FAILURE_REASONS, SCENARIO_NAMES, SCENARIOS, run_bench
from .config import AgentConfig
from .events import Event
from .llm import LLMError
from .memory import Memory
from .tools import DEFAULT_TOOLS

PRESETS = {
    "research": ["web_search"],
    "files": ["file_reader", "file_writer"],
    "all": ["web_search", "file_reader", "file_writer", "command_runner"],
}
LEGACY_PRESETS = {"research": "research", "file-analyzer": "files", "workflow": "all"}

EPILOG = """\
examples:
  teiforge run "Search for open-source Python agent frameworks"
  teiforge run "read:README.md" --agent files
  teiforge search "AI agent libraries" --out results.txt --no-llm
  teiforge chat --agent all --memory chat.json
  teiforge tools
  teiforge bench --runs 10
"""


# -- helpers -------------------------------------------------------------------------


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("agent options")
    group.add_argument(
        "--agent", choices=sorted(PRESETS), default=None, help="Tool preset (default: research)."
    )
    group.add_argument(
        "--tools", default=None, help="Comma-separated built-in tools, e.g. web_search,file_reader."
    )
    group.add_argument("--config", default=None, help="Load an AgentConfig from a .toml or .json file.")
    group.add_argument("--workspace", default=None, help="Directory that file tools are confined to.")
    group.add_argument("--provider", choices=["openai", "anthropic", "none"], default=None)
    group.add_argument("--model", default=None, help="Model name for the LLM provider.")
    group.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. Ollama.")
    group.add_argument("--max-steps", type=int, default=None, help="Maximum tool-calling steps.")
    group.add_argument("--no-llm", action="store_true", help="Use the keyword router; never call an LLM.")


def build_agent(args: argparse.Namespace, default_preset: str = "research") -> Agent:
    """Create an agent from parsed CLI options."""
    if args.config:
        config = AgentConfig.from_file(args.config)
        if args.agent or args.tools:
            config.tools = _tool_names(args, default_preset)
    else:
        preset = args.agent or default_preset
        config = AgentConfig(name=f"{preset}-agent", tools=_tool_names(args, default_preset))

    if getattr(args, "provider", None):
        config.provider = args.provider
    if getattr(args, "model", None):
        config.model = args.model
    if getattr(args, "base_url", None):
        config.base_url = args.base_url
    if getattr(args, "max_steps", None):
        config.max_steps = args.max_steps
    if getattr(args, "no_llm", False):
        config.mode = "router"
        config.provider = "none"
    if getattr(args, "workspace", None):
        config.workspace = args.workspace
    config.__post_init__()
    return Agent(config=config)


def _tool_names(args: argparse.Namespace, default_preset: str) -> List[str]:
    if getattr(args, "tools", None):
        names = [name.strip() for name in args.tools.split(",") if name.strip()]
        unknown = [name for name in names if name not in DEFAULT_TOOLS]
        if unknown:
            raise ValueError(f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(DEFAULT_TOOLS)}")
        return names
    return list(PRESETS[getattr(args, "agent", None) or default_preset])


def event_printer(stream: TextIO) -> Callable[[Event], None]:
    """Return an event handler that prints tool activity in a compact form."""

    def handle(event: Event) -> None:
        data = event.data
        if event.type == "tool_call":
            arguments = json.dumps(data.get("arguments"), ensure_ascii=False)
            if len(arguments) > 120:
                arguments = arguments[:117] + "..."
            print(f"  -> {data.get('name')} {arguments}", file=stream)
        elif event.type == "tool_result":
            status = "ok" if data.get("ok") else "failed"
            size = len(str(data.get("output", "")))
            print(f"  <- {data.get('name')} {status} ({size} chars)", file=stream)
        elif event.type == "error":
            print(f"  !! {data.get('message')}", file=stream)
        stream.flush()

    return handle


def _mode_label(agent: Agent) -> str:
    try:
        provider = agent.provider if agent.enable_llm and agent.config.mode != "router" else None
    except LLMError as exc:
        return f"LLM unavailable ({exc})"
    if provider is None:
        return "keyword router (no LLM)"
    return f"LLM: {getattr(provider, 'name', 'custom')} / {getattr(provider, 'model', '?')}"


# -- commands ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    agent = build_agent(args)
    if args.verbose and not args.json:
        print(f"Agent: {agent}  [{_mode_label(agent)}]")
        print(agent.get_tools_description())
        print()
    handler = event_printer(sys.stderr) if args.verbose and not args.json else None
    result = agent.run_detailed(args.task, on_event=handler)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.output)
        if args.verbose:
            print()
            print(agent.get_memory_summary())
    return 0 if result.ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    args.agent, args.tools, args.config = "research", None, getattr(args, "config", None)
    agent = build_agent(args)
    if args.out:
        message = agent.run_search_and_save(args.query, args.out)
        print(message)
        return 1 if message.startswith("Error:") else 0
    result = agent.run_detailed(f"search: {args.query}")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.json else result.output)
    return 0 if result.ok else 1


CHAT_HELP = """\
Commands:
  /help           Show this help
  /tools          List the agent's tools
  /memory         Show the memory summary
  /save [FILE]    Save the conversation to a JSON file
  /clear          Forget the conversation
  /exit           Leave the chat (Ctrl-D also works)"""


def cmd_chat(
    args: argparse.Namespace,
    input_func: Callable[[str], str] = input,
    stream: TextIO = sys.stdout,
) -> int:
    agent = build_agent(args, default_preset="all")
    memory_path: Optional[Path] = Path(args.memory) if args.memory else None
    if memory_path and memory_path.exists():
        agent.memory = Memory.load(memory_path, max_entries=agent.config.memory_size)

    print(f"TeiForge {__version__} chat with '{agent.name}' [{_mode_label(agent)}]", file=stream)
    print("Type /help for commands, /exit to quit.", file=stream)
    handler = event_printer(stream)

    while True:
        try:
            line = input_func("you> ")
        except (EOFError, KeyboardInterrupt):
            print(file=stream)
            break
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            command, _, rest = text.partition(" ")
            command = command.lower()
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                print(CHAT_HELP, file=stream)
            elif command == "/tools":
                print(agent.get_tools_description(), file=stream)
            elif command == "/memory":
                print(agent.get_memory_summary(), file=stream)
            elif command == "/clear":
                agent.memory.clear()
                print("Conversation cleared.", file=stream)
            elif command == "/save":
                target = Path(rest.strip()) if rest.strip() else memory_path
                if target is None:
                    print("Usage: /save FILE", file=stream)
                else:
                    agent.memory.save(target)
                    print(f"Saved {len(agent.memory)} entries to {target}", file=stream)
            else:
                print(f"Unknown command {command}. Type /help.", file=stream)
            continue

        result = agent.run_detailed(text, on_event=handler)
        print(f"{agent.name}> {result.output}", file=stream)

    if memory_path:
        agent.memory.save(memory_path)
        print(f"Conversation saved to {memory_path}", file=stream)
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps([item.to_openai_schema() for item in DEFAULT_TOOLS.values()], indent=2))
        return 0
    for name, item in DEFAULT_TOOLS.items():
        params = ", ".join(item.parameters.get("properties", {}))
        print(f"{name}({params})\n    {item.description}")
    print("\ncommand_runner is disabled unless TEIFORGE_ENABLE_COMMANDS=1.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.name:<14} {scenario.description}")
        print("\nFailure reasons:")
        for reason, meaning in FAILURE_REASONS.items():
            print(f"  {reason:<19} {meaning}")
        return 0
    selected = [item for item in SCENARIOS if not args.scenario or item.name in args.scenario]
    config = AgentConfig(name="bench-agent", temperature=args.temperature)
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    config.__post_init__()

    def progress(_: str, reason: Optional[str]) -> None:
        print("." if reason is None else "F", end="", file=sys.stderr, flush=True)

    print(f"Running {len(selected)} scenario(s) x {args.runs} run(s)...", file=sys.stderr)
    report = run_bench(config, runs=args.runs, scenarios=selected, progress=progress)
    print(file=sys.stderr)
    full_report = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(full_report, encoding="utf-8")
    if args.json:
        print(full_report)
    else:
        print(report.format_table())
        if args.out:
            print(f"Full report saved to {args.out}")
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(f"teiforge {__version__}")
    return 0


# -- entry point --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teiforge",
        description="TeiForge - a lightweight toolkit for building AI agents with tools and memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("-V", "--version", action="version", version=f"teiforge {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = commands.add_parser("run", help="Run one task and print the answer.")
    run.add_argument("task", help="The task for the agent.")
    run.add_argument("--json", action="store_true", help="Print the full result with events as JSON.")
    run.add_argument("-v", "--verbose", action="store_true", help="Show tool calls and a summary.")
    _add_agent_options(run)
    run.set_defaults(handler=cmd_run)

    search = commands.add_parser("search", help="Search the web, optionally saving to a file.")
    search.add_argument("query")
    search.add_argument("--out", default=None, help="Save the summary and raw results to this file.")
    search.add_argument("--json", action="store_true")
    search.add_argument("--no-llm", action="store_true", help="Skip the LLM summary.")
    search.add_argument("--provider", choices=["openai", "anthropic", "none"], default=None)
    search.add_argument("--model", default=None)
    search.add_argument("--base-url", default=None)
    search.set_defaults(handler=cmd_search, workspace=None, max_steps=None)

    chat = commands.add_parser("chat", help="Start an interactive chat (REPL).")
    chat.add_argument("--memory", default=None, help="Load and save the conversation in this JSON file.")
    _add_agent_options(chat)
    chat.set_defaults(handler=cmd_chat)

    tools_parser = commands.add_parser("tools", help="List the built-in tools.")
    tools_parser.add_argument("--json", action="store_true", help="Print the tool schemas as JSON.")
    tools_parser.set_defaults(handler=cmd_tools)

    bench = commands.add_parser(
        "bench", help="Measure how reliably the configured model drives an agent (uses the LLM)."
    )
    bench.add_argument("--runs", type=int, default=5, help="Runs per scenario (default: 5).")
    bench.add_argument(
        "--scenario", action="append", choices=SCENARIO_NAMES, help="Only run this scenario (repeatable)."
    )
    bench.add_argument("--temperature", type=float, default=0.7, help="Model temperature (default: 0.7).")
    bench.add_argument("--json", action="store_true", help="Print the full report, with answers, as JSON.")
    bench.add_argument("--out", default=None, help="Also save the full JSON report to this file.")
    bench.add_argument("--list", action="store_true", help="List the scenarios and failure reasons.")
    bench.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    bench.add_argument("--model", default=None)
    bench.add_argument("--base-url", default=None)
    bench.set_defaults(handler=cmd_bench)

    version = commands.add_parser("version", help="Print the version.")
    version.set_defaults(handler=cmd_version)
    return parser


def _translate_legacy(argv: List[str]) -> List[str]:
    """Map the 0.2 flags (``--agent X --task T [--out F]``) onto the new subcommands."""
    legacy = argparse.ArgumentParser(add_help=False)
    legacy.add_argument("--agent", default="research")
    legacy.add_argument("--task", required=True)
    legacy.add_argument("--verbose", action="store_true")
    legacy.add_argument("--out", default=None)
    legacy.add_argument("--no-llm", action="store_true")
    parsed = legacy.parse_args(argv)
    print(
        "Note: '--task' is deprecated; use 'teiforge run TASK' or 'teiforge search QUERY'.",
        file=sys.stderr,
    )
    if parsed.out:
        new = ["search", parsed.task, "--out", parsed.out]
    else:
        new = ["run", parsed.task, "--agent", LEGACY_PRESETS.get(parsed.agent, "research")]
        if parsed.verbose:
            new.append("--verbose")
    if parsed.no_llm:
        new.append("--no-llm")
    return new


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--task" in arguments:
        arguments = _translate_legacy(arguments)

    parser = build_parser()
    args = parser.parse_args(arguments)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    try:
        return int(args.handler(args))
    except (ValueError, LLMError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
