"""Tests for the command-line interface."""

import io
import json
import runpy
from argparse import Namespace

import pytest

from teiforge import __version__
from teiforge.cli import build_agent, cmd_chat, event_printer, main
from teiforge.events import Event
from teiforge.testing import ScriptedProvider
from teiforge.tools import Tool


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the network search with a deterministic function."""

    def fake(query: str, max_results: int = 5) -> str:
        return f"Search results for: '{query}'\n\n- Result\n  Link: https://example.com"

    monkeypatch.setattr("teiforge.tools.WebSearchTool.func", fake)
    return fake


def test_version_command_and_flag(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"teiforge {__version__}"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage: teiforge" in capsys.readouterr().out


def test_tools_command(capsys):
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "web_search(query, max_results)" in out and "TEIFORGE_ENABLE_COMMANDS" in out
    assert main(["tools", "--json"]) == 0
    schemas = json.loads(capsys.readouterr().out)
    assert {schema["function"]["name"] for schema in schemas} >= {"web_search", "file_reader"}


def test_run_command_prints_answer(fake_search, capsys):
    assert main(["run", "search for bees", "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[Using web_search]") and "Link: https://example.com" in out


def test_run_command_json_and_exit_code(capsys, isolated_environment):
    assert main(["run", "read:missing.txt", "--agent", "files", "--json", "--no-llm"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False and data["output"] == "Error: File not found: missing.txt"
    assert [event["type"] for event in data["events"]][-1] == "final"


def test_run_command_verbose_prints_tool_activity(fake_search, capsys):
    assert main(["run", "search for bees", "--no-llm", "--verbose"]) == 0
    captured = capsys.readouterr()
    assert "keyword router (no LLM)" in captured.out
    assert "Tasks run: 1" in captured.out
    assert "-> web_search" in captured.err and "<- web_search ok" in captured.err


def test_run_with_workspace_and_tools_option(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("inside", encoding="utf-8")
    assert (
        main(["run", "read:a.txt", "--tools", "file_reader", "--workspace", str(tmp_path), "--no-llm"]) == 0
    )
    assert "inside" in capsys.readouterr().out


def test_unknown_tool_option_is_a_usage_error(capsys):
    assert main(["run", "x", "--tools", "web_search,laser"]) == 2
    assert "Unknown tool(s): laser" in capsys.readouterr().err


def test_run_with_config_file(tmp_path, capsys, fake_search):
    config = tmp_path / "agent.toml"
    config.write_text('name = "from-file"\ntools = ["web_search"]\n', encoding="utf-8")
    assert main(["run", "search for x", "--config", str(config), "--no-llm", "--max-steps", "3"]) == 0
    assert "[Using web_search]" in capsys.readouterr().out


def test_build_agent_applies_overrides(tmp_path):
    config = tmp_path / "agent.json"
    config.write_text('{"name": "cfg", "tools": ["web_search"]}', encoding="utf-8")
    args = Namespace(
        config=str(config),
        agent="files",
        tools=None,
        provider="none",
        model="local-model",
        base_url="http://localhost:1234/v1",
        max_steps=2,
        no_llm=False,
        workspace=str(tmp_path),
    )
    agent = build_agent(args)
    assert agent.name == "cfg"
    assert sorted(agent.tools) == ["file_reader", "file_writer"]
    assert agent.model == "local-model" and agent.config.max_steps == 2
    assert agent.config.base_url == "http://localhost:1234/v1"
    assert agent.workspace.root == tmp_path.resolve()


def test_misconfigured_provider_is_reported(capsys):
    assert main(["run", "hello", "--provider", "openai", "--verbose"]) == 1
    captured = capsys.readouterr()
    assert "LLM unavailable" in captured.out
    assert "Error:" in captured.out


def test_search_command_saves_file(fake_search, tmp_path, capsys):
    out = tmp_path / "results.txt"
    assert main(["search", "bees", "--out", str(out), "--no-llm"]) == 0
    assert "Saved search results" in capsys.readouterr().out
    assert "Search results for: 'bees'" in out.read_text(encoding="utf-8")


def test_search_command_without_file(fake_search, capsys):
    assert main(["search", "bees", "--json", "--no-llm"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_search_command_returns_nonzero_when_saving_fails(fake_search, tmp_path):
    assert main(["search", "bees", "--out", str(tmp_path / "missing" / "x.txt"), "--no-llm"]) == 1


def test_legacy_flags_are_translated(fake_search, tmp_path, capsys):
    assert main(["--agent", "research", "--task", "search for bees", "--no-llm", "--verbose"]) == 0
    captured = capsys.readouterr()
    assert "deprecated" in captured.err and "[Using web_search]" in captured.out
    out = tmp_path / "legacy.txt"
    assert main(["--task", "bees", "--out", str(out), "--no-llm"]) == 0
    assert out.exists()
    assert main(["--agent", "file-analyzer", "--task", "read:missing.txt"]) == 1


def chat_args(**overrides):
    values = dict(
        config=None,
        agent=None,
        tools=None,
        provider="none",
        model=None,
        base_url=None,
        max_steps=None,
        no_llm=True,
        workspace=None,
        memory=None,
    )
    values.update(overrides)
    return Namespace(**values)


def run_chat(lines, args):
    inputs = iter(lines)

    def fake_input(_prompt):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    stream = io.StringIO()
    code = cmd_chat(args, input_func=fake_input, stream=stream)
    return code, stream.getvalue()


def test_chat_repl_commands_and_memory_file(tmp_path, isolated_environment):
    (isolated_environment / "notes.txt").write_text("chat notes", encoding="utf-8")
    memory_file = tmp_path / "chat.json"
    code, out = run_chat(
        ["", "/help", "/tools", "read:notes.txt", "/memory", "/bogus", "/save", "/exit"],
        chat_args(memory=str(memory_file)),
    )
    assert code == 0
    assert "Type /help" in out and "/save [FILE]" in out
    assert "file_reader" in out
    assert "-> file_reader" in out and "chat notes" in out
    assert "Tasks run: 1" in out
    assert "Unknown command /bogus" in out
    assert f"Saved 2 entries to {memory_file}" in out
    assert json.loads(memory_file.read_text(encoding="utf-8"))["entries"][0]["content"] == "read:notes.txt"

    code, out = run_chat(["/memory", "/clear", "/save"], chat_args(memory=str(memory_file)))
    assert "Memory entries: 2" in out and "Conversation cleared." in out


def test_chat_save_without_target_and_eof(tmp_path):
    code, out = run_chat(["/save", f"/save {tmp_path / 'x.json'}"], chat_args())
    assert code == 0
    assert "Usage: /save FILE" in out
    assert (tmp_path / "x.json").exists()


def test_chat_uses_llm_provider(monkeypatch):
    provider = ScriptedProvider(["Hello there!"])
    monkeypatch.setattr("teiforge.agent.provider_from_config", lambda _config: provider)
    _, out = run_chat(["hi", "/quit"], chat_args(provider=None, no_llm=False))
    assert "LLM: scripted / scripted-model" in out
    assert "all-agent> Hello there!" in out


def test_event_printer_formats_events():
    stream = io.StringIO()
    handle = event_printer(stream)
    handle(Event("tool_call", {"name": "web_search", "arguments": {"query": "x" * 200}}))
    handle(Event("tool_result", {"name": "web_search", "ok": False, "output": "boom"}))
    handle(Event("error", {"message": "bad"}))
    handle(Event("final", {"output": "ignored"}))
    lines = stream.getvalue().splitlines()
    assert lines[0].endswith("...") and len(lines[0]) < 140
    assert lines[1] == "  <- web_search failed (4 chars)"
    assert lines[2] == "  !! bad"
    assert len(lines) == 3


def test_python_dash_m_entry_point(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["teiforge", "version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("teiforge", run_name="__main__")
    assert exit_info.value.code == 0
    assert "teiforge" in capsys.readouterr().out


def test_cli_custom_tool_failure_exit_code(monkeypatch, capsys):
    broken = Tool("web_search", "broken", lambda _: "Web search failed: offline")
    monkeypatch.setattr("teiforge.tools.DEFAULT_TOOLS", {**__import__("teiforge").tools.DEFAULT_TOOLS})
    monkeypatch.setitem(__import__("teiforge").tools.DEFAULT_TOOLS, "web_search", broken)
    assert main(["run", "search for x", "--no-llm"]) == 1
    assert "Error: Web search failed: offline" in capsys.readouterr().out
