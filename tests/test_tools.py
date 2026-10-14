"""Tests for tools, the @tool decorator and the built-in tools."""

from typing import List, Literal, Optional
from unittest.mock import Mock

import pytest
import requests

from teiforge.tools import (
    CommandRunnerTool,
    FileReaderTool,
    FileWriterTool,
    Tool,
    ToolError,
    ToolResult,
    WebSearchTool,
    builtin_tools,
    make_file_reader,
    make_file_writer,
    normalize_search_query,
    parse_write_command,
    tool,
)
from teiforge.workspace import Workspace

# -- Tool and ToolResult -------------------------------------------------------------


def test_legacy_tool_runs_with_a_string():
    echo = Tool(name="echo", description="Echo tool", func=lambda x: f"Echo: {x}")
    result = echo.run("hello")
    assert result == ToolResult(ok=True, output="Echo: hello")
    assert str(result) == "Echo: hello"
    assert echo.structured is False


def test_legacy_tool_error_prefix_is_a_failure():
    broken = Tool(name="broken", description="", func=lambda _: "Web search failed: offline")
    result = broken.run("x")
    assert not result.ok
    assert result.error == "Web search failed: offline"
    assert str(result) == "Error: Web search failed: offline"


def test_tool_exceptions_become_failed_results():
    def bad_func(_):
        raise ValueError("explode")

    result = Tool(name="bad", description="bad", func=bad_func).run("x")
    assert not result.ok
    assert "Tool error: ValueError: explode" == result.error


def test_tool_rejects_invalid_names():
    with pytest.raises(ValueError):
        Tool(name="has space", description="", func=lambda x: x)


def test_legacy_tool_accepts_mapping_input():
    echo = Tool(name="echo", description="", func=lambda x: x)
    assert echo.run({"input": "hi"}).output == "hi"
    assert echo.run({"input": {"a": 1}}).output == '{"a": 1}'


# -- @tool decorator ----------------------------------------------------------------------


@tool
def add(a: int, b: int = 1) -> str:
    """Add two numbers.

    Longer description that is not part of the summary.

    Args:
        a: The first number.
        b: The second number,
            which defaults to one.

    Returns:
        The sum as text.
    """
    return str(a + b)


def test_tool_decorator_builds_schema_from_hints_and_docstring():
    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.parameters == {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "The first number."},
            "b": {
                "type": "integer",
                "description": "The second number, which defaults to one.",
                "default": 1,
            },
        },
        "required": ["a"],
    }


def test_structured_tool_runs_with_dict_kwargs_and_single_string():
    assert add.run({"a": 2, "b": 3}).output == "5"
    assert add.run(a=2).output == "3"
    assert add.run("4").ok is False  # "4" is a string, int("4") is not attempted
    shout = tool(lambda text: text.upper(), name="shout")
    assert shout.run("hi").output == "HI"


def test_structured_tool_validates_arguments():
    missing = add.run({})
    assert not missing.ok and "missing argument(s): a" in missing.error
    unknown = add.run({"a": 1, "c": 2})
    assert not unknown.ok and "unknown argument(s): c" in unknown.error
    ambiguous = tool(lambda x, y: x + y, name="pair").run("text")
    assert not ambiguous.ok and "needs structured arguments" in ambiguous.error


def test_tool_decorator_supports_optional_list_literal_and_overrides():
    @tool(name="pick", description="Pick things")
    def pick(items: List[str], mode: Literal["first", "last"] = "first", note: Optional[str] = None) -> str:
        return items[0] if mode == "first" else items[-1]

    props = pick.parameters["properties"]
    assert props["items"] == {"type": "array", "items": {"type": "string"}}
    assert props["mode"] == {"enum": ["first", "last"], "default": "first"}
    assert props["note"] == {"type": "string"}
    assert pick.description == "Pick things"
    assert pick.run({"items": ["a", "b"], "mode": "last"}).output == "b"


def test_tool_decorator_rejects_varargs_and_unknown_types():
    with pytest.raises(TypeError):
        tool(lambda *args: "x", name="varargs")

    class Custom:
        pass

    def uses_custom(value: Custom) -> str:
        return "x"

    with pytest.raises(TypeError):
        tool(uses_custom)


def test_tool_error_is_reported_without_prefix():
    @tool
    def fails() -> str:
        """Always fails."""
        raise ToolError("nope")

    assert fails.run().error == "nope"


def test_tool_returning_tool_result_is_passed_through():
    custom = Tool("custom", "", lambda _: ToolResult(ok=False, error="custom failure"))
    assert custom.run("x").error == "custom failure"


def test_schemas_for_providers():
    openai_schema = add.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "add"
    anthropic_schema = add.to_anthropic_schema()
    assert anthropic_schema["input_schema"] == add.parameters
    assert repr(add) == "Tool(name='add')"


# -- web search ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("search for Python programming", "Python programming"),
        ("Please look up AI news", "AI news"),
        ("search: agents", "agents"),
        ("search for", ""),
        ("research papers", "papers"),
        ("findings about bees", "findings about bees"),
    ],
)
def test_normalize_search_query(raw, expected):
    assert normalize_search_query(raw) == expected


def test_web_search_uses_instant_answer(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "AbstractText": "Python is a general-purpose programming language.",
        "AbstractURL": "https://example.com/python",
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("teiforge.tools.requests.get", request)

    result = WebSearchTool.run({"query": "search for Python programming"})

    assert result.ok
    assert "Python is a general-purpose programming language." in result.output
    assert "Source: https://example.com/python" in result.output
    request.assert_called_once_with(
        "https://api.duckduckgo.com/",
        params={"q": "Python programming", "format": "json", "no_html": "1", "skip_disambig": "1"},
        timeout=10,
    )


def test_web_search_uses_related_topics_with_max_results(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "RelatedTopics": [
            {"Text": "Topic A", "FirstURL": "https://a"},
            {"Topics": [{"Text": "Topic B", "FirstURL": "https://b"}]},
            {"Text": "Group Category", "FirstURL": "https://c"},
            {"Text": "Topic D", "FirstURL": "https://d"},
        ]
    }
    monkeypatch.setattr("teiforge.tools.requests.get", Mock(return_value=response))

    result = WebSearchTool.run({"query": "topics", "max_results": 2})

    assert "Topic A" in result.output and "Topic B" in result.output
    assert "Category" not in result.output and "Topic D" not in result.output


def test_web_search_falls_back_to_html_results(monkeypatch):
    api_response = Mock()
    api_response.json.return_value = {}
    html_response = Mock()
    html_response.text = (
        '<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpy">'
        "Python <b>&amp;</b> programming</a>"
    )
    request = Mock(side_effect=[api_response, html_response])
    monkeypatch.setattr("teiforge.tools.requests.get", request)

    result = WebSearchTool.run("AI agents")

    assert "Python & programming" in result.output
    assert "https://example.com/py" in result.output
    assert request.call_args_list[1].kwargs["params"] == {"q": "AI agents"}


def test_web_search_reports_no_results(monkeypatch):
    api_response = Mock()
    api_response.json.return_value = {}
    html_response = Mock(text="<html><body>No results</body></html>")
    monkeypatch.setattr("teiforge.tools.requests.get", Mock(side_effect=[api_response, html_response]))

    result = WebSearchTool.run("unknown query")

    assert result.ok and "No results found" in result.output


def test_web_search_reports_network_failures(monkeypatch):
    monkeypatch.setattr("teiforge.tools.requests.get", Mock(side_effect=requests.ConnectionError("offline")))
    result = WebSearchTool.run("anything")
    assert not result.ok
    assert result.error.startswith("Web search failed: Instant Answer API ConnectionError")


def test_web_search_reports_api_error_when_html_has_nothing(monkeypatch):
    html_response = Mock(text="<html></html>")
    monkeypatch.setattr(
        "teiforge.tools.requests.get",
        Mock(side_effect=[requests.Timeout("slow"), html_response]),
    )
    result = WebSearchTool.run("anything")
    assert not result.ok and "Timeout" in result.error


def test_web_search_reports_html_only_failure(monkeypatch):
    api_response = Mock()
    api_response.json.return_value = {}
    monkeypatch.setattr(
        "teiforge.tools.requests.get",
        Mock(side_effect=[api_response, requests.ConnectionError("down")]),
    )
    result = WebSearchTool.run("anything")
    assert result.error == "Web search failed: ConnectionError: down"


def test_web_search_requires_a_query():
    result = WebSearchTool.run("search for")
    assert not result.ok and "non-empty query" in result.error


# -- file tools -----------------------------------------------------------------------------


def test_file_writer_and_reader_round_trip(tmp_path):
    space = Workspace(tmp_path)
    writer, reader = make_file_writer(space), make_file_reader(space)

    written = writer.run({"path": "notes/a.txt", "content": "  keep: spaces  "})
    assert written.ok, written.error
    assert (tmp_path / "notes" / "a.txt").read_text(encoding="utf-8") == "  keep: spaces  "

    read = reader.run({"path": "notes/a.txt"})
    assert read.ok
    assert read.output.endswith("  keep: spaces  ")


def test_file_writer_does_not_overwrite_without_permission(tmp_path):
    (tmp_path / "a.txt").write_text("original", encoding="utf-8")
    writer = make_file_writer(Workspace(tmp_path))

    refused = writer.run({"path": "a.txt", "content": "new"})
    assert not refused.ok and "already exists" in refused.error
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "original"

    replaced = writer.run({"path": "a.txt", "content": "new", "overwrite": True})
    assert replaced.ok
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "sub/../../x.txt"])
def test_file_tools_refuse_paths_outside_the_workspace(tmp_path, path):
    space = Workspace(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    assert "outside the workspace" in make_file_reader(space).run({"path": path}).error
    assert "outside the workspace" in make_file_writer(space).run({"path": path, "content": "x"}).error


def test_file_reader_errors(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00")
    reader = make_file_reader(Workspace(tmp_path))
    assert reader.run({"path": "missing.txt"}).error == "File not found: missing.txt"
    assert "is a directory" in reader.run({"path": "dir"}).error
    assert "not a UTF-8 text file" in reader.run({"path": "bin.dat"}).error
    assert reader.run({"path": "  "}).error == "No file path was provided."


def test_file_reader_truncates_long_files(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 50, encoding="utf-8")
    result = make_file_reader(Workspace(tmp_path), max_chars=10).run({"path": "long.txt"})
    assert "x" * 10 + "\n\n[Truncated" in result.output


def test_file_writer_refuses_directories(tmp_path):
    (tmp_path / "dir").mkdir()
    result = make_file_writer(Workspace(tmp_path)).run({"path": "dir", "content": "x"})
    assert "is a directory" in result.error


def test_default_file_tools_use_the_current_directory(isolated_environment):
    assert FileWriterTool.run({"path": "demo.txt", "content": "hello"}).ok
    assert (isolated_environment / "demo.txt").exists()
    assert "hello" in FileReaderTool.run({"path": "demo.txt"}).output


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("notes.txt:hello", ("notes.txt", "hello")),
        ("notes.txt:a:b", ("notes.txt", "a:b")),
        ("C:\\tmp\\a.txt:hello", ("C:\\tmp\\a.txt", "hello")),
        ("D:/data/b.txt:x:y", ("D:/data/b.txt", "x:y")),
        (" spaced.txt :  keep ", ("spaced.txt", "  keep ")),
    ],
)
def test_parse_write_command_handles_windows_drives(payload, expected):
    assert parse_write_command(payload) == expected


@pytest.mark.parametrize("payload", ["no-colon", ":content"])
def test_parse_write_command_rejects_bad_input(payload):
    with pytest.raises(ToolError, match="Usage"):
        parse_write_command(payload)


# -- command runner -------------------------------------------------------------------------


def test_command_runner_is_disabled_by_default():
    result = CommandRunnerTool.run("echo hello")
    assert not result.ok
    assert "disabled" in result.error and "TEIFORGE_ENABLE_COMMANDS" in result.error


@pytest.fixture
def commands_enabled(monkeypatch):
    monkeypatch.setenv("TEIFORGE_ENABLE_COMMANDS", "1")


@pytest.mark.usefixtures("commands_enabled")
class TestCommandRunner:
    def test_echo_treats_shell_operators_as_text(self):
        assert CommandRunnerTool.run("run echo safe & whoami").output == "Command output:\nsafe & whoami"

    def test_echo_without_text(self):
        assert CommandRunnerTool.run("echo").output == "Command output:\nCommand returned no output."

    def test_rejects_arbitrary_python(self):
        assert "restricted to --version or -V" in CommandRunnerTool.run('python -c "print(1)"').error

    def test_python_version(self):
        assert CommandRunnerTool.run("python --version").output.startswith("Command output:\n3.")

    def test_pwd_reports_workspace(self, tmp_path):
        runner = builtin_tools(Workspace(tmp_path))["command_runner"]
        assert runner.run("pwd").output == f"Command output:\n{tmp_path.resolve()}"
        assert runner.run("pwd extra").error == "Usage: pwd"

    def test_ls_lists_inside_workspace_only(self, tmp_path):
        (tmp_path / "example.txt").write_text("example", encoding="utf-8")
        runner = builtin_tools(Workspace(tmp_path))["command_runner"]
        assert runner.run('ls "."').output == "Command output:\nexample.txt"
        assert "outside the workspace" in runner.run("ls /").error

    def test_ls_empty_directory_and_errors(self, tmp_path):
        (tmp_path / "empty").mkdir()
        runner = builtin_tools(Workspace(tmp_path))["command_runner"]
        assert runner.run("ls empty").output == "Command output:\nCommand returned no output."
        assert "Command execution failed" in runner.run("ls missing").error
        assert runner.run("ls . & whoami").error == "Usage: ls [path]"
        assert "Invalid ls arguments" in runner.run('ls "unclosed').error

    def test_rejects_unknown_and_empty_commands(self):
        assert "is not allowed" in CommandRunnerTool.run("rm -rf /").error
        assert CommandRunnerTool.run("   ").error == "No command was provided."


def test_builtin_tools_without_workspace_returns_shared_instances():
    tools = builtin_tools()
    assert tools["web_search"] is WebSearchTool
    assert tools["file_reader"] is FileReaderTool


def test_file_reader_strips_utf8_bom(tmp_path):
    # Windows PowerShell 5 and older editors write UTF-8 files with a byte-order mark.
    (tmp_path / "bom.txt").write_bytes("﻿Meeting notes".encode())
    output = make_file_reader(Workspace(tmp_path)).run({"path": "bom.txt"}).output
    assert "﻿" not in output
    assert output.endswith("\n\nMeeting notes")
