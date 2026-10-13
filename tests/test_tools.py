"""Tests for the tools module."""

from unittest.mock import Mock

from agentforge.tools import (
    CommandRunnerTool,
    FileReaderTool,
    FileWriterTool,
    Tool,
    WebSearchTool,
)


def test_tool_creation():
    tool = Tool(name="echo", description="Echo tool", func=lambda x: f"Echo: {x}")
    assert tool.name == "echo"
    assert tool.description == "Echo tool"


def test_tool_run():
    tool = Tool(name="echo", description="Echo tool", func=lambda x: f"Echo: {x}")
    assert tool.run("hello") == "Echo: hello"


def test_web_search_tool_uses_instant_answer_and_cleans_query_prefix(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "AbstractText": "Python is a general-purpose programming language.",
        "AbstractURL": "https://example.com/python",
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("agentforge.tools.requests.get", request)

    result = WebSearchTool.run("search for Python programming")

    assert "Python is a general-purpose programming language." in result
    assert "https://example.com/python" in result
    request.assert_called_once_with(
        "https://api.duckduckgo.com/",
        params={
            "q": "Python programming",
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        },
        timeout=10,
    )


def test_web_search_tool_returns_search_results(monkeypatch):
    api_response = Mock()
    api_response.json.return_value = {}
    html_response = Mock()
    html_response.text = (
        '<a class="result__a" href="https://example.com/python">Python &amp; programming</a>'
    )
    request = Mock(side_effect=[api_response, html_response])
    monkeypatch.setattr("agentforge.tools.requests.get", request)

    response = WebSearchTool.run("AI agents")

    assert "Python & programming" in response
    assert "https://example.com/python" in response
    assert request.call_args_list[1].args == (
        "https://html.duckduckgo.com/html/",
    )
    assert request.call_args_list[1].kwargs == {
        "params": {"q": "AI agents"},
        "headers": {"User-Agent": "Mozilla/5.0"},
        "timeout": 10,
    }


def test_web_search_tool_reports_no_results(monkeypatch):
    api_response = Mock()
    api_response.json.return_value = {}
    html_response = Mock()
    html_response.text = "<html><body>No results</body></html>"
    monkeypatch.setattr(
        "agentforge.tools.requests.get",
        Mock(side_effect=[api_response, html_response]),
    )

    result = WebSearchTool.run("unknown query")

    assert "No results found" in result


def test_file_reader_tool_usage_message():
    response = FileReaderTool.run("hello")
    assert "Usage" in response


def test_file_writer_tool_usage_message():
    response = FileWriterTool.run("hello")
    assert "Usage" in response


def test_tool_error_handling():
    def bad_func(_):
        raise ValueError("explode")

    tool = Tool(name="bad", description="bad", func=bad_func)
    result = tool.run("x")
    assert "Tool error" in result
    assert "ValueError" in result


def test_command_runner_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENTFORGE_ENABLE_COMMANDS", raising=False)

    result = CommandRunnerTool.run("echo hello")

    assert "disabled" in result
    assert "hello" not in result


def test_command_runner_treats_shell_operators_as_plain_echo_text(monkeypatch):
    monkeypatch.setenv("AGENTFORGE_ENABLE_COMMANDS", "1")

    result = CommandRunnerTool.run("run echo safe & whoami")

    assert result == "Command output:\nsafe & whoami"


def test_command_runner_rejects_arbitrary_python_execution(monkeypatch):
    monkeypatch.setenv("AGENTFORGE_ENABLE_COMMANDS", "1")

    result = CommandRunnerTool.run('python -c "print(1)"')

    assert "restricted to --version or -V" in result


def test_command_runner_lists_a_single_directory_without_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTFORGE_ENABLE_COMMANDS", "1")
    (tmp_path / "example.txt").write_text("example", encoding="utf-8")

    result = CommandRunnerTool.run(f'ls "{tmp_path}"')

    assert result == "Command output:\nexample.txt"


def test_command_runner_rejects_extra_ls_arguments(monkeypatch):
    monkeypatch.setenv("AGENTFORGE_ENABLE_COMMANDS", "1")

    result = CommandRunnerTool.run("ls . & whoami")

    assert result == "Usage: ls [path]"
