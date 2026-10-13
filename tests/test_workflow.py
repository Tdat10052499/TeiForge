"""Workflow tests for AgentForge."""

from unittest.mock import Mock

import pytest

from agentforge import Agent
from agentforge.cli import main
from agentforge.tools import Tool


def test_run_search_and_save_creates_file(tmp_path):
    agent = Agent(name="test-agent")

    # Add a mock web_search tool returning predictable raw text
    def mock_search(query: str) -> str:
        return "MOCK RAW SEARCH RESULT: This is a test search output. Source: https://example.com"

    mock_tool = Tool(name="web_search", description="Mock web search", func=mock_search)
    agent.add_tool(mock_tool)

    # Disable LLM for deterministic output in tests
    agent.enable_llm = False

    out_file = tmp_path / "search_output.txt"

    result = agent.run_search_and_save("search for testing", str(out_file))
    assert "Saved search results to" in result
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "MOCK RAW SEARCH RESULT" in content
    assert agent.task_count == 1
    entries = agent.memory.get()
    assert [entry.role for entry in entries] == ["user", "assistant"]


def test_run_search_and_save_reports_missing_tool_and_records_task(tmp_path):
    agent = Agent(name="test-agent")

    result = agent.run_search_and_save("search for testing", str(tmp_path / "out.txt"))

    assert result == "Error: No web_search tool is available for this task."
    assert not (tmp_path / "out.txt").exists()
    assert agent.task_count == 1
    assert [entry.role for entry in agent.memory.get()] == ["user", "assistant"]


@pytest.mark.parametrize(
    "search_result",
    [
        "Tool error: RuntimeError: search unavailable",
        "Web search failed: ConnectionError: offline",
        "",
    ],
)
def test_run_search_and_save_does_not_save_failed_or_empty_results(
    tmp_path, search_result
):
    agent = Agent(name="test-agent")
    agent.enable_llm = False
    agent.add_tool(Tool(name="web_search", description="Mock search", func=lambda _: search_result))
    out_file = tmp_path / "out.txt"

    result = agent.run_search_and_save("search for testing", str(out_file))

    assert result.startswith("Error:")
    assert not out_file.exists()
    assert agent.task_count == 1
    assert [entry.role for entry in agent.memory.get()] == ["user", "assistant"]


@pytest.mark.parametrize(
    ("query", "out_path", "expected"),
    [
        ("  ", "out.txt", "Error: A search query is required."),
        ("search", "  ", "Error: An output file path is required."),
    ],
)
def test_run_search_and_save_rejects_missing_inputs(query, out_path, expected):
    agent = Agent(name="test-agent")

    result = agent.run_search_and_save(query, out_path)

    assert result == expected
    assert agent.task_count == 1
    assert [entry.role for entry in agent.memory.get()] == ["user", "assistant"]


def test_run_search_and_save_reports_file_write_error(tmp_path):
    agent = Agent(name="test-agent")
    agent.enable_llm = False
    agent.add_tool(
        Tool(
            name="web_search",
            description="Mock search",
            func=lambda _: "Search results: a valid result",
        )
    )

    result = agent.run_search_and_save("search", str(tmp_path / "missing" / "out.txt"))

    assert result.startswith("Error: Failed to save search results")
    assert agent.task_count == 1
    assert [entry.role for entry in agent.memory.get()] == ["user", "assistant"]


def test_run_search_and_save_falls_back_to_raw_result_without_summary(tmp_path):
    agent = Agent(name="test-agent")
    agent._summarize_text = Mock(return_value=None)
    agent.add_tool(
        Tool(
            name="web_search",
            description="Mock search",
            func=lambda _: "Search results: a valid result",
        )
    )
    out_file = tmp_path / "out.txt"

    result = agent.run_search_and_save("search", str(out_file))

    assert result.startswith("Saved search results to")
    assert "Original search result:" in out_file.read_text(encoding="utf-8")


def test_cli_returns_nonzero_when_search_results_cannot_be_saved(monkeypatch, tmp_path):
    agent = Agent(name="test-agent")
    monkeypatch.setattr("agentforge.cli.create_research_agent", lambda: agent)
    out_path = tmp_path / "out.txt"

    result = main(["--task", "search for testing", "--out", str(out_path)])

    assert result == 1


def test_cli_returns_nonzero_for_failed_search_without_output_file(monkeypatch):
    agent = Agent(name="test-agent")
    agent.add_tool(
        Tool(name="web_search", description="Broken search", func=lambda _: "Web search failed: offline")
    )
    monkeypatch.setattr("agentforge.cli.create_research_agent", lambda: agent)

    result = main(["--task", "search for testing"])

    assert result == 1
