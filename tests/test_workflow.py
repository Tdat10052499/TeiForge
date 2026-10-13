"""Tests for Agent.run_search_and_save."""

from unittest.mock import Mock

import pytest

from teiforge import Agent
from teiforge.testing import ScriptedProvider
from teiforge.tools import Tool


def search_agent(result):
    agent = Agent(name="test-agent")
    agent.enable_llm = False
    agent.add_tool(Tool(name="web_search", description="Mock search", func=lambda _: result))
    return agent


def assert_task_recorded(agent):
    assert agent.task_count == 1
    assert [entry.role for entry in agent.memory.get()] == ["user", "assistant"]


def test_run_search_and_save_creates_file(tmp_path):
    agent = search_agent("MOCK RAW SEARCH RESULT. Source: https://example.com")
    out_file = tmp_path / "search_output.txt"

    result = agent.run_search_and_save("search for testing", str(out_file))

    assert result == f"Saved search results to {out_file}"
    assert out_file.read_text(encoding="utf-8") == (
        "Original search result:\nMOCK RAW SEARCH RESULT. Source: https://example.com\n"
    )
    assert_task_recorded(agent)


def test_run_search_and_save_includes_llm_summary(tmp_path):
    agent = search_agent("raw results")
    agent.enable_llm = True
    agent.provider = ScriptedProvider(["Short summary."])
    out_file = tmp_path / "out.txt"

    agent.run_search_and_save("bees", str(out_file))

    assert out_file.read_text(encoding="utf-8").startswith(
        "Summary:\nShort summary.\n\nOriginal search result:"
    )


def test_run_search_and_save_reports_missing_tool(tmp_path):
    agent = Agent(name="test-agent")
    result = agent.run_search_and_save("search for testing", str(tmp_path / "out.txt"))
    assert result == "Error: No web_search tool is available for this task."
    assert not (tmp_path / "out.txt").exists()
    assert_task_recorded(agent)


@pytest.mark.parametrize(
    ("search_result", "message"),
    [
        ("Tool error: RuntimeError: search unavailable", "Error: Search failed: Tool error"),
        ("Web search failed: ConnectionError: offline", "Error: Search failed: Web search failed"),
        ("   ", "Error: Search returned an empty result."),
    ],
)
def test_run_search_and_save_does_not_save_failures(tmp_path, search_result, message):
    agent = search_agent(search_result)
    out_file = tmp_path / "out.txt"
    result = agent.run_search_and_save("search for testing", str(out_file))
    assert result.startswith(message)
    assert not out_file.exists()
    assert_task_recorded(agent)


@pytest.mark.parametrize(
    ("query", "out_path", "expected"),
    [
        ("  ", "out.txt", "Error: A search query is required."),
        ("search", "  ", "Error: An output file path is required."),
    ],
)
def test_run_search_and_save_rejects_missing_inputs(query, out_path, expected):
    agent = Agent(name="test-agent")
    assert agent.run_search_and_save(query, out_path) == expected
    assert_task_recorded(agent)


def test_run_search_and_save_reports_file_write_error(tmp_path):
    agent = search_agent("Search results: a valid result")
    result = agent.run_search_and_save("search", str(tmp_path / "missing" / "out.txt"))
    assert result.startswith("Error: Failed to save search results")
    assert_task_recorded(agent)


def test_run_search_and_save_falls_back_to_raw_result_without_summary(tmp_path):
    agent = search_agent("Search results: a valid result")
    agent.enable_llm = True
    agent._summarize_text = Mock(return_value=None)
    out_file = tmp_path / "out.txt"
    assert agent.run_search_and_save("search", str(out_file)).startswith("Saved search results to")
    assert "Original search result:" in out_file.read_text(encoding="utf-8")
