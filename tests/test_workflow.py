"""Workflow tests for AgentForge."""

import os
from agentforge import Agent
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

    # Cleanup - tmp_path will be cleaned by pytest
