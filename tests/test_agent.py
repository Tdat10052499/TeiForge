"""Tests for the agent module."""

from unittest.mock import Mock

import pytest

from agentforge import Agent, AgentConfig, Tool, WebSearchTool


def test_agent_creation():
    agent = Agent(name="test-agent")
    assert agent.name == "test-agent"
    assert agent.model == "gpt-4o-mini"
    assert len(agent.tools) == 0
    assert agent.task_count == 0


def test_agent_add_tool():
    agent = Agent(name="test-agent")
    tool = Tool(name="test_tool", description="A test tool", func=lambda x: f"Result: {x}")
    agent.add_tool(tool)
    assert "test_tool" in agent.tools


def test_agent_run():
    agent = Agent(name="test-agent")
    tool = Tool(name="hello_tool", description="Greeting tool", func=lambda x: f"Hello, {x}!")
    agent.add_tool(tool)
    result = agent.run("world")
    assert "task" in result.lower() or "hello" in result.lower()
    assert agent.task_count == 1
    assert len(agent.memory) == 2


def test_agent_returns_tool_failures_as_explicit_errors():
    agent = Agent(name="test-agent")
    agent.add_tool(Tool(name="web_search", description="Broken search", func=lambda _: "Web search failed: offline"))

    result = agent.run("search for testing")

    assert result.startswith("Error:")
    assert "Web search failed: offline" in result
    assert agent.task_count == 1


def test_agent_remove_tool():
    agent = Agent(name="test-agent")
    tool = Tool(name="test_tool", description="A test tool", func=lambda x: x)
    agent.add_tool(tool)
    assert agent.remove_tool("test_tool") is True
    assert agent.remove_tool("missing") is False


def test_agent_tools_description():
    agent = Agent(name="test-agent")
    description = agent.get_tools_description()
    assert "No tools" in description

    tool = Tool(name="demo_tool", description="Demo tool", func=lambda x: x)
    agent.add_tool(tool)
    description = agent.get_tools_description()
    assert "demo_tool" in description
    assert "Demo tool" in description


def test_memory_summary():
    agent = Agent(name="test-agent")
    _ = agent.run("First task")
    summary = agent.get_memory_summary()
    assert "Tasks run: 1" in summary
    assert "Memory entries" in summary


def test_agent_string_repr():
    agent = Agent(name="test-agent")
    assert "test-agent" in str(agent)
    assert "gpt-4o-mini" in str(agent)


def test_agent_uses_config_for_name_model_memory_and_tools():
    config = AgentConfig(
        name="configured-agent",
        model="configured-model",
        memory_size=3,
        tools=["web_search"],
    )

    agent = Agent(config=config)

    assert agent.name == "configured-agent"
    assert agent.model == "configured-model"
    assert agent.memory.max_entries == 3
    assert agent.tools["web_search"] is WebSearchTool


def test_agent_config_values_are_used_for_llm_call(monkeypatch):
    config = AgentConfig(
        name="configured-agent",
        model="configured-model",
        temperature=0.4,
        max_tokens=123,
        system_prompt="Use the configured summary prompt.",
    )
    agent = Agent(config=config)
    completion = Mock()
    completion.choices = [Mock()]
    completion.choices[0].message.content = "summary"
    client = Mock()
    client.chat.completions.create.return_value = completion
    monkeypatch.setattr("agentforge.agent.OpenAI", Mock(return_value=client))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = agent._summarize_text("search result")

    assert result == "summary"
    client.chat.completions.create.assert_called_once()
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "configured-model"
    assert call["temperature"] == 0.4
    assert call["max_tokens"] == 123
    assert call["messages"][0]["content"] == "Use the configured summary prompt."


@pytest.mark.parametrize(
    "config_values",
    [
        {"name": " "},
        {"name": "agent", "memory_size": 0},
        {"name": "agent", "max_tokens": 0},
        {"name": "agent", "temperature": 2.1},
    ],
)
def test_agent_config_rejects_invalid_values(config_values):
    with pytest.raises(ValueError):
        AgentConfig(**config_values)


def test_agent_config_rejects_unknown_builtin_tool():
    with pytest.raises(ValueError, match="Unknown built-in tool"):
        Agent(config=AgentConfig(name="configured-agent", tools=["unknown"]))


def test_agent_rejects_ambiguous_config_arguments():
    with pytest.raises(ValueError, match="either name/model or config"):
        Agent(name="agent", config=AgentConfig(name="configured-agent"))
