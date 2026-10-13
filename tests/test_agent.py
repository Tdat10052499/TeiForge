"""Tests for the Agent: construction, the keyword router and the LLM loop."""

import pytest

from teiforge import Agent, AgentConfig, LLMError, LLMResponse, Tool, ToolCall, WebSearchTool, tool
from teiforge.agent import extract_path
from teiforge.testing import ScriptedProvider
from teiforge.tools import builtin_tools
from teiforge.workspace import Workspace

# -- construction ----------------------------------------------------------------------


def test_agent_creation_defaults():
    agent = Agent(name="test-agent")
    assert agent.name == "test-agent"
    assert agent.model == "gpt-4o-mini"
    assert agent.tools == {}
    assert agent.task_count == 0
    assert "test-agent" in str(agent) and "gpt-4o-mini" in repr(agent)


def test_agent_model_from_environment(monkeypatch):
    monkeypatch.setenv("TEIFORGE_MODEL", "env-model")
    assert Agent(name="a").model == "env-model"


def test_agent_uses_config_for_name_model_memory_and_tools():
    config = AgentConfig(name="configured", model="configured-model", memory_size=3, tools=["web_search"])
    agent = Agent(config=config)
    assert agent.name == "configured"
    assert agent.model == "configured-model"
    assert agent.memory.max_entries == 3
    assert agent.tools["web_search"] is WebSearchTool


def test_agent_binds_file_tools_to_workspace(tmp_path):
    agent = Agent(config=AgentConfig(name="a", tools=["file_writer"], workspace=str(tmp_path)))
    assert agent.workspace.root == tmp_path.resolve()
    agent.tools["file_writer"].run({"path": "x.txt", "content": "hi"})
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "hi"


def test_agent_accepts_workspace_object_and_extra_tools(tmp_path):
    space = Workspace(tmp_path)

    def shout(text: str) -> str:
        """Upper-case text."""
        return text.upper()

    agent = Agent(name="a", workspace=space, tools=[shout])
    assert agent.workspace is space
    assert agent.tools["shout"].run("hi").output == "HI"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "a", "config": AgentConfig(name="b")}, "either name/model or config"),
        ({"config": AgentConfig(name="b"), "model": "m"}, "through config"),
        ({}, "name or config"),
    ],
)
def test_agent_rejects_ambiguous_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Agent(**kwargs)


def test_agent_rejects_unknown_builtin_tool():
    with pytest.raises(ValueError, match="Unknown built-in tool"):
        Agent(config=AgentConfig(name="a", tools=["unknown"]))


def test_add_remove_and_describe_tools():
    agent = Agent(name="a")
    assert "No tools" in agent.get_tools_description()
    agent.add_tool(Tool(name="demo_tool", description="Demo tool", func=lambda x: x))
    assert "demo_tool: Demo tool" in agent.get_tools_description()
    assert agent.remove_tool("demo_tool") is True
    assert agent.remove_tool("missing") is False
    with pytest.raises(TypeError):
        agent.add_tool("not a tool")  # type: ignore[arg-type]


def test_memory_summary_and_task_recording():
    agent = Agent(name="a")
    agent.run("First task")
    summary = agent.get_memory_summary()
    assert "Tasks run: 1" in summary
    assert len(agent.memory) == 2


# -- keyword router ----------------------------------------------------------------------


def make_router_agent(tmp_path, **config):
    agent = Agent(
        config=AgentConfig(name="router", tools=list(builtin_tools()), workspace=str(tmp_path), **config)
    )
    agent.enable_llm = False
    return agent


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Search for open-source agent frameworks", "web_search"),
        ("Summarize the findings", "file_reader"),  # 'find' must not match 'findings'
        ("What should I return for brunch?", None),  # 'run' must not match 'return'/'brunch'
        ("Write a profile summary to notes.txt", "file_writer"),  # 'file' not in 'profile'
        ("Create a file", "file_writer"),
        ("Please read:demo.txt", "file_reader"),
        ("search: python", "web_search"),
        ("look up the weather", "web_search"),
        ("run echo hi", "command_runner"),
        ("Explain recursion", None),
    ],
)
def test_router_matches_whole_words(tmp_path, task, expected):
    assert make_router_agent(tmp_path).route(task) == expected


def test_router_reads_a_path_mentioned_in_prose(tmp_path):
    (tmp_path / "demo.txt").write_text("hello from demo", encoding="utf-8")
    agent = make_router_agent(tmp_path)

    result = agent.run_detailed("Read demo.txt if it exists")

    assert result.ok, result.output
    assert result.output.startswith("[Using file_reader]")
    assert "hello from demo" in result.output
    assert result.mode == "router"
    assert [e.type for e in result.events] == ["task_start", "tool_call", "tool_result", "final"]


def test_router_explicit_read_prefix(tmp_path):
    (tmp_path / "demo.txt").write_text("content", encoding="utf-8")
    assert "content" in make_router_agent(tmp_path).run("Please read:demo.txt")


def test_router_reports_missing_path(tmp_path):
    result = make_router_agent(tmp_path).run_detailed("Summarize the findings")
    assert not result.ok
    assert "Could not find a file path" in result.output
    assert result.events[1].type == "error"


def test_router_write_requires_explicit_syntax(tmp_path):
    agent = make_router_agent(tmp_path)
    assert "Use write:path/to/file.txt:content" in agent.run("Write a profile summary to notes.txt")
    assert agent.run("write:notes.txt:hello").startswith("[Using file_writer]")
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert "already exists" in agent.run("write:notes.txt:again")


def test_router_search_with_summary(tmp_path):
    agent = make_router_agent(tmp_path)
    agent.enable_llm = True
    agent.provider = ScriptedProvider(["A short summary."])
    agent.config.mode = "router"
    agent.tools["web_search"] = Tool("web_search", "mock", lambda q: f"results for {q}")

    output = agent.run("search for bees")

    assert output.startswith("[Using web_search]\n\nA short summary.\n\nFull results:\nresults for")


def test_router_returns_tool_failures_as_errors():
    agent = Agent(name="a")
    agent.add_tool(Tool(name="web_search", description="Broken", func=lambda _: "Web search failed: offline"))
    result = agent.run_detailed("search for testing")
    assert not result.ok
    assert result.output == "Error: Web search failed: offline"


def test_router_passes_legacy_prefixes_to_legacy_file_tools():
    seen = []
    agent = Agent(name="a")
    agent.add_tool(Tool("file_reader", "legacy", lambda text: seen.append(text) or "ok"))
    agent.run("read the notes")
    agent.run("please read:notes.txt")
    assert seen == ["read:read the notes", "read:notes.txt"]


def test_router_custom_structured_tool_with_single_parameter():
    @tool(name="command_runner")
    def fake(command: str) -> str:
        """Fake runner."""
        return f"ran {command}"

    agent = Agent(name="a", tools=[fake])
    assert agent.run("run tests") == "[Using command_runner]\n\nran run tests"


def test_default_response_lists_tools():
    agent = Agent(name="a")
    agent.add_tool(Tool("hello_tool", "Greeting tool", lambda x: x))
    output = agent.run("world")
    assert "received task" in output and "hello_tool" in output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Read demo.txt if it exists", "demo.txt"),
        ("open 'docs/guide.md' please", "docs/guide.md"),
        ("check C:\\data\\report.csv now", "C:\\data\\report.csv"),
        ("see https://example.com/page.html", None),
        ("no path here", None),
    ],
)
def test_extract_path(text, expected):
    assert extract_path(text) == expected


# -- LLM loop ------------------------------------------------------------------------------


def make_llm_agent(tmp_path, script, **config):
    provider = ScriptedProvider(script)
    agent = Agent(
        config=AgentConfig(
            name="llm", tools=["web_search", "file_reader"], workspace=str(tmp_path), **config
        ),
        provider=provider,
    )
    return agent, provider


def test_llm_loop_calls_tools_and_returns_final_answer(tmp_path):
    (tmp_path / "notes.txt").write_text("TeiForge notes", encoding="utf-8")
    agent, provider = make_llm_agent(
        tmp_path,
        [{"tool": "file_reader", "args": {"path": "notes.txt"}}, "The notes mention TeiForge."],
    )
    seen = []

    result = agent.run_detailed("What do my notes say?", on_event=seen.append)

    assert result.ok and result.mode == "llm" and result.steps == 2
    assert result.output == "The notes mention TeiForge."
    assert [e.type for e in result.events] == [
        "task_start",
        "llm_response",
        "tool_call",
        "tool_result",
        "llm_response",
        "final",
    ]
    assert seen == result.events
    assert result.tool_calls()[0].data["arguments"] == {"path": "notes.txt"}

    second_request = provider.requests[1]["messages"]
    assert second_request[0]["role"] == "system"
    assert second_request[-1]["role"] == "tool"
    assert "TeiForge notes" in second_request[-1]["content"]
    assert provider.requests[0]["tools"] == ["web_search", "file_reader"]


def test_llm_loop_reports_tool_errors_to_the_model(tmp_path):
    agent, provider = make_llm_agent(
        tmp_path,
        [{"tool": "file_reader", "args": {"path": "../secret"}}, {"tool": "nope", "args": {}}, "Sorry."],
    )
    result = agent.run_detailed("read secret")
    results = [e for e in result.events if e.type == "tool_result"]
    assert [r.data["ok"] for r in results] == [False, False]
    assert "outside the workspace" in provider.requests[1]["messages"][-1]["content"]
    assert provider.requests[1]["messages"][-1]["is_error"] is True
    assert "Unknown tool 'nope'" in provider.requests[2]["messages"][-1]["content"]
    assert result.output == "Sorry."


def test_llm_loop_stops_after_max_steps(tmp_path):
    loop = {"tool": "web_search", "args": {"query": ""}}
    agent, _ = make_llm_agent(tmp_path, [loop, loop], max_steps=2)
    result = agent.run_detailed("loop forever")
    assert not result.ok
    assert result.output == "Error: Stopped after 2 steps without a final answer."


def test_llm_loop_rejects_empty_answers(tmp_path):
    agent, _ = make_llm_agent(tmp_path, [LLMResponse(content="  ")])
    assert agent.run("hi") == "Error: The model returned an empty answer."


def test_llm_loop_includes_memory_and_system_prompt(tmp_path):
    agent, provider = make_llm_agent(tmp_path, ["first", "second"], system_prompt="Be brief.")
    agent.run("hello")
    agent.run("again")
    messages = provider.requests[1]["messages"]
    assert messages[0] == {"role": "system", "content": "Be brief."}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[2]["content"] == "first"


def test_llm_errors_become_failed_results(tmp_path):
    agent, _ = make_llm_agent(tmp_path, [])
    result = agent.run_detailed("hi")
    assert not result.ok
    assert result.output == "Error: ScriptedProvider has no more steps."
    assert result.events[-2].type == "error"


def test_mode_llm_without_provider_is_an_error():
    agent = Agent(config=AgentConfig(name="a", mode="llm", provider="none"))
    result = agent.run_detailed("hi")
    assert not result.ok and "needs an LLM provider" in result.output


def test_router_mode_and_enable_llm_flag_skip_the_provider(tmp_path):
    agent, provider = make_llm_agent(tmp_path, ["unused"], mode="router")
    assert agent.run_detailed("hello").mode == "router"
    agent.config.mode = "auto"
    agent.enable_llm = False
    assert agent.run_detailed("hello").mode == "router"
    assert provider.requests == []


def test_provider_is_resolved_lazily_from_config():
    agent = Agent(config=AgentConfig(name="a", provider="none"))
    assert agent.provider is None


def test_event_handler_errors_do_not_stop_the_run(tmp_path):
    agent, _ = make_llm_agent(tmp_path, ["done"])

    def broken(_event):
        raise RuntimeError("handler bug")

    assert agent.run_detailed("hi", on_event=broken).output == "done"


def test_run_result_serialises(tmp_path):
    agent, _ = make_llm_agent(tmp_path, [LLMResponse(content="ok", tool_calls=[])])
    data = agent.run_detailed("hi").to_dict()
    assert data["output"] == "ok" and data["mode"] == "llm"
    assert data["events"][0]["type"] == "task_start"
    assert "timestamp" in data["events"][0]


def test_tool_call_dataclass_defaults():
    assert ToolCall(id="1", name="x").arguments == {}


def test_summarize_handles_provider_errors(monkeypatch):
    agent = Agent(name="a")

    class Failing:
        name, model = "failing", "m"

        def complete(self, messages, tools=()):
            raise LLMError("boom")

    agent.provider = Failing()
    assert agent._summarize_text("text") is None
    assert agent._summarize_text("   ") is None

    misconfigured = Agent(config=AgentConfig(name="b", provider="openai"))
    monkeypatch.setattr(
        "teiforge.agent.provider_from_config", lambda _c: (_ for _ in ()).throw(LLMError("no key"))
    )
    assert misconfigured._summarize_text("text") is None


def test_router_explicit_read_takes_the_path_from_a_sentence(tmp_path):
    (tmp_path / "notes.txt").write_text("from notes", encoding="utf-8")
    (tmp_path / "My File.txt").write_text("spaced name", encoding="utf-8")
    agent = make_router_agent(tmp_path)
    assert "from notes" in agent.run("read: notes.txt and summarize it")
    assert "spaced name" in agent.run("read:My File.txt")
    assert "File not found: missing words" in agent.run("read: missing words")


@pytest.mark.parametrize("text", ["Summarize the notes for v0.3", "Show me the file, e.g. the main one"])
def test_extract_path_ignores_versions_and_abbreviations(text):
    assert extract_path(text) is None


def _system_prompt_sent(tmp_path, tools):
    provider = ScriptedProvider(["done"])
    agent = Agent(
        config=AgentConfig(name="prompted", tools=tools, workspace=str(tmp_path)),
        provider=provider,
    )
    agent.run("hi")
    return provider.requests[0]["messages"][0]["content"]


def test_default_system_prompt_mentions_web_search_only_when_available(tmp_path):
    without_search = _system_prompt_sent(tmp_path, ["file_reader"])
    assert "web search" not in without_search.lower()
    assert "Available tools: file_reader." in without_search

    with_search = _system_prompt_sent(tmp_path, ["web_search", "file_reader"])
    assert "source links" in with_search
    assert "Available tools: web_search, file_reader." in with_search


def test_default_system_prompt_without_tools_says_so(tmp_path):
    prompt = _system_prompt_sent(tmp_path, [])
    assert "No tools are available" in prompt
    assert "Use the available tools" not in prompt
    assert prompt.startswith("You are prompted, an assistant built with TeiForge.")
