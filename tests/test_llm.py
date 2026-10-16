"""Tests for the LLM providers, using fake SDK clients (no network)."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from teiforge import AgentConfig, tool
from teiforge.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicProvider,
    LLMError,
    LLMProvider,
    OpenAIProvider,
    ToolCall,
    _parse_arguments,
    parse_text_tool_calls,
    provider_from_config,
)
from teiforge.testing import ScriptedProvider


@tool
def lookup(term: str) -> str:
    """Look something up."""
    return term


CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [ToolCall(id="c1", name="lookup", arguments={"term": "x"})],
    },
    {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "result", "is_error": False},
]


# -- OpenAI ---------------------------------------------------------------------------------


def openai_client(message):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return client


def test_openai_provider_sends_config_and_tools():
    client = openai_client(SimpleNamespace(content="answer", tool_calls=None))
    provider = OpenAIProvider("m", temperature=0.4, max_tokens=123, client=client)

    response = provider.complete(CONVERSATION, [lookup])

    assert response.content == "answer" and response.tool_calls == []
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "m"
    assert request["temperature"] == 0.4 and request["max_tokens"] == 123
    assert request["tools"][0]["function"]["name"] == "lookup"
    assistant = request["messages"][2]
    assert assistant["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"term": "x"}'}
    assert request["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_openai_provider_parses_tool_calls():
    call = SimpleNamespace(id="c9", function=SimpleNamespace(name="lookup", arguments='{"term": "y"}'))
    client = openai_client(SimpleNamespace(content=None, tool_calls=[call]))
    response = OpenAIProvider("m", client=client).complete([{"role": "user", "content": "q"}])
    assert response.tool_calls == [ToolCall(id="c9", name="lookup", arguments={"term": "y"})]
    assert "tools" not in client.chat.completions.create.call_args.kwargs


def test_openai_provider_wraps_sdk_errors():
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("rate limited")
    with pytest.raises(LLMError, match="OpenAI request failed: RuntimeError: rate limited"):
        OpenAIProvider("m", client=client).complete([{"role": "user", "content": "q"}])


def install_fake_module(monkeypatch, name, attribute):
    module = types.ModuleType(name)
    constructor = Mock()
    setattr(module, attribute, constructor)
    monkeypatch.setitem(sys.modules, name, module)
    return constructor


def test_openai_provider_builds_client_from_environment(monkeypatch):
    constructor = install_fake_module(monkeypatch, "openai", "OpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    OpenAIProvider("m")
    constructor.assert_called_once_with(api_key="sk-test", base_url=None)


def test_openai_provider_with_base_url_needs_no_key(monkeypatch):
    constructor = install_fake_module(monkeypatch, "openai", "OpenAI")
    OpenAIProvider("llama3", base_url="http://localhost:11434/v1")
    constructor.assert_called_once_with(api_key="not-needed", base_url="http://localhost:11434/v1")


def test_openai_provider_errors_without_key_or_package(monkeypatch):
    install_fake_module(monkeypatch, "openai", "OpenAI")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAIProvider("m")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMError, match="teiforge\\[openai\\]"):
        OpenAIProvider("m")


# -- Anthropic ------------------------------------------------------------------------------


def test_anthropic_message_conversion_merges_tool_results():
    conversation = CONVERSATION + [
        {"role": "tool", "tool_call_id": "c2", "name": "lookup", "content": "bad", "is_error": True}
    ]
    system, messages = AnthropicProvider.convert_messages(conversation)
    assert system == "sys"
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["content"] == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"term": "x"}},
    ]
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "result"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "bad", "is_error": True},
        ],
    }


def test_anthropic_tool_use_without_text():
    _, messages = AnthropicProvider.convert_messages(
        [{"role": "assistant", "content": "", "tool_calls": [ToolCall(id="a", name="lookup")]}]
    )
    assert messages[0]["content"] == [{"type": "tool_use", "id": "a", "name": "lookup", "input": {}}]


def test_anthropic_provider_request_and_response():
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(type="tool_use", id="t1", name="lookup", input={"term": "z"}),
        ]
    )
    provider = AnthropicProvider("claude-test", temperature=1.5, max_tokens=50, client=client)

    response = provider.complete(CONVERSATION, [lookup])

    assert response.content == "Let me check."
    assert response.tool_calls == [ToolCall(id="t1", name="lookup", arguments={"term": "z"})]
    request = client.messages.create.call_args.kwargs
    assert request["system"] == "sys"
    assert request["temperature"] == 1.0  # clamped to Anthropic's range
    assert request["tools"][0]["input_schema"] == lookup.parameters


def test_anthropic_provider_without_system_or_tools():
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(content=[])
    response = AnthropicProvider(client=client).complete([{"role": "user", "content": "q"}])
    assert response.content is None
    request = client.messages.create.call_args.kwargs
    assert "system" not in request and "tools" not in request
    assert request["model"] == DEFAULT_ANTHROPIC_MODEL


def test_anthropic_provider_wraps_errors_and_requires_key(monkeypatch):
    client = Mock()
    client.messages.create.side_effect = RuntimeError("overloaded")
    with pytest.raises(LLMError, match="Anthropic request failed"):
        AnthropicProvider(client=client).complete([{"role": "user", "content": "q"}])

    constructor = install_fake_module(monkeypatch, "anthropic", "Anthropic")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    AnthropicProvider()
    constructor.assert_called_once_with(api_key="key")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMError, match="teiforge\\[anthropic\\]"):
        AnthropicProvider()


# -- helpers and discovery ------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["not json", "[1, 2]"])
def test_parse_arguments_rejects_bad_json(raw):
    with pytest.raises(LLMError):
        _parse_arguments(raw)


def test_parse_arguments_accepts_dicts_and_empty():
    assert _parse_arguments({"a": 1}) == {"a": 1}
    assert _parse_arguments("") == {}


def test_auto_discovery_returns_none_without_keys():
    assert provider_from_config(AgentConfig(name="a")) is None


def test_explicit_none_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert provider_from_config(AgentConfig(name="a", provider="none")) is None


def test_auto_discovery_prefers_openai_then_anthropic(monkeypatch):
    install_fake_module(monkeypatch, "openai", "OpenAI")
    install_fake_module(monkeypatch, "anthropic", "Anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    provider = provider_from_config(AgentConfig(name="a"))
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == DEFAULT_ANTHROPIC_MODEL  # gpt default swapped for a Claude model

    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    provider = provider_from_config(AgentConfig(name="a", model="gpt-x", temperature=0.2))
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-x" and provider.temperature == 0.2


def test_environment_selects_provider_model_and_base_url(monkeypatch):
    constructor = install_fake_module(monkeypatch, "openai", "OpenAI")
    monkeypatch.setenv("TEIFORGE_PROVIDER", "openai")
    monkeypatch.setenv("TEIFORGE_MODEL", "llama3")
    monkeypatch.setenv("TEIFORGE_BASE_URL", "http://localhost:11434/v1")
    provider = provider_from_config(AgentConfig(name="a"))
    assert provider.model == "llama3"
    constructor.assert_called_once_with(api_key="not-needed", base_url="http://localhost:11434/v1")


def test_unknown_provider_from_environment(monkeypatch):
    monkeypatch.setenv("TEIFORGE_PROVIDER", "mystery")
    with pytest.raises(LLMError, match="Unknown provider"):
        provider_from_config(AgentConfig(name="a"))


def test_scripted_provider_is_an_llm_provider():
    provider = ScriptedProvider(["x"])
    assert isinstance(provider, LLMProvider)
    assert provider.complete([{"role": "user", "content": "q"}]).content == "x"


@pytest.mark.parametrize(
    ("model", "reasoning"),
    [
        ("o3-mini", True),
        ("o1", True),
        ("gpt-5", True),
        ("openai/o4-mini", True),
        ("gpt-4o-mini", False),
        ("llama3", False),
    ],
)
def test_reasoning_model_detection(model, reasoning):
    assert OpenAIProvider.is_reasoning_model(model) is reasoning


def test_reasoning_models_use_max_completion_tokens():
    client = openai_client(SimpleNamespace(content="ok", tool_calls=None))
    OpenAIProvider("o3-mini", temperature=0.3, max_tokens=99, client=client).complete(
        [{"role": "user", "content": "q"}]
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["max_completion_tokens"] == 99
    assert "max_tokens" not in request and "temperature" not in request


# -- tool calls written as text ---------------------------------------------------------

NAMES = ["file_reader", "days_until"]


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "file_reader", "parameters": {"path": "meeting.txt"}}',
        '  {"name": "file_reader", "arguments": "{\\"path\\": \\"meeting.txt\\"}"}  ',
        '```json\n{"name": "file_reader", "arguments": {"path": "meeting.txt"}}\n```',
        '<tool_call>\n{"name": "file_reader", "arguments": {"path": "meeting.txt"}}\n</tool_call>',
        '<|python_tag|>{"name": "file_reader", "parameters": {"path": "meeting.txt"}}',
        '{"type": "function", "function": {"name": "file_reader", "arguments": {"path": "meeting.txt"}}}',
    ],
)
def test_parse_text_tool_calls_accepts_common_formats(content):
    calls = parse_text_tool_calls(content, NAMES)
    assert [(call.id, call.name, call.arguments) for call in calls] == [
        ("text_call_1", "file_reader", {"path": "meeting.txt"})
    ]


def test_parse_text_tool_calls_handles_several_calls():
    as_list = '[{"name": "file_reader", "parameters": {"path": "a.txt"}}, {"name": "days_until"}]'
    per_line = '{"name": "file_reader", "parameters": {"path": "a.txt"}};\n\n{"name": "days_until"}'
    tagged = (
        '<tool_call>{"name": "file_reader", "arguments": {"path": "a.txt"}}</tool_call>'
        '<tool_call>{"name": "days_until", "arguments": ""}</tool_call>'
    )
    for content in (as_list, per_line, tagged):
        calls = parse_text_tool_calls(content, NAMES)
        assert [call.name for call in calls] == ["file_reader", "days_until"]
        assert [call.id for call in calls] == ["text_call_1", "text_call_2"]
        assert calls[1].arguments == {}


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "The file_reader tool read meeting.txt for you.",
        'I will call {"name": "file_reader", "parameters": {"path": "a.txt"}} now.',
        '{"name": "delete_everything", "parameters": {}}',
        '[{"name": "file_reader", "parameters": {}}, {"name": "unknown"}]',
        '{"name": "file_reader", "parameters": ["a.txt"]}',
        '{"name": "file_reader", "arguments": "not json"}',
        '{"tool": "file_reader"}',
        '"file_reader"',
        '<tool_call>{"name": "file_reader"}</tool_call> and some prose',
        '{"name": "file_reader"}\nnot json',
    ],
)
def test_parse_text_tool_calls_rejects_anything_else(content):
    assert parse_text_tool_calls(content, NAMES) == []


def test_parse_text_tool_calls_needs_registered_tools():
    assert parse_text_tool_calls('{"name": "file_reader"}', []) == []
