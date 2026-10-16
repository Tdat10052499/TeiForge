"""Tests for the per-user configuration, Ollama discovery and the tool approver."""

from unittest.mock import Mock

import pytest
import requests

from teiforge import Agent, AgentConfig
from teiforge.agent import DECLINED_MESSAGE
from teiforge.llm import list_ollama_models
from teiforge.testing import ScriptedProvider
from teiforge.userconfig import chat_path, config_dir, config_path, load_user_config, save_user_config


def test_config_dir_follows_teiforge_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TEIFORGE_HOME", str(tmp_path / "home"))
    assert config_dir() == tmp_path / "home"
    assert config_path() == tmp_path / "home" / "config.toml"
    assert chat_path() == tmp_path / "home" / "chat.json"
    monkeypatch.delenv("TEIFORGE_HOME")
    assert config_dir().name == ".teiforge"


def test_save_and_load_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("TEIFORGE_HOME", str(tmp_path))
    assert load_user_config() is None
    config = AgentConfig(
        name="tui",
        provider="openai",
        model="llama3.1",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
        workspace='C:\\Users\\me\\notes "x"',
        tools=["file_reader", "file_writer"],
        system_prompt="not saved",
    )
    path = save_user_config(config)
    text = path.read_text(encoding="utf-8")
    assert "[agent]" in text and "system_prompt" not in text and "API keys are never stored" in text
    loaded = load_user_config()
    assert loaded is not None
    assert loaded.workspace == 'C:\\Users\\me\\notes "x"'
    assert (loaded.provider, loaded.model, loaded.temperature) == ("openai", "llama3.1", 0)
    assert loaded.tools == ["file_reader", "file_writer"] and loaded.system_prompt is None


def test_save_skips_unset_values(tmp_path):
    path = save_user_config(AgentConfig(name="tui"), tmp_path / "sub" / "c.toml")
    text = path.read_text(encoding="utf-8")
    assert "provider" not in text and "max_steps = 6" in text
    assert load_user_config(path).name == "tui"


def fake_response(json_data=None, error=None):
    response = Mock()
    response.raise_for_status.side_effect = error
    response.json.return_value = json_data
    return response


def test_list_ollama_models(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return fake_response({"models": [{"name": "llama3.1:latest"}, {"name": "qwen2.5:7b"}, {"x": 1}]})

    monkeypatch.setattr(requests, "get", fake_get)
    assert list_ollama_models() == ["llama3.1:latest", "qwen2.5:7b"]
    assert list_ollama_models("http://host:11434/") == ["llama3.1:latest", "qwen2.5:7b"]
    assert calls == ["http://127.0.0.1:11434/api/tags", "http://host:11434/api/tags"]


@pytest.mark.parametrize(
    "outcome",
    [
        requests.ConnectionError("refused"),
        fake_response(error=requests.HTTPError("404")),
        fake_response(json_data=["not", "a", "dict"]),
        fake_response(json_data={"models": "nope"}),
    ],
)
def test_list_ollama_models_without_a_server(monkeypatch, outcome):
    def fake_get(url, timeout):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", fake_get)
    assert list_ollama_models() is None


def test_list_ollama_models_with_no_models(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout: fake_response({"models": []}))
    assert list_ollama_models() == []


# -- approver ---------------------------------------------------------------------------


def make_agent(tmp_path, script, approver):
    return Agent(
        config=AgentConfig(name="a", tools=["file_reader", "file_writer"], workspace=str(tmp_path)),
        provider=ScriptedProvider(script),
        approver=approver,
    )


def test_approver_sees_each_call_and_can_decline(tmp_path):
    seen = []

    def approver(call):
        seen.append((call.name, call.arguments))
        return call.name != "file_writer"

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    script = [
        {"tool": "file_reader", "args": {"path": "a.txt"}},
        {"tool": "file_writer", "args": {"path": "b.txt", "content": "x"}},
        "I could not write the file because you declined.",
    ]
    agent = make_agent(tmp_path, script, approver)
    result = agent.run_detailed("copy a.txt to b.txt")

    assert seen == [("file_reader", {"path": "a.txt"}), ("file_writer", {"path": "b.txt", "content": "x"})]
    assert not (tmp_path / "b.txt").exists()
    results = [e.data for e in result.events if e.type == "tool_result"]
    assert results[0]["ok"] and "declined" not in results[0]
    assert results[1] == {
        "step": 2,
        "id": "call_2",
        "name": "file_writer",
        "ok": False,
        "output": DECLINED_MESSAGE,
        "declined": True,
    }
    # A declined tool counts as failed, so the self-check does not ask again.
    assert not any(e.type == "self_check" for e in result.events)


def test_approver_applies_to_the_router(tmp_path):
    agent = Agent(
        config=AgentConfig(name="a", tools=["file_writer"], workspace=str(tmp_path), mode="router"),
        approver=lambda call: False,
    )
    result = agent.run_detailed("write:b.txt:hello")
    assert not result.ok and result.output == "Error: The user declined to run this tool."
    assert not (tmp_path / "b.txt").exists()
    assert [e.data.get("declined") for e in result.events if e.type == "tool_result"] == [True]

    calls = []
    agent.approver = lambda call: calls.append(call) or True
    assert agent.run_detailed("write:b.txt:hello").ok
    assert calls[0].arguments == {"path": "b.txt", "content": "hello"}
