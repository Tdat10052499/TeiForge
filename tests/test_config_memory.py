"""Tests for AgentConfig and Memory."""

import json

import pytest

from teiforge import AgentConfig, Memory

# -- AgentConfig ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"name": " "},
        {"name": "a", "model": " "},
        {"name": "a", "memory_size": 0},
        {"name": "a", "max_tokens": 0},
        {"name": "a", "temperature": 2.1},
        {"name": "a", "max_steps": 0},
        {"name": "a", "provider": "mystery"},
        {"name": "a", "mode": "fast"},
    ],
)
def test_config_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        AgentConfig(**values)


def test_config_round_trips_through_dict():
    config = AgentConfig(name="a", tools=["web_search"], provider="openai", max_steps=3)
    assert AgentConfig.from_dict(config.to_dict()) == config


def test_config_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown AgentConfig field"):
        AgentConfig.from_dict({"name": "a", "colour": "blue"})


def test_config_from_toml_with_agent_table(tmp_path):
    path = tmp_path / "agent.toml"
    path.write_text(
        '[agent]\nname = "researcher"\ntools = ["web_search"]\nprovider = "none"\nmax_steps = 4\n',
        encoding="utf-8",
    )
    config = AgentConfig.from_file(path)
    assert config.name == "researcher" and config.tools == ["web_search"] and config.max_steps == 4


def test_config_from_json_top_level(tmp_path):
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"name": "j", "mode": "router"}), encoding="utf-8")
    assert AgentConfig.from_file(path).mode == "router"


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("agent.yaml", "name: a", "must end in .toml or .json"),
        ("agent.json", "[1, 2]", "must contain a table"),
        ("agent.json", '{"agent": 5}', "must be a table"),
    ],
)
def test_config_from_file_errors(tmp_path, filename, content, message):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        AgentConfig.from_file(path)


# -- Memory ---------------------------------------------------------------------------------


def test_memory_defaults_and_validation():
    memory = Memory()
    assert len(memory) == 0 and memory.max_entries == 100
    assert str(memory) == "Memory(0 entries)"
    assert memory.get_context() == "No previous context."
    with pytest.raises(ValueError):
        Memory(max_entries=0)


def test_memory_add_get_and_trim():
    memory = Memory(max_entries=3)
    for index in range(5):
        memory.add(f"Message {index}", role="user" if index % 2 else "assistant")
    entries = memory.get()
    assert [entry.content for entry in entries] == ["Message 2", "Message 3", "Message 4"]
    entries.clear()
    assert len(memory) == 3  # get() returns a copy


def test_memory_context_and_messages():
    memory = Memory()
    memory.add("Hi", role="user")
    memory.add("Hello", role="assistant")
    memory.add("note", role="system")
    assert memory.get_context() == "User: Hi\nAssistant: Hello\nSystem: note"
    assert memory.to_messages() == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert memory.to_messages(limit=2) == [{"role": "assistant", "content": "Hello"}]


def test_memory_save_and_load_round_trip(tmp_path):
    memory = Memory(max_entries=10)
    memory.add("Xin chào", role="user")
    memory.add("Chào bạn", role="assistant")
    path = tmp_path / "memory.json"
    memory.save(path)

    assert "Xin chào" in path.read_text(encoding="utf-8")
    loaded = Memory.load(path)
    assert loaded.max_entries == 10
    assert [(e.role, e.content) for e in loaded.get()] == [("user", "Xin chào"), ("assistant", "Chào bạn")]
    assert loaded.get()[0].timestamp == memory.get()[0].timestamp

    trimmed = Memory.load(path, max_entries=1)
    assert [e.content for e in trimmed.get()] == ["Chào bạn"]


def test_memory_clear():
    memory = Memory()
    memory.add("x")
    memory.clear()
    assert len(memory) == 0
