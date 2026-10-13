"""Tests for the memory module."""

from agentforge.memory import Memory


def test_memory_creation():
    """Test memory creation."""
    memory = Memory()
    assert len(memory) == 0
    assert memory.max_entries == 100


def test_memory_add():
    """Test adding to memory."""
    memory = Memory()
    memory.add("Test message")
    assert len(memory) == 1


def test_memory_add_with_role():
    """Test adding message with role."""
    memory = Memory()
    memory.add("User message", role="user")
    memory.add("Assistant response", role="assistant")

    entries = memory.get()
    assert len(entries) == 2
    assert entries[0].role == "user"
    assert entries[1].role == "assistant"


def test_memory_get():
    """Test getting memory entries."""
    memory = Memory()
    memory.add("Message 1")
    memory.add("Message 2")

    entries = memory.get()
    assert len(entries) == 2
    assert entries[0].content == "Message 1"
    assert entries[1].content == "Message 2"


def test_memory_clear():
    """Test clearing memory."""
    memory = Memory()
    memory.add("Message 1")
    memory.add("Message 2")

    assert len(memory) == 2
    memory.clear()
    assert len(memory) == 0


def test_memory_max_entries():
    """Test memory max entries limit."""
    memory = Memory(max_entries=3)

    memory.add("Message 1")
    memory.add("Message 2")
    memory.add("Message 3")
    memory.add("Message 4")  # Should remove Message 1

    assert len(memory) == 3
    entries = memory.get()
    assert entries[0].content == "Message 2"
    assert entries[-1].content == "Message 4"


def test_memory_context():
    """Test getting memory context."""
    memory = Memory()
    memory.add("User: Hello", role="user")
    memory.add("Assistant: Hi!", role="assistant")

    context = memory.get_context()
    assert "User" in context
    assert "Assistant" in context
    assert "Hello" in context
    assert "Hi!" in context


def test_memory_len():
    """Test memory length."""
    memory = Memory()
    assert len(memory) == 0

    memory.add("Message")
    assert len(memory) == 1


def test_memory_str():
    """Test memory string representation."""
    memory = Memory()
    memory.add("Message 1")
    memory.add("Message 2")

    str_repr = str(memory)
    assert "Memory" in str_repr
    assert "2" in str_repr
