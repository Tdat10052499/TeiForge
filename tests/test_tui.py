"""Tests for the Textual terminal UI, driven headlessly with Textual's pilot."""

import asyncio
import json
import threading
from datetime import datetime, timezone

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, Markdown, RichLog, Select, Static  # noqa: E402

import teiforge.tui as tui_package  # noqa: E402
import teiforge.tui.app as tui_app  # noqa: E402
from teiforge import AgentConfig, LLMResponse  # noqa: E402
from teiforge.cli import main  # noqa: E402
from teiforge.events import Event  # noqa: E402
from teiforge.llm import OLLAMA_BASE_URL, ToolCall  # noqa: E402
from teiforge.memory import Memory  # noqa: E402
from teiforge.testing import ScriptedProvider  # noqa: E402
from teiforge.tui.app import ApprovalScreen, SetupScreen, TeiForgeApp, config_from_choices  # noqa: E402
from teiforge.tui.steps import default_tools, describe_call, format_step, needs_approval  # noqa: E402
from teiforge.userconfig import load_user_config, save_user_config  # noqa: E402

SIZE = (120, 40)


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("TEIFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(tui_app, "list_ollama_models", lambda: ["llama3.1:latest", "qwen2.5:7b"])
    return tmp_path / "home"


@pytest.fixture
def workspace(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "meeting.txt").write_text("Lan sends the budget. Minh prepares slides.", encoding="utf-8")
    return folder


def make_config(workspace, **extra):
    return AgentConfig(
        name="teiforge", tools=["file_reader", "file_writer"], workspace=str(workspace), **extra
    )


async def wait_until(pilot, condition, steps=200):
    for _ in range(steps):
        await pilot.pause(0.05)
        if condition():
            return True
    return False


async def ask(pilot, app, text):
    prompt = app.query_one("#prompt", Input)
    prompt.value = text
    await pilot.press("enter")


def texts(app, selector):
    return [str(widget.render()) for widget in app.query(selector)]


# -- helpers without the UI ----------------------------------------------------------


def event(kind, **data):
    return Event(type=kind, data=data, timestamp=datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc))


def test_format_step_covers_every_event():
    lines = [
        format_step(event("task_start", task="Read [notes]")),
        format_step(event("llm_response", tool_calls=["file_reader"])),
        format_step(event("llm_response", tool_calls=[])),
        format_step(event("tool_call", name="file_reader", arguments={"path": "a.txt"}, recovered=True)),
        format_step(event("tool_result", ok=True, output="12345")),
        format_step(event("tool_result", ok=False, output="Error: nope")),
        format_step(event("tool_result", ok=False, output="", declined=True)),
        format_step(event("self_check")),
        format_step(event("error", message="boom")),
        format_step(event("final", ok=True)),
        format_step(event("final", ok=False)),
    ]
    assert all(lines)
    assert "Read \\[notes]" in lines[0]
    assert "asks for file_reader" in lines[1] and "answered" in lines[2]
    assert "path=a.txt" in lines[3] and "recovered from text" in lines[3]
    assert "5 chars" in lines[4] and "Error: nope" in lines[5] and "declined" in lines[6]
    assert "check its answer" in lines[7] and "boom" in lines[8]
    assert "done" in lines[9] and "failed" in lines[10]
    assert format_step(Event(type="unknown", data={})) is None  # type: ignore[arg-type]


def test_long_arguments_are_shortened():
    line = format_step(event("tool_call", name="file_writer", arguments={"content": "x" * 200}))
    assert "..." in line and len(line) < 200


def test_describe_call_and_approval_rules():
    write = ToolCall(id="1", name="file_writer", arguments={"path": "a.md", "content": "y" * 700})
    assert describe_call(write).startswith("create the file a.md (700 characters)")
    assert describe_call(write).endswith("...")
    overwrite = ToolCall(
        id="2", name="file_writer", arguments={"path": "a.md", "content": "", "overwrite": True}
    )
    assert describe_call(overwrite).startswith("overwrite")
    command = ToolCall(id="3", name="command_runner", arguments={"command": "ls"})
    assert describe_call(command) == "run the command:\n\nls"
    other = ToolCall(id="4", name="lookup", arguments={"q": "x"})
    assert describe_call(other) == 'run lookup with {"q": "x"}'
    assert needs_approval(write) and needs_approval(command) and not needs_approval(other)


def test_default_tools_add_commands_only_when_enabled(monkeypatch):
    assert default_tools() == ["web_search", "file_reader", "file_writer"]
    monkeypatch.setenv("TEIFORGE_ENABLE_COMMANDS", "1")
    assert default_tools()[-1] == "command_runner"


def test_config_from_choices(tmp_path):
    ollama = config_from_choices("ollama", "llama3.1:latest", str(tmp_path))
    assert (ollama.provider, ollama.base_url, ollama.model) == ("openai", OLLAMA_BASE_URL, "llama3.1:latest")
    assert ollama.workspace == str(tmp_path.resolve()) and ollama.temperature == 0
    assert config_from_choices("openai", " ", str(tmp_path)).model == "gpt-4o-mini"
    assert config_from_choices("anthropic", "claude-x", str(tmp_path)).model == "claude-x"
    none = config_from_choices("none", "", "")
    assert none.provider == "none" and none.workspace


# -- the chat screen -------------------------------------------------------------------


def test_chat_shows_answer_steps_and_saves_memory(home, workspace):
    provider = ScriptedProvider(
        [{"tool": "file_reader", "args": {"path": "meeting.txt"}}, "**Minh** does slides."]
    )

    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=provider)
        async with app.run_test(size=SIZE) as pilot:
            assert "workspace: notes" in app.sub_title
            await ask(pilot, app, "Who prepares the slides?")
            assert await wait_until(pilot, lambda: not app.busy)
            assert app.query(Markdown).last().source == "**Minh** does slides."
            steps = [str(line.text) for line in app.query_one("#steps", RichLog).lines]
            assert any("file_reader" in line for line in steps) and any("done" in line for line in steps)
            assert "Ready." in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())
    saved = json.loads((home / "chat.json").read_text(encoding="utf-8"))
    assert [entry["role"] for entry in saved["entries"]] == ["user", "assistant"]


def test_writes_need_approval(home, workspace):
    write = {"tool": "file_writer", "args": {"path": "out.md", "content": "hi"}}
    provider = ScriptedProvider([write, "Declined.", write, "Written.", write, "Again."])

    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=provider)
        async with app.run_test(size=SIZE) as pilot:
            await ask(pilot, app, "write out.md")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            assert "create the file out.md" in "".join(texts(app.screen, "Static"))
            await pilot.press("n")
            assert await wait_until(pilot, lambda: not app.busy)
            assert not (workspace / "out.md").exists()

            await ask(pilot, app, "write out.md")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            await pilot.click("#session")
            assert await wait_until(pilot, lambda: not app.busy)
            assert (workspace / "out.md").read_text(encoding="utf-8") == "hi"

            (workspace / "out.md").unlink()
            await ask(pilot, app, "write again")
            assert await wait_until(pilot, lambda: not app.busy)
            assert not isinstance(app.screen, ApprovalScreen) and (workspace / "out.md").exists()

    asyncio.run(scenario())


def test_approval_allow_once(home, workspace):
    provider = ScriptedProvider([{"tool": "file_writer", "args": {"path": "x.md", "content": "1"}}, "ok"])

    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=provider)
        async with app.run_test(size=SIZE) as pilot:
            await ask(pilot, app, "write x")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            await pilot.press("y")
            assert await wait_until(pilot, lambda: not app.busy)
            assert (workspace / "x.md").exists() and "file_writer" not in app.allowed_for_session

    asyncio.run(scenario())


def test_commands(home, workspace):
    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=ScriptedProvider(["one"]))
        async with app.run_test(size=SIZE) as pilot:
            await ask(pilot, app, "hello")
            assert await wait_until(pilot, lambda: not app.busy)
            for command in ("/help", "/tools", "/memory", "/nope"):
                await ask(pilot, app, command)
                await pilot.pause()
            sources = [widget.source for widget in app.query(Markdown)]
            assert any("**Commands**" in source for source in sources)
            assert any("file_reader" in source for source in sources)
            assert any("Memory entries: 2" in source for source in sources)
            assert any("Unknown command /nope" in text for text in texts(app, ".failed"))

            await ask(pilot, app, "/clear")
            await pilot.pause()
            assert len(app.agent.memory) == 0 and not list(app.query(".you"))

            await ask(pilot, app, "")
            await ask(pilot, app, "/setup")
            assert await wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
            await pilot.click("#cancel")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))
            await pilot.press("ctrl+l")
            await ask(pilot, app, "exit")
            await pilot.pause()
        assert app.return_code == 0

    asyncio.run(scenario())


class BlockingProvider:
    """Waits until released, so a test can stop waiting while the model is busy."""

    name = "blocking"
    model = "slow"

    def __init__(self):
        self.release = threading.Event()
        self.answers = ["late answer", "second answer"]

    def complete(self, messages, tools=()):
        if len(self.answers) == 2:
            self.release.wait(5)
        return LLMResponse(content=self.answers.pop(0))


def test_escape_stops_waiting_and_drops_the_late_answer(home, workspace):
    provider = BlockingProvider()

    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=provider)
        async with app.run_test(size=SIZE) as pilot:
            await ask(pilot, app, "slow question")
            assert await wait_until(pilot, lambda: app.busy)
            await ask(pilot, app, "too early")
            await pilot.press("escape")
            assert not app.busy and "Stopped waiting" in str(app.query_one("#status", Static).render())
            await pilot.press("escape")  # nothing to stop
            provider.release.set()
            await pilot.pause(0.3)
            assert not any(widget.source == "late answer" for widget in app.query(Markdown))
            assert len(app.agent.memory) == 0
            await ask(pilot, app, "next question")
            assert await wait_until(pilot, lambda: not app.busy)
            assert app.query(Markdown).last().source == "second answer"

    asyncio.run(scenario())


def test_failures_are_shown(home, workspace):
    class Broken:
        name, model = "broken", "x"

        def complete(self, messages, tools=()):
            raise RuntimeError("socket closed")

    async def scenario():
        app = TeiForgeApp(make_config(workspace), provider=Broken())
        async with app.run_test(size=SIZE) as pilot:
            await ask(pilot, app, "hi")
            assert await wait_until(pilot, lambda: not app.busy)
            assert "RuntimeError: socket closed" in texts(app, ".failed")[-1]

    asyncio.run(scenario())


def test_saved_memory_is_loaded_and_bad_files_are_reported(home, workspace):
    memory = Memory()
    memory.add("earlier", role="user")
    memory.add("reply", role="assistant")
    home.mkdir(parents=True)
    memory.save(home / "chat.json")

    async def loaded():
        app = TeiForgeApp(make_config(workspace), provider=ScriptedProvider([]))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert len(app.agent.memory) == 2
            assert "2 earlier messages" in str(app.query_one("#status", Static).render())

    asyncio.run(loaded())
    (home / "chat.json").write_text("{not json", encoding="utf-8")

    async def broken():
        app = TeiForgeApp(make_config(workspace), provider=ScriptedProvider([]))
        async with app.run_test(size=SIZE, notifications=True) as pilot:
            await pilot.pause()
            assert len(app.agent.memory) == 0

    asyncio.run(broken())


def test_router_mode_without_a_model(home, workspace):
    async def scenario():
        app = TeiForgeApp(make_config(workspace, provider="none"))
        async with app.run_test(size=SIZE) as pilot:
            assert "keyword router" in app.sub_title
            await ask(pilot, app, "read:meeting.txt")
            assert await wait_until(pilot, lambda: not app.busy)
            assert "Minh prepares slides" in app.query(Markdown).last().source

    asyncio.run(scenario())


# -- setup -------------------------------------------------------------------------------


def test_first_run_setup_tests_and_saves_ollama(home, workspace, monkeypatch):
    monkeypatch.chdir(workspace)

    async def scenario():
        app = TeiForgeApp(provider=ScriptedProvider(["pong"]))
        async with app.run_test(size=SIZE) as pilot:
            assert await wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
            assert await wait_until(pilot, lambda: app.screen.ollama_models is not None)
            assert app.screen.query_one("#ollama-model", Select).value == "llama3.1:latest"
            await pilot.click("#save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))
            assert app.agent is not None and not app.query_one("#prompt", Input).disabled

    asyncio.run(scenario())
    saved = load_user_config()
    assert saved.model == "llama3.1:latest" and saved.base_url == OLLAMA_BASE_URL
    assert saved.workspace == str(workspace.resolve())


def test_setup_reports_a_model_that_does_not_answer(home, workspace, monkeypatch):
    monkeypatch.chdir(workspace)

    async def scenario():
        app = TeiForgeApp(provider=ScriptedProvider([LLMResponse(content="")]))
        async with app.run_test(size=SIZE) as pilot:
            assert await wait_until(pilot, lambda: getattr(app.screen, "ollama_models", None) is not None)
            await pilot.click("#save")
            status = app.screen.query_one("#setup-status", Static)
            assert await wait_until(pilot, lambda: "did not answer" in str(status.render()))
            assert isinstance(app.screen, SetupScreen) and load_user_config() is None
            await pilot.press("escape")  # cannot cancel the first setup
            assert isinstance(app.screen, SetupScreen)

    asyncio.run(scenario())


def test_setup_other_providers_and_messages(home, workspace, monkeypatch):
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(tui_app, "list_ollama_models", lambda: None)

    async def scenario():
        app = TeiForgeApp()
        async with app.run_test(size=SIZE) as pilot:
            screen = app.screen
            status = screen.query_one("#setup-status", Static)
            assert await wait_until(pilot, lambda: "not found" in str(status.render()))
            await pilot.click("#save")
            await pilot.pause()
            assert "Choose an Ollama model" in str(status.render())

            screen.query_one("#provider", Select).value = "openai"
            await pilot.pause()
            name = screen.query_one("#model-name", Input)
            assert name.display and name.value == "gpt-4o-mini"
            screen.query_one("#workspace", Input).value = str(workspace / "new" / "folder")
            screen.query_one("#provider", Select).value = "none"
            await pilot.pause()
            await pilot.click("#save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))
            assert (workspace / "new" / "folder").is_dir()
            assert "keyword router" in app.sub_title

    asyncio.run(scenario())
    assert load_user_config().provider == "none"


def test_setup_with_an_empty_ollama(home, monkeypatch):
    monkeypatch.setattr(tui_app, "list_ollama_models", lambda: [])

    async def scenario():
        app = TeiForgeApp()
        async with app.run_test(size=SIZE) as pilot:
            status = app.screen.query_one("#setup-status", Static)
            assert await wait_until(pilot, lambda: "ollama pull" in str(status.render()))

    asyncio.run(scenario())


def test_saved_config_and_environment_skip_setup(home, workspace, monkeypatch):
    save_user_config(make_config(workspace, provider="none"))

    async def from_file():
        app = TeiForgeApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert not isinstance(app.screen, SetupScreen) and app.agent.config.tools

    asyncio.run(from_file())
    (home / "config.toml").unlink()
    monkeypatch.setattr("teiforge.agent.provider_from_config", lambda config: ScriptedProvider([]))

    async def from_environment():
        app = TeiForgeApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert not isinstance(app.screen, SetupScreen)
            assert app.agent.config.tools == default_tools()

    asyncio.run(from_environment())


def test_environment_errors_lead_to_setup(home, monkeypatch):
    from teiforge.llm import LLMError

    def broken(config):
        raise LLMError("bad provider")

    monkeypatch.setattr("teiforge.agent.provider_from_config", broken)

    async def scenario():
        app = TeiForgeApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SetupScreen)

    asyncio.run(scenario())


# -- entry points ------------------------------------------------------------------------


def test_bare_teiforge_opens_the_ui_only_in_a_terminal(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(tui_package, "run_tui", lambda: opened.append(True) or 0)
    monkeypatch.setattr("teiforge.cli._interactive", lambda: True)
    assert main([]) == 0 and opened == [True]

    monkeypatch.setattr(tui_package, "textual_available", lambda: False)
    assert main([]) == 2
    out = capsys.readouterr().out
    assert 'pip install "teiforge[tui]"' in out and "usage: teiforge" in out

    monkeypatch.setattr("teiforge.cli._interactive", lambda: False)
    assert main([]) == 2 and opened == [True]


def test_ui_command_and_missing_textual(monkeypatch, capsys):
    monkeypatch.setattr(tui_app, "run_tui", lambda: 0)
    assert main(["ui"]) == 0
    monkeypatch.setattr(tui_package, "textual_available", lambda: False)
    assert tui_package.run_tui() == 2
    assert "teiforge[tui]" in capsys.readouterr().out
