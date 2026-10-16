"""The full-screen terminal UI, built with Textual: ``teiforge`` or ``teiforge ui``."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    RichLog,
    Select,
    Static,
)

from .._version import __version__
from ..agent import Agent
from ..config import AgentConfig
from ..events import Event, RunResult
from ..llm import (
    DEFAULT_ANTHROPIC_MODEL,
    OLLAMA_BASE_URL,
    LLMError,
    LLMProvider,
    ToolCall,
    list_ollama_models,
)
from ..memory import Memory
from ..userconfig import chat_path, config_path, load_user_config, save_user_config
from .steps import default_tools, describe_call, format_step, needs_approval

HELP_TEXT = """\
**Commands**

| Command | What it does |
| --- | --- |
| `/help` | Show this help |
| `/tools` | List the agent's tools |
| `/memory` | Show the memory summary |
| `/clear` | Forget the conversation |
| `/setup` | Choose another model or folder |
| `/exit` | Save and quit (`exit`, `quit` and Ctrl+Q work too) |

**Keys:** Esc stops waiting for the model · F2 opens setup · Ctrl+L clears the conversation.

The agent can only read and write files inside the workspace folder shown on the right, and asks
you before it writes a file.
"""

TEST_PROMPT = "Reply with only the word pong."
CHAT_AGENT_NAME = "teiforge"

PROVIDER_CHOICES: List[Tuple[str, str]] = [
    ("Ollama on this computer (free, private)", "ollama"),
    ("OpenAI (reads OPENAI_API_KEY)", "openai"),
    ("Anthropic (reads ANTHROPIC_API_KEY)", "anthropic"),
    ("No model: keyword router only", "none"),
]
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": DEFAULT_ANTHROPIC_MODEL}

APP_CSS = """
#main { height: 1fr; }
#chat-pane { width: 3fr; }
#side-pane { width: 2fr; border-left: solid $panel; }
#messages { height: 1fr; padding: 0 1; }
.you { margin: 1 0 0 0; color: $accent; }
.answer { margin: 0 0 0 2; }
.note { margin: 1 0 0 2; color: $text-muted; }
.failed { margin: 0 0 0 2; color: $error; }
#status { height: 1; padding: 0 1; color: $text-muted; }
.pane-title { height: 1; padding: 0 1; background: $panel; text-style: bold; }
#steps { height: 1fr; overflow-x: hidden; }
#tree-box { height: 1fr; }
SetupScreen { align: center middle; }
#setup-box { width: 76; height: auto; border: round $accent; padding: 1 2; }
#setup-box Label { margin: 1 0 0 0; }
#setup-status { margin: 1 0; height: auto; }
#setup-buttons { height: auto; margin: 1 0 0 0; }
ApprovalScreen { align: center middle; }
#approval-box { width: 80; height: auto; max-height: 90%; border: thick $warning; padding: 1 2; }
#approval-text { height: auto; max-height: 20; margin: 1 0; }
#approval-buttons { height: auto; }
"""


def config_from_choices(provider: str, model: str, workspace: str) -> AgentConfig:
    """Build the agent configuration the setup screen saves."""
    folder = str(Path(workspace).expanduser().resolve()) if workspace.strip() else str(Path.cwd())
    config = AgentConfig(name=CHAT_AGENT_NAME, tools=default_tools(), workspace=folder, temperature=0)
    if provider == "ollama":
        config.provider, config.base_url = "openai", OLLAMA_BASE_URL
    else:
        config.provider = provider
    if provider != "none":
        config.model = model.strip() or DEFAULT_MODELS.get(provider, config.model)
    return config


# -- dialogs ---------------------------------------------------------------------------


class ApprovalScreen(ModalScreen[str]):
    """Ask whether a tool that changes something may run: ``allow``, ``session`` or ``deny``."""

    BINDINGS = [
        Binding("y", "choose('allow')", "Allow"),
        Binding("a", "choose('session')", "Allow for this session"),
        Binding("n,escape", "choose('deny')", "Deny"),
    ]

    def __init__(self, call: ToolCall) -> None:
        super().__init__()
        self.call = call

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(f"[b]The agent wants to use {escape(self.call.name)}[/b]")
            with VerticalScroll(id="approval-text"):
                yield Static(escape(f"It wants to {describe_call(self.call)}"))
            with Horizontal(id="approval-buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Allow for this session (a)", id="session")
                yield Button("Deny (n)", variant="error", id="deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "deny")

    def action_choose(self, choice: str) -> None:
        self.dismiss(choice)


class SetupScreen(Screen[Optional[AgentConfig]]):
    """First-run setup: pick a model provider, a model and a workspace, test, then save."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, current: Optional[AgentConfig] = None, *, can_cancel: bool = False) -> None:
        super().__init__()
        self.current = current
        self.can_cancel = can_cancel
        self.ollama_models: Optional[List[str]] = None

    def compose(self) -> ComposeResult:
        workspace = (self.current.workspace if self.current else None) or str(Path.cwd())
        with Vertical(id="setup-box"):
            yield Static(f"[b]TeiForge {__version__} setup[/b]\nChoose the model the agent uses.")
            yield Label("Where should the model run?")
            yield Select(PROVIDER_CHOICES, value="ollama", allow_blank=False, id="provider")
            yield Label("Model")
            yield Select([], prompt="Looking for Ollama models...", id="ollama-model")
            yield Input(placeholder="Model name, e.g. gpt-4o-mini", id="model-name")
            yield Label("Workspace: the only folder the agent may read and write")
            yield Input(value=workspace, id="workspace")
            yield Static("", id="setup-status")
            with Horizontal(id="setup-buttons"):
                yield Button("Test and save", variant="primary", id="save")
                if self.can_cancel:
                    yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#model-name", Input).display = False
        self.detect_ollama()

    @work(thread=True, exclusive=True, group="setup-detect")
    def detect_ollama(self) -> None:
        models = list_ollama_models()
        self.app.call_from_thread(self.show_ollama_models, models)

    def show_ollama_models(self, models: Optional[List[str]]) -> None:
        self.ollama_models = models
        select = self.query_one("#ollama-model", Select)
        status = self.query_one("#setup-status", Static)
        if models:
            select.set_options([(name, name) for name in models])
            preferred = next((name for name in models if name.startswith("llama3.1")), models[0])
            select.value = preferred
            status.update(f"[green]Found Ollama with {len(models)} model(s).[/]")
        elif models == []:
            status.update("[yellow]Ollama is running but has no models. Run: ollama pull llama3.1[/]")
        else:
            status.update(
                "[yellow]Ollama was not found on 127.0.0.1:11434. Start it, or pick another option.[/]"
            )

    @on(Select.Changed, "#provider")
    def provider_changed(self, event: Select.Changed) -> None:
        provider = str(event.value)
        self.query_one("#ollama-model", Select).display = provider == "ollama"
        name = self.query_one("#model-name", Input)
        name.display = provider in DEFAULT_MODELS
        if provider in DEFAULT_MODELS and not name.value:
            name.value = DEFAULT_MODELS[provider]

    def chosen_config(self) -> AgentConfig:
        provider = str(self.query_one("#provider", Select).value)
        if provider == "ollama":
            value = self.query_one("#ollama-model", Select).value
            model = value if isinstance(value, str) else ""
            if not model:
                raise ValueError("Choose an Ollama model first.")
        else:
            model = self.query_one("#model-name", Input).value
        return config_from_choices(provider, model, self.query_one("#workspace", Input).value)

    @on(Button.Pressed, "#save")
    def save(self) -> None:
        status = self.query_one("#setup-status", Static)
        try:
            config = self.chosen_config()
            Path(str(config.workspace)).mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as exc:
            status.update(f"[red]{escape(str(exc))}[/]")
            return
        if config.provider == "none":
            self.finish(config)
            return
        status.update(f"Testing {escape(config.model)} ...")
        self.query_one("#save", Button).disabled = True
        self.test_model(config)

    @work(thread=True, exclusive=True, group="setup-test")
    def test_model(self, config: AgentConfig) -> None:
        app = self.app
        provider = getattr(app, "provider_override", None)
        trial = AgentConfig.from_dict({**config.to_dict(), "tools": [], "mode": "llm"})
        try:
            result = Agent(config=trial, provider=provider).run_detailed(TEST_PROMPT)
            error = None if result.ok else result.output
        except (LLMError, ValueError) as exc:
            error = str(exc)
        app.call_from_thread(self.tested, config, error)

    def tested(self, config: AgentConfig, error: Optional[str]) -> None:
        self.query_one("#save", Button).disabled = False
        if error:
            message = f"[red]The model did not answer: {escape(error)}[/]"
            self.query_one("#setup-status", Static).update(message)
            return
        self.finish(config)

    def finish(self, config: AgentConfig) -> None:
        save_user_config(config)
        self.dismiss(config)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        if self.can_cancel:
            self.dismiss(None)


# -- the app ---------------------------------------------------------------------------


class TeiForgeApp(App[None]):
    """Chat with an agent, watch its steps, and browse its workspace."""

    TITLE = "TeiForge"
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
        Binding("escape", "stop", "Stop waiting"),
        Binding("f2", "setup", "Setup"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        provider: Optional[LLMProvider] = None,
        memory_path: Optional[Path] = None,
    ) -> None:
        """``config`` skips the saved settings; ``provider`` replaces the model (for tests)."""
        super().__init__()
        self.initial_config = config
        self.provider_override = provider
        self.memory_path = memory_path or chat_path()
        self.agent: Optional[Agent] = None
        self.generation = 0
        self.busy = False
        self.allowed_for_session: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="chat-pane"):
                yield VerticalScroll(id="messages")
                yield Static("", id="status")
                yield Input(placeholder="Ask TeiForge...  (/help for commands)", id="prompt", disabled=True)
            with Vertical(id="side-pane"):
                yield Static("Steps", classes="pane-title")
                yield RichLog(id="steps", markup=True, wrap=True)
                yield Static("Workspace", classes="pane-title", id="tree-title")
                yield Vertical(id="tree-box")
        yield Footer()

    def on_mount(self) -> None:
        config = self.initial_config or load_user_config() or self.config_from_environment()
        if config is None:
            self.push_screen(SetupScreen(), callback=self.apply_config)
        else:
            self.apply_config(config)

    def config_from_environment(self) -> Optional[AgentConfig]:
        """Use the TEIFORGE_* / API key variables when they already configure a model."""
        config = AgentConfig(name=CHAT_AGENT_NAME, tools=default_tools(), temperature=0)
        try:
            found = Agent(config=AgentConfig(name="probe")).provider
        except LLMError:
            return None
        return config if found is not None else None

    # -- configuration -------------------------------------------------------------

    def apply_config(self, config: Optional[AgentConfig]) -> None:
        if config is None:
            if self.agent is None:
                self.exit()
            return
        if not config.tools:
            config.tools = default_tools()
        agent = Agent(config=config, provider=self.provider_override, approver=self.approve)
        if config.provider == "none":
            agent.config.mode = "router"
        if self.agent is not None:
            agent.memory = self.agent.memory
        elif self.memory_path.is_file():
            try:
                agent.memory = Memory.load(self.memory_path, max_entries=config.memory_size)
            except (OSError, ValueError, KeyError):
                self.notify("The saved conversation could not be read; starting fresh.", severity="warning")
        self.agent = agent
        self.allowed_for_session.clear()
        self.sub_title = self.mode_label()
        self.mount_tree(agent.workspace.root)
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        loaded = len(agent.memory)
        self.set_status(f"Ready. {loaded} earlier messages loaded." if loaded else "Ready.")

    def mode_label(self) -> str:
        assert self.agent is not None
        agent = self.agent
        if agent.config.mode == "router":
            model = "keyword router (no model)"
        else:
            try:
                provider = agent.provider
            except LLMError as exc:
                return f"model unavailable: {exc}"
            model = f"{agent.model} ({getattr(provider, 'name', 'custom')})" if provider else "no model"
        return f"{model} · workspace: {agent.workspace.root.name or agent.workspace.root}"

    def mount_tree(self, root: Path) -> None:
        box = self.query_one("#tree-box", Vertical)
        box.remove_children()
        box.mount(DirectoryTree(str(root), id="tree"))
        self.query_one("#tree-title", Static).update(f"Workspace: {escape(root.name or str(root))}")

    # -- helpers ---------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def add_message(self, kind: str, text: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        if kind == "you":
            widget: Widget = Static(f"[b]You[/b]  {escape(text)}", classes="you")
        elif kind == "answer":
            widget = Markdown(text, classes="answer")
        elif kind == "failed":
            widget = Static(escape(text), classes="failed")
        else:
            widget = Markdown(text, classes="note")
        messages.mount(widget)
        messages.scroll_end(animate=False)

    def save_memory(self) -> None:
        if self.agent is None:
            return
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent.memory.save(self.memory_path)
        except OSError as exc:
            self.notify(f"Could not save the conversation: {exc}", severity="error")

    # -- input ------------------------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.agent is None:
            return
        if text.lower() in {"exit", "quit"} or text.startswith("/"):
            self.run_command(text)
            return
        if self.busy:
            self.notify("Still working on the last question. Press Esc to stop waiting.")
            return
        self.add_message("you", text)
        self.busy = True
        self.generation += 1
        self.set_status("Thinking...  (Esc to stop waiting)")
        self.run_task(text, self.generation)

    def run_command(self, text: str) -> None:
        assert self.agent is not None
        command = text.split()[0].lower().lstrip("/")
        if command in {"exit", "quit"}:
            self.action_quit_app()
        elif command == "help":
            self.add_message("note", HELP_TEXT)
        elif command == "tools":
            self.add_message("note", "```\n" + self.agent.get_tools_description() + "\n```")
        elif command == "memory":
            self.add_message("note", "```\n" + self.agent.get_memory_summary() + "```")
        elif command == "clear":
            self.action_clear()
        elif command in {"setup", "model"}:
            self.action_setup()
        else:
            self.add_message("failed", f"Unknown command {text.split()[0]}. Type /help.")

    # -- running the agent ------------------------------------------------------------

    @work(thread=True, group="agent")
    def run_task(self, text: str, generation: int) -> None:
        assert self.agent is not None
        agent = self.agent

        def forward(event: Event) -> None:
            self.call_from_thread(self.add_step, event, generation)

        try:
            result = agent.run_detailed(text, on_event=forward)
        except Exception as exc:  # noqa: BLE001 - show any failure instead of crashing the UI
            result = RunResult(task=text, output=f"Error: {type(exc).__name__}: {exc}", ok=False, mode="llm")
        try:
            self.call_from_thread(self.finish_task, result, generation)
        except RuntimeError:  # pragma: no cover - the app closed while the model was answering
            pass

    def add_step(self, event: Event, generation: int) -> None:
        line = format_step(event)
        if line is not None:
            prefix = "" if generation == self.generation else "[dim](stopped) [/]"
            self.query_one("#steps", RichLog).write(prefix + line)
        if event.type == "tool_result" and event.data.get("name") == "file_writer" and event.data.get("ok"):
            self.query_one("#tree", DirectoryTree).reload()

    def finish_task(self, result: RunResult, generation: int) -> None:
        if generation != self.generation:
            self.forget(result)
            return
        self.busy = False
        self.add_message("answer" if result.ok else "failed", result.output)
        self.set_status("Ready.")
        self.save_memory()

    def forget(self, result: RunResult) -> None:
        """Drop the memory entries of a run whose answer the user stopped waiting for."""
        assert self.agent is not None
        history = self.agent.memory.history
        for index in range(len(history) - 1, 0, -1):
            if history[index - 1].content == result.task and history[index].content == result.output:
                del history[index - 1 : index + 1]
                break

    def approve(self, call: ToolCall) -> bool:
        """Called from the agent's thread: ask in a dialog before tools that change things."""
        if not needs_approval(call) or call.name in self.allowed_for_session:
            return True
        answer: Dict[str, Any] = {}
        done = threading.Event()

        def record(choice: Optional[str]) -> None:
            answer["choice"] = choice
            done.set()

        self.call_from_thread(self.push_screen, ApprovalScreen(call), record)
        while not done.wait(0.2):
            if not self.is_running:  # pragma: no cover - the app closed while asking
                return False
        if answer.get("choice") == "session":
            self.allowed_for_session.add(call.name)
            return True
        return answer.get("choice") == "allow"

    # -- actions ------------------------------------------------------------------------

    def action_stop(self) -> None:
        if not self.busy:
            return
        self.generation += 1
        self.busy = False
        self.set_status("Stopped waiting. If the model still answers, the answer is dropped.")

    def action_clear(self) -> None:
        if self.agent is None:
            return
        self.agent.memory.clear()
        self.query_one("#messages", VerticalScroll).remove_children()
        self.query_one("#steps", RichLog).clear()
        self.save_memory()
        self.set_status("Conversation cleared.")

    def action_setup(self) -> None:
        current = self.agent.config if self.agent else None
        self.push_screen(SetupScreen(current, can_cancel=self.agent is not None), callback=self.apply_config)

    def action_quit_app(self) -> None:
        self.save_memory()
        self.exit()


def run_tui() -> int:
    """Start the terminal UI and return an exit code."""
    TeiForgeApp().run()
    return 0


__all__ = ["ApprovalScreen", "SetupScreen", "TeiForgeApp", "config_from_choices", "config_path", "run_tui"]
