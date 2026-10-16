"""The agent: an LLM tool-calling loop with a keyword router as fallback."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .config import AgentConfig
from .events import Event, EventHandler, EventType, RunResult
from .llm import (
    MODEL_ENV_VAR,
    PROVIDER_ENV_VAR,
    LLMError,
    LLMProvider,
    Message,
    parse_text_tool_calls,
    provider_from_config,
)
from .memory import Memory
from .tools import Tool, ToolError, ToolResult, builtin_tools, parse_write_command, tool
from .workspace import Workspace

logger = logging.getLogger(__name__)

#: Opening line of the default system prompt; ``{name}`` is the agent name.
DEFAULT_SYSTEM_PROMPT = "You are {name}, an assistant built with TeiForge."

_TOOLS_GUIDANCE = (
    "Available tools: {tools}. Use them when they help you answer accurately, and call a tool "
    "only through the tool-calling interface, never by writing the call as text. Prefer one tool "
    "call at a time and check each result. If a tool fails, explain what went wrong instead of "
    "guessing."
)
_NO_TOOLS_GUIDANCE = (
    "No tools are available in this session. Answer from what you know, and say so plainly "
    "when a task needs information or actions you cannot access."
)
_WEB_SEARCH_GUIDANCE = "Include source links for information you found with web search."
_ANSWER_GUIDANCE = "Keep the final answer concise."


def build_system_prompt(name: str, tool_names: Sequence[str]) -> str:
    """Build the default system prompt so it only describes tools the agent really has."""
    parts = [DEFAULT_SYSTEM_PROMPT.format(name=name)]
    if tool_names:
        parts.append(_TOOLS_GUIDANCE.format(tools=", ".join(tool_names)))
    else:
        parts.append(_NO_TOOLS_GUIDANCE)
    parts.append(_ANSWER_GUIDANCE)
    if "web_search" in tool_names:
        parts.append(_WEB_SEARCH_GUIDANCE)
    return " ".join(parts)


QUICKSTART_URL = "https://github.com/Tdat10052499/teiforge#quickstart-free-on-your-own-machine-5-minutes"

NO_LLM_WARNING = (
    "No LLM provider is configured, so TeiForge is using its keyword router. Set "
    "TEIFORGE_PROVIDER, TEIFORGE_BASE_URL and TEIFORGE_MODEL (for Ollama) or OPENAI_API_KEY / "
    "ANTHROPIC_API_KEY to use a model, or pass AgentConfig(mode='router') to silence this warning. "
    "See " + QUICKSTART_URL
)

#: An answer that blames a tool, e.g. "the tool call failed to extract relevant information".
_FAILURE_CLAIM = re.compile(
    r"\b(?:tools?|(?:tool |function )?calls?|\w+_(?:reader|writer|search|runner))\b[^.!?\n]{0,60}?"
    r"\b(?:fail(?:ed|s|ure)?|could(?:n't| not)|unable|not able|did(?:n't| not) work)\b"
    r"|\b(?:fail(?:ed|s)? to|unable to|could(?:n't| not)|not able to)\s+(?:read|extract|find|access|open)\b",
    flags=re.IGNORECASE,
)

SELF_CHECK_PROMPT = (
    "Your answer says that a tool failed, but every tool call above succeeded and its result is "
    "shown above. Read those results again and answer the original task using them: {task}"
)

SUMMARY_PROMPT = "You are a concise assistant that summarizes the content of web search results."

#: Router rules, checked in order. Keywords match whole words only.
ROUTER_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("web_search", ("search", "find", "lookup", "look up", "latest", "news", "research")),
    ("file_writer", ("write", "save", "create", "store")),
    ("file_reader", ("read", "open", "file", "load", "summarize", "show")),
    ("command_runner", ("run", "execute", "command", "bash", "shell")),
)

_EXPLICIT_PREFIX = re.compile(r"\b(read|write|search)\s*:", flags=re.IGNORECASE)
_PREFIX_TARGETS = {"read": "file_reader", "write": "file_writer", "search": "web_search"}
# A path-like token: optional drive, then a name ending in an extension of 2-10 characters
# that starts with a letter (so "v0.3" and "e.g." are not mistaken for files).
_PATH_TOKEN = re.compile(r"""(?:[A-Za-z]:[\\/])?[\w.~\\/-]*[\w-]\.[A-Za-z][A-Za-z0-9]{1,9}\b""")

ToolLike = Union[Tool, Callable[..., Any]]


def _keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(word).replace(r"\ ", r"\s+") for word in keywords)
    return re.compile(rf"\b(?:{alternatives})\b", flags=re.IGNORECASE)


_ROUTER_PATTERNS = [(name, _keyword_pattern(words)) for name, words in ROUTER_RULES]


def extract_path(text: str) -> Optional[str]:
    """Return the first token in ``text`` that looks like a file path, if any."""
    for match in _PATH_TOKEN.finditer(text):
        start = match.start()
        if text[max(0, start - 3) : start] == "://" or "://" in match.group(0):
            continue
        prefix = text[:start]
        if re.search(r"https?://\S*$", prefix):
            continue
        return match.group(0).strip("'\"")
    return None


class Agent:
    """An agent that answers tasks with tools and keeps a conversation memory.

    With an LLM provider configured it runs a tool-calling loop: the model picks
    tools, sees their results and writes the final answer. Without one it falls
    back to a deterministic keyword router.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        *,
        provider: Optional[LLMProvider] = None,
        workspace: Union[str, os.PathLike[str], Workspace, None] = None,
        tools: Optional[Iterable[ToolLike]] = None,
    ) -> None:
        if config is not None and name is not None:
            raise ValueError("Pass either name/model or config, not both.")
        if config is None:
            if name is None:
                raise ValueError("Agent name or config must be provided.")
            config = AgentConfig(name=name, model=model or os.getenv("TEIFORGE_MODEL") or "gpt-4o-mini")
        elif model is not None:
            raise ValueError("Pass the model through config when using AgentConfig.")

        self.config = config
        self.name = config.name
        self.tools: Dict[str, Tool] = {}
        self.memory = Memory(max_entries=config.memory_size)
        self.task_count = 0
        self.enable_llm: bool = True

        workspace = workspace if workspace is not None else config.workspace
        if isinstance(workspace, Workspace):
            self.workspace = workspace
            explicit_workspace = True
        else:
            self.workspace = Workspace(workspace)
            explicit_workspace = workspace is not None

        available = builtin_tools(self.workspace if explicit_workspace else None)
        for tool_name in config.tools:
            if tool_name not in available:
                raise ValueError(f"Unknown built-in tool in AgentConfig: {tool_name}")
            self.add_tool(available[tool_name])
        for extra in tools or ():
            self.add_tool(extra)

        self._provider: Optional[LLMProvider] = provider
        self._provider_resolved = provider is not None
        self._provider_explicit = provider is not None
        self._fell_back_to_router = False
        self._warned_no_llm = False

    # -- tools ---------------------------------------------------------------------

    def add_tool(self, new_tool: ToolLike) -> Tool:
        """Register a :class:`Tool`, or a typed function that is wrapped with ``@tool``."""
        if isinstance(new_tool, Tool):
            registered = new_tool
        elif callable(new_tool):
            registered = tool(new_tool)
        else:
            raise TypeError("add_tool expects a Tool or a callable.")
        self.tools[registered.name] = registered
        return registered

    def remove_tool(self, tool_name: str) -> bool:
        return self.tools.pop(tool_name, None) is not None

    def get_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."
        lines = ["Available tools:"]
        lines.extend(f"  - {name}: {item.description}" for name, item in self.tools.items())
        return "\n".join(lines)

    # -- provider ------------------------------------------------------------------

    @property
    def provider(self) -> Optional[LLMProvider]:
        """The LLM provider, resolved from the config and environment on first use."""
        if not self._provider_resolved:
            self._provider = provider_from_config(self.config)
            self._provider_resolved = True
        return self._provider

    @provider.setter
    def provider(self, value: Optional[LLMProvider]) -> None:
        self._provider = value
        self._provider_resolved = True
        self._provider_explicit = True

    @property
    def model(self) -> str:
        """The model this agent uses.

        When an LLM provider is configured this is the provider's model, which reflects the
        ``TEIFORGE_MODEL`` environment variable; otherwise it is the configured model.
        """
        provider: Optional[LLMProvider] = None
        if self._provider_resolved:
            provider = self._provider
        elif self.enable_llm and self.config.mode != "router":
            try:
                provider = self.provider
            except LLMError:
                provider = None
        if provider is not None:
            return str(getattr(provider, "model", self.config.model))
        return os.getenv(MODEL_ENV_VAR) or self.config.model

    @model.setter
    def model(self, value: str) -> None:
        self.config.model = value
        if not self._provider_explicit:
            self._provider = None
            self._provider_resolved = False

    # -- running -------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Run a task and return the final text. See :meth:`run_detailed` for events."""
        return self.run_detailed(user_input).output

    def run_detailed(self, user_input: str, on_event: Optional[EventHandler] = None) -> RunResult:
        """Run a task and return the output together with every event of the run."""
        events: List[Event] = []

        def emit(event_type: EventType, **data: Any) -> None:
            event = Event(type=event_type, data=data)
            events.append(event)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001 - a broken handler must not stop the run
                    logger.exception("Event handler failed for %s event.", event_type)

        emit("task_start", task=user_input, agent=self.name)
        steps = 0
        mode = "router"
        try:
            mode = self._select_mode()
            if mode == "llm":
                output, ok, steps = self._run_llm(user_input, emit)
            else:
                output, ok = self._run_router(user_input, emit)
                steps = 1
        except LLMError as exc:
            output, ok = f"Error: {exc}", False
            emit("error", message=str(exc))

        emit("final", output=output, ok=ok)
        self._record_task(user_input, output)
        return RunResult(task=user_input, output=output, ok=ok, mode=mode, steps=steps, events=events)

    def _select_mode(self) -> str:
        self._fell_back_to_router = False
        if self.config.mode == "router" or not self.enable_llm:
            return "router"
        provider = self.provider
        if self.config.mode == "llm" and provider is None:
            raise LLMError(
                "mode='llm' needs an LLM provider. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
                "or pass provider=..."
            )
        if provider is not None:
            return "llm"
        if not self._llm_disabled_on_purpose():
            self._fell_back_to_router = True
            if not self._warned_no_llm:
                logger.warning(NO_LLM_WARNING)
                self._warned_no_llm = True
        return "router"

    def _llm_disabled_on_purpose(self) -> bool:
        choice = self.config.provider or os.getenv(PROVIDER_ENV_VAR) or ""
        return choice.strip().lower() == "none"

    def _record_task(self, user_input: str, response: str) -> None:
        self.task_count += 1
        self.memory.add(user_input, role="user")
        self.memory.add(response, role="assistant")

    # -- LLM loop ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return self.config.system_prompt or build_system_prompt(self.name, list(self.tools))

    def _run_llm(self, task: str, emit: Callable[..., None]) -> Tuple[str, bool, int]:
        provider = self.provider
        assert provider is not None
        messages: List[Message] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.memory.to_messages(limit=20))
        messages.append({"role": "user", "content": task})
        tool_list = list(self.tools.values())
        tool_results: List[bool] = []
        checked = False

        for step in range(1, self.config.max_steps + 1):
            response = provider.complete(messages, tool_list)
            emit(
                "llm_response",
                step=step,
                content=response.content,
                tool_calls=[call.name for call in response.tool_calls],
            )
            calls = response.tool_calls
            content = response.content or ""
            recovered = False
            if not calls and self.config.recover_text_tool_calls:
                calls = parse_text_tool_calls(content, list(self.tools))
                if calls:
                    # The text was the call itself; keeping it would show the model its own mistake.
                    recovered, content = True, ""
            if not calls:
                answer = content.strip()
                if not answer:
                    return "Error: The model returned an empty answer.", False, step
                if (
                    self.config.self_check
                    and not checked
                    and step < self.config.max_steps
                    and tool_results
                    and all(tool_results)
                    and _FAILURE_CLAIM.search(answer)
                ):
                    checked = True
                    emit("self_check", step=step, answer=answer)
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": SELF_CHECK_PROMPT.format(task=task)})
                    continue
                return answer, True, step

            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for call in calls:
                extra = {"recovered": True} if recovered else {}
                emit("tool_call", step=step, id=call.id, name=call.name, arguments=call.arguments, **extra)
                selected = self.tools.get(call.name)
                if selected is None:
                    result = ToolResult(ok=False, error=f"Unknown tool '{call.name}'.")
                else:
                    result = selected.run(call.arguments)
                emit("tool_result", step=step, id=call.id, name=call.name, ok=result.ok, output=result.text)
                tool_results.append(result.ok)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.output if result.ok else f"Error: {result.error}",
                        "is_error": not result.ok,
                    }
                )

        return (
            f"Error: Stopped after {self.config.max_steps} steps without a final answer.",
            False,
            self.config.max_steps,
        )

    # -- keyword router --------------------------------------------------------------

    def route(self, user_input: str) -> Optional[str]:
        """Return the name of the tool the keyword router would use, if any."""
        explicit = _EXPLICIT_PREFIX.search(user_input)
        if explicit:
            target = _PREFIX_TARGETS[explicit.group(1).lower()]
            if target in self.tools:
                return target
        for tool_name, pattern in _ROUTER_PATTERNS:
            if tool_name in self.tools and pattern.search(user_input):
                return tool_name
        return None

    def _run_router(self, user_input: str, emit: Callable[..., None]) -> Tuple[str, bool]:
        tool_name = self.route(user_input)
        if tool_name is None:
            return self._default_response(user_input), False

        selected = self.tools[tool_name]
        try:
            arguments = self._router_arguments(tool_name, selected, user_input)
        except ToolError as exc:
            emit("error", message=str(exc), tool=tool_name)
            return f"Error: {exc}", False

        emit("tool_call", step=1, id="router", name=tool_name, arguments=arguments)
        result = selected.run(arguments)
        emit("tool_result", step=1, id="router", name=tool_name, ok=result.ok, output=result.text)
        if not result.ok:
            return f"Error: {result.error}", False

        output = result.output
        if tool_name == "web_search":
            summary = self._summarize_text(output)
            if summary:
                output = f"{summary}\n\nFull results:\n{output}"
        return self._format_response(tool_name, output), True

    def _router_arguments(self, tool_name: str, selected: Tool, text: str) -> Union[str, Dict[str, Any]]:
        explicit = _EXPLICIT_PREFIX.search(text)
        if explicit and _PREFIX_TARGETS[explicit.group(1).lower()] != tool_name:
            explicit = None
        if not selected.structured:
            if tool_name in {"file_reader", "file_writer"} and explicit is None:
                prefix = "read:" if tool_name == "file_reader" else "write:"
                return prefix + text
            return text[explicit.start() :] if explicit else text

        if tool_name == "web_search":
            return {"query": text[explicit.end() :] if explicit else text}
        if tool_name == "file_reader":
            path: Optional[str]
            if explicit:
                rest = text[explicit.end() :].strip()
                # "read: notes.txt and summarize" -> "notes.txt"; "read:My File.txt" stays whole.
                path = rest
                if not self._exists_in_workspace(rest):
                    path = extract_path(rest) or rest
            else:
                path = extract_path(text)
            if not path:
                raise ToolError("Could not find a file path in the task. Use read:path/to/file.txt")
            return {"path": path}
        if tool_name == "file_writer":
            if explicit is None or explicit.group(1).lower() != "write":
                raise ToolError(
                    "Use write:path/to/file.txt:content, or configure an LLM provider "
                    "for natural-language file writing."
                )
            path, content = parse_write_command(text[explicit.end() :])
            return {"path": path, "content": content}
        if tool_name == "command_runner":
            return {"command": text}
        required = selected.parameters.get("required", [])
        if len(required) == 1:
            return {required[0]: text}
        raise ToolError(f"The router cannot build arguments for tool '{tool_name}'.")

    def _exists_in_workspace(self, path: str) -> bool:
        try:
            return self.workspace.resolve(path).exists()
        except ValueError:
            return False

    def _format_response(self, tool_name: str, tool_output: str) -> str:
        return f"[Using {tool_name}]\n\n{tool_output}"

    def _default_response(self, user_input: str) -> str:
        router_tools = {name for name, _ in ROUTER_RULES}
        custom = [name for name in self.tools if name not in router_tools]
        parts = [f"Agent '{self.name}' received task: \"{user_input}\""]
        if self._fell_back_to_router:
            parts.append(
                "No LLM provider is configured, so TeiForge used its keyword router. The router "
                "only handles simple tasks such as 'search <query>', 'read <file>', "
                "'write:<file>:<text>' and 'run <command>'."
            )
        else:
            parts.append(
                "The keyword router found no matching tool. It only handles simple tasks such as "
                "'search <query>', 'read <file>', 'write:<file>:<text>' and 'run <command>'."
            )
        if custom:
            parts.append(
                f"Custom tools ({', '.join(custom)}) are not used by the keyword router; "
                "only an LLM provider can call them."
            )
        if self._fell_back_to_router:
            parts.append(
                "To use a model, set TEIFORGE_PROVIDER, TEIFORGE_BASE_URL and TEIFORGE_MODEL (for "
                "Ollama) or OPENAI_API_KEY / ANTHROPIC_API_KEY in the same terminal, then run again. "
                f"See {QUICKSTART_URL}"
            )
        parts.append(self.get_tools_description())
        return "\n\n".join(parts)

    # -- search and save -------------------------------------------------------------

    def _summarize_text(self, text: str, max_bullets: int = 5) -> Optional[str]:
        """Summarize ``text`` with the LLM; return ``None`` when no LLM is available."""
        if not self.enable_llm or not text or not text.strip():
            return None
        try:
            provider = self.provider
        except LLMError:
            logger.exception("LLM provider is misconfigured; continuing without a summary.")
            return None
        if provider is None:
            return None
        messages: List[Message] = [
            {"role": "system", "content": self.config.system_prompt or SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (
                    "Summarize the following content into a short 1-line summary and then up to "
                    f"{max_bullets} bullet points. Keep it concise and include any source URLs "
                    "if present.\n\nContent:\n" + text
                ),
            },
        ]
        try:
            return provider.complete(messages).content
        except LLMError:
            logger.exception("LLM summarization failed; continuing without a summary.")
            return None

    def run_search_and_save(self, query: str, out_path: str) -> str:
        """Search the web, optionally summarize, and save summary and raw results to a file.

        Returns a confirmation message or an explicit ``Error:`` string.
        """
        if not query or not query.strip():
            response = "Error: A search query is required."
        elif not out_path or not out_path.strip():
            response = "Error: An output file path is required."
        elif "web_search" not in self.tools:
            response = "Error: No web_search tool is available for this task."
        else:
            result = self.tools["web_search"].run(query)
            if not result.ok:
                response = f"Error: Search failed: {result.error}"
            elif not result.output.strip():
                response = "Error: Search returned an empty result."
            else:
                raw_result = result.output
                summary = self._summarize_text(raw_result)
                if summary and summary.strip():
                    content = f"Summary:\n{summary}\n\nOriginal search result:\n{raw_result}\n"
                else:
                    content = f"Original search result:\n{raw_result}\n"
                try:
                    with open(out_path, "w", encoding="utf-8") as handle:
                        handle.write(content)
                except OSError as exc:
                    response = (
                        f"Error: Failed to save search results to {out_path}: {type(exc).__name__}: {exc}"
                    )
                else:
                    response = f"Saved search results to {out_path}"

        self._record_task(query, response)
        return response

    # -- reporting -------------------------------------------------------------------

    def get_memory_summary(self) -> str:
        return (
            "Agent Memory Summary\n"
            "==================\n"
            f"Agent: {self.name}\n"
            f"Model: {self.model}\n"
            f"Tasks run: {self.task_count}\n"
            f"Memory entries: {len(self.memory)}\n"
            f"Tools available: {len(self.tools)}\n"
        )

    def __str__(self) -> str:
        return f"Agent(name='{self.name}', model='{self.model}', tools={len(self.tools)})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["Agent", "DEFAULT_SYSTEM_PROMPT", "ROUTER_RULES", "build_system_prompt", "extract_path"]
