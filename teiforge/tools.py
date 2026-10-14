"""Tools that agents can call.

A :class:`Tool` wraps a Python callable together with a JSON Schema describing its
parameters, so the same tool can be called directly, by the keyword router, or by
an LLM through function calling. Use the :func:`tool` decorator to build one from
type hints.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import sys
import types
import typing
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import requests

from .workspace import Workspace, WorkspaceError

JSONSchema = Dict[str, Any]

#: Output prefixes that mark a failure for legacy ``str -> str`` tools.
#: Structured tools should raise :class:`ToolError` instead.
LEGACY_ERROR_PREFIXES: Tuple[str, ...] = ("Error:", "Tool error:", "Web search failed:")


class ToolError(Exception):
    """Raise from a tool to report a user-facing failure."""


@dataclass
class ToolResult:
    """The outcome of a tool call."""

    ok: bool
    output: str = ""
    error: Optional[str] = None

    @property
    def text(self) -> str:
        """The output on success, or the error message on failure."""
        return self.output if self.ok else (self.error or "Unknown error")

    def __str__(self) -> str:
        return self.output if self.ok else f"Error: {self.error}"


_LEGACY_PARAMETERS: JSONSchema = {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "Free-form input for the tool."}},
    "required": ["input"],
}


class Tool:
    """A named, described callable that an agent can use.

    Two styles are supported:

    * **Structured** tools pass ``parameters`` (a JSON Schema object). They are
      called with keyword arguments and report failures by raising
      :class:`ToolError`. The :func:`tool` decorator builds these.
    * **Legacy** tools (no ``parameters``) take a single string and return a
      string. Output starting with one of :data:`LEGACY_ERROR_PREFIXES` is
      treated as a failure.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[JSONSchema] = None,
    ) -> None:
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Tool name must be 1-64 characters of letters, digits, '_' or '-'.")
        self.name = name
        self.description = description
        self.func = func
        self.structured = parameters is not None
        self.parameters: JSONSchema = parameters if parameters is not None else _LEGACY_PARAMETERS

    # -- calling -----------------------------------------------------------------

    def run(self, args: Union[str, Mapping[str, Any], None] = None, **kwargs: Any) -> ToolResult:
        """Call the tool and return a :class:`ToolResult`; never raises."""
        try:
            call_args = self._prepare_args(args, kwargs)
            if self.structured:
                raw = self.func(**call_args)
            else:
                raw = self.func(call_args["input"])
        except (ToolError, WorkspaceError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return ToolResult(ok=False, error=f"Tool error: {type(exc).__name__}: {exc}")

        if isinstance(raw, ToolResult):
            return raw
        text = "" if raw is None else str(raw)
        if not self.structured and text.startswith(LEGACY_ERROR_PREFIXES):
            return ToolResult(ok=False, error=text)
        return ToolResult(ok=True, output=text)

    __call__ = run

    def _prepare_args(
        self, args: Union[str, Mapping[str, Any], None], kwargs: Mapping[str, Any]
    ) -> Dict[str, Any]:
        properties: Dict[str, Any] = self.parameters.get("properties", {})
        required: List[str] = list(self.parameters.get("required", []))

        if isinstance(args, str):
            if not self.structured:
                return {"input": args}
            if len(required) != 1:
                raise ToolError(f"Tool '{self.name}' needs structured arguments: {', '.join(properties)}.")
            call_args: Dict[str, Any] = {required[0]: args}
        elif args is None:
            call_args = {}
        else:
            call_args = dict(args)
        call_args.update(kwargs)

        if not self.structured:
            value = call_args.get("input", "")
            return {"input": value if isinstance(value, str) else json.dumps(value)}

        unknown = sorted(set(call_args) - set(properties))
        if unknown:
            raise ToolError(f"Tool '{self.name}' got unknown argument(s): {', '.join(unknown)}.")
        missing = [name for name in required if name not in call_args]
        if missing:
            raise ToolError(f"Tool '{self.name}' is missing argument(s): {', '.join(missing)}.")
        return call_args

    # -- schemas -----------------------------------------------------------------

    def to_openai_schema(self) -> JSONSchema:
        """Return the tool in OpenAI Chat Completions ``tools`` format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> JSONSchema:
        """Return the tool in Anthropic Messages ``tools`` format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


# -- @tool decorator ---------------------------------------------------------------

_SIMPLE_TYPES: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _schema_for(annotation: Any) -> JSONSchema:
    if annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}

    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)

    union_types: Tuple[Any, ...] = (Union,)
    if hasattr(types, "UnionType"):
        union_types = (Union, types.UnionType)
    if origin in union_types:
        non_null = [arg for arg in arguments if arg is not type(None)]
        if len(non_null) == 1:
            return _schema_for(non_null[0])
        return {"anyOf": [_schema_for(arg) for arg in non_null]}
    if origin is Literal:
        return {"enum": list(arguments)}
    if origin in (list, List):
        schema: JSONSchema = {"type": "array"}
        if arguments:
            schema["items"] = _schema_for(arguments[0])
        return schema
    if origin in (dict, Dict):
        return {"type": "object"}
    raise TypeError(f"Unsupported parameter type for @tool: {annotation!r}")


def _parse_docstring(doc: str) -> Tuple[str, Dict[str, str]]:
    """Return the summary and per-argument descriptions from a Google-style docstring."""
    doc = inspect.cleandoc(doc or "")
    summary_lines: List[str] = []
    arg_docs: Dict[str, str] = {}
    section = "summary"
    current: Optional[str] = None
    arg_indent: Optional[int] = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            section = "args"
            continue
        if stripped.endswith(":") and stripped[:-1].lower() in {"returns", "raises", "example", "examples"}:
            section = "other"
            continue
        if section == "summary":
            if not stripped and summary_lines:
                section = "other"
                continue
            if stripped:
                summary_lines.append(stripped)
        elif section == "args" and stripped:
            indent = len(line) - len(line.lstrip())
            match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", stripped)
            if match and (arg_indent is None or indent <= arg_indent):
                arg_indent = indent
                current = match.group(1)
                arg_docs[current] = match.group(2)
            elif current:
                arg_docs[current] = f"{arg_docs[current]} {stripped}".strip()
    return " ".join(summary_lines), arg_docs


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Turn a typed function into a :class:`Tool`.

    The JSON Schema is built from the type hints, the description from the first
    paragraph of the docstring, and parameter descriptions from an ``Args:``
    section::

        @tool
        def add(a: int, b: int = 1) -> str:
            \"\"\"Add two numbers.

            Args:
                a: The first number.
                b: The second number.
            \"\"\"
            return str(a + b)
    """

    def build(target: Callable[..., Any]) -> Tool:
        hints = typing.get_type_hints(target)
        summary, arg_docs = _parse_docstring(target.__doc__ or "")
        properties: Dict[str, JSONSchema] = {}
        required: List[str] = []
        for parameter in inspect.signature(target).parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("@tool functions cannot use *args or **kwargs.")
            schema = _schema_for(hints.get(parameter.name, str))
            if parameter.name in arg_docs:
                schema["description"] = arg_docs[parameter.name]
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
            elif parameter.default is not None:
                schema["default"] = parameter.default
            properties[parameter.name] = schema

        parameters: JSONSchema = {"type": "object", "properties": properties, "required": required}
        return Tool(
            name=name or target.__name__,
            description=description or summary or target.__name__,
            func=target,
            parameters=parameters,
        )

    if func is not None:
        return build(func)
    return build


# -- web search --------------------------------------------------------------------


class _DuckDuckGoResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._capture_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = attributes.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": self._destination_url(attributes.get("href", ""))})
            self._capture_depth = 1
        elif self._capture_depth:
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture_depth:
            self._capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capture_depth and self.results:
            self.results[-1]["title"] += data

    @staticmethod
    def _destination_url(href: str) -> str:
        parsed_url = urlparse(href)
        if parsed_url.path.endswith("/l/"):
            destination = parse_qs(parsed_url.query).get("uddg")
            if destination:
                return destination[0]
        return href


_QUERY_PREFIX = re.compile(
    r"^(?:please\s+)?(?:search\s*:|search\s+for|search|look\s+up|lookup|find|research)(?:\s+|$)",
    flags=re.IGNORECASE,
)


def normalize_search_query(query: str) -> str:
    """Strip conversational prefixes such as 'search for' from a query."""
    return _QUERY_PREFIX.sub("", query.strip(), count=1).strip()


def _flatten_related_topics(topics: List[Dict[str, Any]]) -> Any:
    for topic in topics:
        if "Topics" in topic:
            yield from _flatten_related_topics(topic["Topics"])
        else:
            yield topic


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo and return titles and links.

    Args:
        query: What to search for.
        max_results: Maximum number of results to return (1-10).
    """
    normalized = normalize_search_query(query or "")
    if not normalized:
        raise ToolError("Web search requires a non-empty query.")
    max_results = max(1, min(int(max_results), 10))

    api_error: Optional[Exception] = None
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": normalized, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        abstract = payload.get("AbstractText", "").strip()
        if abstract:
            lines = [f"Search results for: '{normalized}'", "", abstract]
            if payload.get("AbstractURL"):
                lines.append(f"Source: {payload['AbstractURL']}")
            return "\n".join(lines)

        lines = [f"Search results for: '{normalized}'", ""]
        count = 0
        for topic in _flatten_related_topics(payload.get("RelatedTopics", [])):
            text = topic.get("Text", "").strip()
            first_url = topic.get("FirstURL", "")
            if text and first_url and not text.endswith(" Category"):
                lines.extend([f"- {text}", f"  Link: {first_url}"])
                count += 1
            if count >= max_results:
                break
        if count:
            return "\n".join(lines)
    except (requests.RequestException, ValueError) as exc:
        api_error = exc

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": normalized},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        parser = _DuckDuckGoResultsParser()
        parser.feed(response.text)
        results = [r for r in parser.results if r["title"].strip() and r["url"]][:max_results]
        if results:
            lines = [f"Search results for: '{normalized}'", ""]
            for result in results:
                lines.extend([f"- {result['title'].strip()}", f"  Link: {result['url']}"])
            return "\n".join(lines)
    except requests.RequestException as exc:
        if api_error:
            raise ToolError(
                "Web search failed: "
                f"Instant Answer API {type(api_error).__name__}: {api_error}; "
                f"HTML search {type(exc).__name__}: {exc}"
            ) from exc
        raise ToolError(f"Web search failed: {type(exc).__name__}: {exc}") from exc

    if api_error:
        raise ToolError(f"Web search failed: {type(api_error).__name__}: {api_error}")
    return f"No results found for '{normalized}'. Try a different search term."


# -- file tools --------------------------------------------------------------------

_WRITE_COMMAND = re.compile(r"^\s*((?:[A-Za-z]:[\\/])?[^:]*):(.*)$", flags=re.DOTALL)


def parse_write_command(payload: str) -> Tuple[str, str]:
    """Split ``path:content`` (the part after ``write:``), allowing Windows drive letters.

    ``C:\\notes\\a.txt:hello`` becomes ``("C:\\notes\\a.txt", "hello")``.
    """
    match = _WRITE_COMMAND.match(payload)
    if not match or not match.group(1).strip():
        raise ToolError("Usage: write:path/to/file.txt:content")
    return match.group(1).strip(), match.group(2)


def make_file_reader(workspace: Optional[Workspace] = None, max_chars: int = 100_000) -> Tool:
    """Create a ``file_reader`` tool confined to ``workspace``."""
    space = workspace or Workspace()

    @tool(name="file_reader")
    def read_file(path: str) -> str:
        """Read a UTF-8 text file inside the workspace.

        Args:
            path: File path, relative to the workspace root.
        """
        resolved = space.resolve(path)
        if not resolved.exists():
            raise ToolError(f"File not found: {path}")
        if resolved.is_dir():
            raise ToolError(f"'{path}' is a directory, not a file.")
        try:
            content = resolved.read_text(encoding="utf-8-sig")  # drops a UTF-8 BOM if present
        except UnicodeDecodeError as exc:
            raise ToolError(f"'{path}' is not a UTF-8 text file.") from exc
        note = ""
        if len(content) > max_chars:
            content = content[:max_chars]
            note = f"\n\n[Truncated to the first {max_chars} characters.]"
        return f"File contents of {space.relative(resolved)}:\n\n{content}{note}"

    return read_file  # type: ignore[no-any-return]


def make_file_writer(workspace: Optional[Workspace] = None) -> Tool:
    """Create a ``file_writer`` tool confined to ``workspace``."""
    space = workspace or Workspace()

    @tool(name="file_writer")
    def write_file(path: str, content: str, overwrite: bool = False) -> str:
        """Write a UTF-8 text file inside the workspace.

        Args:
            path: File path, relative to the workspace root.
            content: Text to write.
            overwrite: Replace the file if it already exists.
        """
        resolved = space.resolve(path)
        if resolved.is_dir():
            raise ToolError(f"'{path}' is a directory, not a file.")
        if resolved.exists() and not overwrite:
            raise ToolError(f"File already exists: {path}. Pass overwrite=true to replace it.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {space.relative(resolved)}"

    return write_file  # type: ignore[no-any-return]


# -- command runner ----------------------------------------------------------------

COMMANDS_ENV_VAR = "TEIFORGE_ENABLE_COMMANDS"


def make_command_runner(workspace: Optional[Workspace] = None) -> Tool:
    """Create a ``command_runner`` tool limited to a few read-only built-in commands."""
    space = workspace or Workspace()

    @tool(name="command_runner")
    def run_command(command: str) -> str:
        """Run a restricted command: echo, pwd, ls [path] or python --version.

        Args:
            command: The command line to run. No shell is used.
        """
        if os.getenv(COMMANDS_ENV_VAR, "0") != "1":
            raise ToolError(
                f"Command execution is disabled. Set {COMMANDS_ENV_VAR}=1 to allow the restricted commands."
            )
        text = re.sub(r"^(?:run|execute)\s+", "", command.strip(), count=1, flags=re.IGNORECASE)
        if not text:
            raise ToolError("No command was provided.")
        parts = text.split(maxsplit=1)
        executable = parts[0]
        arguments = parts[1] if len(parts) == 2 else ""

        if executable == "echo":
            return f"Command output:\n{arguments.strip() or 'Command returned no output.'}"
        if executable == "pwd":
            if arguments:
                raise ToolError("Usage: pwd")
            return f"Command output:\n{space.root}"
        if executable == "ls":
            try:
                parsed = shlex.split(arguments, posix=os.name != "nt")
            except ValueError as exc:
                raise ToolError(f"Invalid ls arguments: {exc}") from exc
            if len(parsed) > 1:
                raise ToolError("Usage: ls [path]")
            target = parsed[0] if parsed else "."
            if os.name == "nt" and len(target) >= 2 and target[0] == target[-1] and target[0] in "\"'":
                target = target[1:-1]
            resolved = space.resolve(target)
            try:
                entries = sorted(os.listdir(resolved))
            except OSError as exc:
                raise ToolError(f"Command execution failed: {type(exc).__name__}: {exc}") from exc
            return "Command output:\n" + ("\n".join(entries) or "Command returned no output.")
        if executable in {"python", "python3"}:
            if arguments not in {"--version", "-V"}:
                raise ToolError(f"Command '{executable}' is restricted to --version or -V.")
            return f"Command output:\n{sys.version.split()[0]}"
        raise ToolError(f"Command '{executable}' is not allowed. Allowed: echo, pwd, ls, python --version.")

    return run_command  # type: ignore[no-any-return]


# -- built-in instances ------------------------------------------------------------

WebSearchTool: Tool = tool(web_search)
FileReaderTool: Tool = make_file_reader()
FileWriterTool: Tool = make_file_writer()
CommandRunnerTool: Tool = make_command_runner()

DEFAULT_TOOLS: Dict[str, Tool] = {
    "web_search": WebSearchTool,
    "file_reader": FileReaderTool,
    "file_writer": FileWriterTool,
    "command_runner": CommandRunnerTool,
}


def builtin_tools(workspace: Optional[Workspace] = None) -> Dict[str, Tool]:
    """Return the built-in tools, with file tools bound to ``workspace`` when given."""
    if workspace is None:
        return dict(DEFAULT_TOOLS)
    return {
        "web_search": WebSearchTool,
        "file_reader": make_file_reader(workspace),
        "file_writer": make_file_writer(workspace),
        "command_runner": make_command_runner(workspace),
    }


__all__ = [
    "Tool",
    "ToolError",
    "ToolResult",
    "tool",
    "web_search",
    "normalize_search_query",
    "parse_write_command",
    "make_file_reader",
    "make_file_writer",
    "make_command_runner",
    "builtin_tools",
    "WebSearchTool",
    "FileReaderTool",
    "FileWriterTool",
    "CommandRunnerTool",
    "DEFAULT_TOOLS",
    "LEGACY_ERROR_PREFIXES",
    "COMMANDS_ENV_VAR",
]
