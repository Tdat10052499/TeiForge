"""TeiForge - a lightweight toolkit for building AI agents with tools, memory and workflows."""

from ._version import __version__
from .agent import Agent
from .config import AgentConfig
from .events import Event, RunResult
from .llm import AnthropicProvider, LLMError, LLMProvider, LLMResponse, OpenAIProvider, ToolCall
from .memory import Memory
from .tools import (
    CommandRunnerTool,
    FileReaderTool,
    FileWriterTool,
    Tool,
    ToolError,
    ToolResult,
    WebSearchTool,
    tool,
)
from .workspace import Workspace, WorkspaceError

__author__ = "Hồ Du Tuấn Đạt (Teichi)"
__license__ = "MIT"

__all__ = [
    "__version__",
    "Agent",
    "AgentConfig",
    "AnthropicProvider",
    "CommandRunnerTool",
    "Event",
    "FileReaderTool",
    "FileWriterTool",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Memory",
    "OpenAIProvider",
    "RunResult",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "WebSearchTool",
    "Workspace",
    "WorkspaceError",
    "tool",
]
