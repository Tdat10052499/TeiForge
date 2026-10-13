"""Core agent implementation."""

import os
import logging
from typing import Dict, Optional

from .config import AgentConfig
from .memory import Memory
from .tools import DEFAULT_TOOLS, Tool

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Agent:
    """An AI agent that can use tools and maintain memory."""

    def __init__(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AgentConfig] = None,
    ):
        if config is not None and name is not None:
            raise ValueError("Pass either name/model or config, not both.")
        if config is None:
            if name is None:
                raise ValueError("Agent name or config must be provided.")
            config = AgentConfig(
                name=name,
                model=model or os.getenv("AGENTFORGE_MODEL", "gpt-4o-mini"),
            )
        elif model is not None:
            raise ValueError("Pass the model through config when using AgentConfig.")

        self.config = config
        self.name = config.name
        self.model = config.model
        self.tools: Dict[str, Tool] = {}
        self.memory = Memory(max_entries=config.memory_size)
        self.task_count = 0
        self.enable_llm: bool = True

        for tool_name in config.tools:
            if tool_name not in DEFAULT_TOOLS:
                raise ValueError(f"Unknown built-in tool in AgentConfig: {tool_name}")
            self.add_tool(DEFAULT_TOOLS[tool_name])

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> bool:
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True
        return False

    def get_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool_name, tool in self.tools.items():
            lines.append(f"  - {tool_name}: {tool.description}")
        return "\n".join(lines)

    def run(self, user_input: str) -> str:
        response = self._think_and_execute(user_input)
        self._record_task(user_input, response)
        return response

    def _record_task(self, user_input: str, response: str) -> None:
        self.task_count += 1
        self.memory.add(user_input, role="user")
        self.memory.add(response, role="assistant")

    def _call_llm(self, messages: list) -> Optional[str]:
        """Call OpenAI if available and configured."""
        if OpenAI is None:
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception:
            logger.exception("OpenAI summarization failed; continuing without a summary.")
            return None

    def _summarize_text(self, text: str, max_bullets: int = 5) -> Optional[str]:
        """Ask the LLM to summarize the text into a short response.

        Returns None if LLM is not available or disabled.
        """
        if not self.enable_llm:
            return None

        if not text or not text.strip():
            return None

        prompt_messages = [
            {
                "role": "system",
                "content": self.config.system_prompt
                or "You are a concise assistant that summarizes the content of web search results.",
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following content into a short 1-line summary and then up to "
                    f"{max_bullets} bullet points. Keep it concise and include any source URLs if present.\n\n"
                    "Content:\n" + text
                ),
            },
        ]
        return self._call_llm(prompt_messages)

    def run_search_and_save(self, query: str, out_path: str) -> str:
        """Run a web search, optionally summarize the result with the LLM, and save both summary and raw output to a file.

        Returns a confirmation message or an explicit error string.
        """
        if not query or not query.strip():
            response = "Error: A search query is required."
        elif not out_path or not out_path.strip():
            response = "Error: An output file path is required."
        elif "web_search" not in self.tools:
            response = "Error: No web_search tool is available for this task."
        else:
            raw_result = self.tools["web_search"].run(query)
            if not isinstance(raw_result, str):
                response = "Error: Search returned an invalid result."
            elif raw_result.startswith(("Tool error:", "Web search failed:")):
                response = f"Error: Search failed: {raw_result}"
            elif not raw_result.strip():
                response = "Error: Search returned an empty result."
            else:
                summary = self._summarize_text(raw_result)
                if summary and summary.strip():
                    content_to_write = (
                        "Summary:\n"
                        f"{summary}\n\n"
                        "Original search result:\n"
                        f"{raw_result}\n"
                    )
                else:
                    content_to_write = f"Original search result:\n{raw_result}\n"

                try:
                    with open(out_path, "w", encoding="utf-8") as file_handle:
                        file_handle.write(content_to_write)
                except OSError as exc:
                    response = (
                        f"Error: Failed to save search results to {out_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    response = f"Saved search results to {out_path}"

        self._record_task(query, response)
        return response

    def _think_and_execute(self, user_input: str) -> str:
        lowered = user_input.lower()

        if any(keyword in lowered for keyword in ["search", "find", "lookup", "latest", "news", "research"]):
            if "web_search" in self.tools:
                raw = self.tools["web_search"].run(user_input)
                summary = self._summarize_text(raw)
                if summary:
                    return self._format_response("web_search", f"{summary}\n\nFull results:\n{raw}")
                return self._format_response("web_search", raw)

        if any(keyword in lowered for keyword in ["read", "open", "file", "load", "summarize"]):
            if "file_reader" in self.tools:
                payload = user_input if "read:" in user_input else f"read:{user_input}"
                return self._format_response("file_reader", self.tools["file_reader"].run(payload))

        if any(keyword in lowered for keyword in ["write", "save", "create", "store"]):
            if "file_writer" in self.tools:
                payload = user_input if "write:" in user_input else f"write:{user_input}"
                return self._format_response("file_writer", self.tools["file_writer"].run(payload))

        if any(keyword in lowered for keyword in ["run", "execute", "command", "bash", "shell"]):
            if "command_runner" in self.tools:
                return self._format_response("command_runner", self.tools["command_runner"].run(user_input))

        return self._default_response(user_input)

    def _format_response(self, tool_name: str, tool_output: str) -> str:
        error_prefixes = (
            "Tool error:",
            "Web search failed:",
            "File not found:",
            "Failed to read file:",
            "Failed to write file:",
            "Command execution failed:",
            "Command execution is disabled",
            "Command '",
            "Usage:",
            "No file path was provided.",
            "No command was provided.",
            "Invalid ls arguments:",
        )
        if tool_output.startswith(error_prefixes):
            return f"Error: {tool_output}"
        return f"[Using {tool_name}]\n\n{tool_output}"

    def _default_response(self, user_input: str) -> str:
        return (
            f"Agent '{self.name}' received task: \"{user_input}\"\n\n"
            "I don't have a direct tool for this task yet.\n\n"
            f"{self.get_tools_description()}\n\n"
            "Try using keywords like 'search', 'read', 'write', or 'run'."
        )

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


__all__ = ["Agent"]
