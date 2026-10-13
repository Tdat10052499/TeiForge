"""Core agent implementation."""

import os
from typing import Dict, Optional

from .memory import Memory
from .tools import Tool

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class Agent:
    """An AI agent that can use tools and maintain memory."""

    def __init__(self, name: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.model = model or os.getenv("AGENTFORGE_MODEL", "gpt-4o-mini")
        self.tools: Dict[str, Tool] = {}
        self.memory = Memory()
        self.task_count = 0
        # Controls whether the agent will call an LLM for summarization
        self.enable_llm: bool = True

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
        self.task_count += 1
        self.memory.add(user_input, role="user")
        response = self._think_and_execute(user_input)
        self.memory.add(response, role="assistant")
        return response

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
                temperature=0.3,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception:
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
                "content": "You are a concise assistant that summarizes the content of web search results.",
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

        Returns a confirmation message or an error string.
        """
        try:
            if "web_search" not in self.tools:
                return "No web_search tool is available for this task."

            # Record user intent
            self.memory.add(query, role="user")

            # Run web search
            raw_result = self.tools["web_search"].run(query)

            # Try to summarize (may return None)
            summary = self._summarize_text(raw_result)

            # Build content to write
            if summary:
                content_to_write = (
                    "Summary:\n"
                    f"{summary}\n\n"
                    "Original search result:\n"
                    f"{raw_result}\n"
                )
            else:
                content_to_write = f"Original search result:\n{raw_result}\n"

            # Write to file with UTF-8
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(content_to_write)

            response = f"Saved search results to {out_path}"
            self.memory.add(response, role="assistant")
            return response

        except Exception as exc:
            error_message = f"Failed to run search and save: {type(exc).__name__}: {exc}"
            self.memory.add(error_message, role="assistant")
            return error_message

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
            f"Tasks completed: {self.task_count}\n"
            f"Memory entries: {len(self.memory)}\n"
            f"Tools available: {len(self.tools)}\n"
        )

    def __str__(self) -> str:
        return f"Agent(name='{self.name}', model='{self.model}', tools={len(self.tools)})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["Agent"]
