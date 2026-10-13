#!/usr/bin/env python3
"""Research agent: web search, with an LLM summary when a provider is configured.

Run: python examples/01_research_agent.py
Set OPENAI_API_KEY or ANTHROPIC_API_KEY to let the model pick tools and summarize.
"""

from teiforge import Agent, AgentConfig


def main() -> None:
    agent = Agent(config=AgentConfig(name="research-agent", tools=["web_search"]))

    for task in ["Find the latest AI agent frameworks", "Search for open-source AI tools"]:
        print(f"Task: {task}")
        print("-" * 60)
        result = agent.run_detailed(task)
        print(f"[mode: {result.mode}, ok: {result.ok}]")
        print(result.output)
        print()


if __name__ == "__main__":
    main()
