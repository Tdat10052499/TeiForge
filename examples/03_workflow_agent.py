#!/usr/bin/env python3
"""Workflow agent: custom tools plus a live view of every step.

Run: python examples/03_workflow_agent.py
With OPENAI_API_KEY (or ANTHROPIC_API_KEY) set, the model chooses which tools to
call; without one, the keyword router handles the tasks.
"""

from teiforge import Agent, AgentConfig, Event, tool


@tool
def word_count(text: str) -> str:
    """Count the words in a piece of text.

    Args:
        text: The text to count.
    """
    return f"{len(text.split())} words"


def show(event: Event) -> None:
    if event.type == "tool_call":
        print(f"  -> {event.data['name']}({event.data['arguments']})")
    elif event.type == "tool_result":
        print(f"  <- {event.data['name']} {'ok' if event.data['ok'] else 'failed'}")


def main() -> None:
    agent = Agent(
        config=AgentConfig(name="workflow-agent", tools=["web_search", "file_reader"], max_steps=5),
        tools=[word_count],
    )

    for task in ["Search for AI agent libraries", "How many words are in 'agents use tools well'?"]:
        print(f"Task: {task}")
        result = agent.run_detailed(task, on_event=show)
        print(result.output)
        print()


if __name__ == "__main__":
    main()
