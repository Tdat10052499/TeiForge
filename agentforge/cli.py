"""Command-line interface for AgentForge."""

import argparse
import sys

from .agent import Agent
from .tools import DEFAULT_TOOLS


def create_research_agent() -> Agent:
    agent = Agent(name="research-agent")
    agent.add_tool(DEFAULT_TOOLS["web_search"])
    return agent


def create_file_agent() -> Agent:
    agent = Agent(name="file-analyzer")
    agent.add_tool(DEFAULT_TOOLS["file_reader"])
    agent.add_tool(DEFAULT_TOOLS["file_writer"])
    return agent


def create_workflow_agent() -> Agent:
    agent = Agent(name="workflow-agent")
    for tool in DEFAULT_TOOLS.values():
        agent.add_tool(tool)
    return agent


def main(args=None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentForge - AI Agent Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agentforge.cli --agent research --task "Search for AI agent frameworks"
  python -m agentforge.cli --agent file-analyzer --task "read:README.md"
  python -m agentforge.cli --agent workflow --task "Research AI agents and summarize the top options"
        """,
    )
    parser.add_argument("--agent", choices=["research", "file-analyzer", "workflow"], default="research")
    parser.add_argument("--task", required=True, help="Task for the agent to perform")
    parser.add_argument("--verbose", action="store_true", help="Print detailed metadata")
    parser.add_argument("--out", type=str, default=None, help="Optional output file to save search+summary results")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM summarization for this run")
    parsed_args = parser.parse_args(args)

    if parsed_args.agent == "research":
        agent = create_research_agent()
    elif parsed_args.agent == "file-analyzer":
        agent = create_file_agent()
    else:
        agent = create_workflow_agent()

    # Optionally disable LLM summarization
    if parsed_args.no_llm:
        agent.enable_llm = False

    if parsed_args.verbose:
        print(f"Agent: {agent}")
        print(agent.get_tools_description())
        print()

    # If an output file was provided, use run_search_and_save
    if parsed_args.out:
        print(f"Task: {parsed_args.task}\n")
        result = agent.run_search_and_save(parsed_args.task, parsed_args.out)
        print(result)
    else:
        print(f"Task: {parsed_args.task}\n")
        print(agent.run(parsed_args.task))

    if parsed_args.verbose:
        print()
        print(agent.get_memory_summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
