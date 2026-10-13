#!/usr/bin/env python3
"""File agent: read and write files inside a sandboxed workspace.

Run: python examples/02_file_agent.py
Everything happens in a temporary directory; paths outside it are refused.
"""

import tempfile

from teiforge import Agent, AgentConfig


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        config = AgentConfig(
            name="file-agent",
            tools=["file_reader", "file_writer"],
            workspace=workspace,
            mode="router",  # deterministic: no LLM needed
        )
        agent = Agent(config=config)

        print(agent.run("write:demo.txt:TeiForge is a toolkit for building AI agents."))
        print(agent.run("Read demo.txt if it exists"))
        print(agent.run("read:../outside.txt"))  # refused: outside the workspace


if __name__ == "__main__":
    main()
