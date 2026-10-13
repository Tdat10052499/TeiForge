---
description: Manually smoke-test TeiForge against a real LLM provider (openai, anthropic or ollama). Uses real API calls and may cost money; never committed as a test.
argument-hint: "[openai|anthropic|ollama]"
disable-model-invocation: true
---

Verify that TeiForge works end to end with a real LLM provider. This closes the acceptance
item "manual test with real OpenAI/Anthropic APIs" in the current plan.

Provider: $ARGUMENTS (default: `openai` if empty)

Rules:

- Never print, echo, log or write an API key. Only check whether the environment variable
  is set (for example `[bool]$env:OPENAI_API_KEY` in PowerShell).
- Do not create any test file in `tests/`. Put temporary scripts in a scratch folder outside
  the repository, or run them inline, and delete them afterwards.
- Use a temporary directory as the agent workspace.
- Ask before each run that calls a paid API, and keep the number of calls small (at most ~6).

Steps:

1. Check prerequisites and stop with clear instructions if one is missing:
   - `openai`: package `openai` installed and `OPENAI_API_KEY` set.
   - `anthropic`: package `anthropic` installed and `ANTHROPIC_API_KEY` set.
   - `ollama`: `http://localhost:11434/v1` reachable and a model pulled; use
     `TEIFORGE_PROVIDER=openai`, `TEIFORGE_BASE_URL=http://localhost:11434/v1`,
     `TEIFORGE_MODEL=<model>`.
2. Run these scenarios with `Agent.run_detailed()` and record mode, ok, steps and tool calls:
   1. Plain question with no tools: "Reply with the word pong." Expect `mode == "llm"`.
   2. File tool: write `notes.txt` in the temp workspace first, then ask the agent to read it
      and summarise. Expect a `file_reader` tool call and the content in the answer.
   3. Custom `@tool` with numbers (e.g. multiply two floats). Expect correct JSON arguments.
   4. Sandbox: ask the agent to read `../outside.txt`. Expect the tool result to fail with
      "outside the workspace" and the model to explain the refusal.
   5. CLI: `teiforge run "<question>" --json --tools file_reader --workspace <temp dir>`.
3. If a scenario fails, separate a TeiForge bug from a provider limitation (model choice,
   rate limit, unsupported parameter). For a TeiForge bug, write a failing offline regression
   test with `ScriptedProvider` or a mocked client, then propose the fix; ask before editing.

Report in Vietnamese: a table of scenarios with pass/fail and evidence (event types, tool
names, short output excerpts without secrets), the provider and model used, and bugs found.
If all scenarios pass, propose ticking the corresponding item in the plan file.
