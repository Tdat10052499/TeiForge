---
description: Critically review the current TeiForge branch or a pull request against its plan item, the security model and the test suite. Read-only.
argument-hint: "[PR number, or empty for the current branch]"
disable-model-invocation: true
---

Review this change: $ARGUMENTS (if empty, review the current branch against `main`).

This is a read-only review: do not edit files, push or comment on GitHub unless asked.

1. Collect the diff (`git diff main...HEAD`, or the pull request files through the GitHub MCP
   server or `gh pr diff`). Identify the plan item it implements.
2. Check, citing file and line for every finding:
   - **Correctness**: logic errors, edge cases, error handling, behaviour on Windows paths.
   - **Plan fit**: does it do what the plan item says, no more and no less?
   - **Security** (see `SECURITY.md`): workspace escapes, new network access, secrets in logs
     or files, prompt-injection exposure from tool output, shell usage.
   - **API and compatibility**: breaking changes, missing type hints or docstrings,
     README/CHANGELOG updates.
   - **Tests**: does each behaviour change have a test that would fail without it? Any test
     touching the network or a real LLM? Coverage of the new branches.
   - **Dependencies**: anything new in core dependencies instead of an optional extra.
3. Run the quality gate in report-only mode and include the results.

Report in Vietnamese, grouped by severity: **Chặn merge**, **Nên sửa**, **Gợi ý**. For each
finding give the location, the evidence, and a concrete fix. Distinguish verified problems
from suspicions. End with a clear verdict: ready to merge, or what must change first.
