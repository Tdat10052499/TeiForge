---
description: Set up the TeiForge dev environment, run the full quality gate and report the project state. Run once per machine or after pulling large changes.
disable-model-invocation: true
---

Prepare this machine to work on TeiForge and report the current state of the project.

1. Read `CLAUDE.md`, `README.md`, `CHANGELOG.md` and the newest plan in `docs/plans/`.
2. Check the toolchain and report versions: `python --version`, `git --version`,
   `git status`, current branch, and whether `.venv` exists.
3. If `.venv` does not exist, create it and install `-e ".[dev,llm]"` as described in
   `CLAUDE.md`. If it exists, make sure the package is installed in editable mode.
4. Run the four quality-gate commands from `CLAUDE.md`, one at a time, and record the result
   of each (pass/fail, test count, coverage %).
5. Do not change any source file. If a check fails, diagnose the cause and propose a fix,
   but wait for approval before editing.

Finish with a short report in Vietnamese:

- Environment: Python version, venv status, branch.
- Quality gate: a table with the four checks and their results.
- Plan status: which acceptance criteria in the newest plan are checked and which remain.
- Suggested next command (for example `/tf-smoke-llm` or `/tf-plan-next`).
