---
description: Run the TeiForge quality gate (ruff lint, ruff format, mypy, pytest with coverage) and fix what fails.
argument-hint: "[--report-only]"
---

Run the TeiForge quality gate exactly as CI does, in this order, using the commands in
`CLAUDE.md`:

1. `ruff check teiforge tests examples`
2. `ruff format --check teiforge tests examples`
3. `mypy teiforge`
4. `python -m pytest --cov=teiforge --cov-fail-under=85`

Arguments: $ARGUMENTS

- If the arguments contain `--report-only`, do not edit anything; just report.
- Otherwise fix failures with the smallest correct change:
  - Formatting: run `ruff format teiforge tests examples`.
  - Lint and type errors: fix the code; add `# type: ignore[...]` or `noqa` only with a
    one-line reason, and never to hide a real bug.
  - Failing tests: find the root cause first. Change a test only if the test itself is wrong,
    and say why. Never delete or skip a test to make the gate pass.
  - Coverage below 85%: add meaningful tests for the uncovered branches.
- Re-run the whole gate after fixing until all four pass.

Report in Vietnamese: a table of the four checks (before and after), the list of files you
changed with one line each on why, and anything you could not fix.
