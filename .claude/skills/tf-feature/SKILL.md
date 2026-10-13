---
description: Implement one work item from a TeiForge plan file end to end - branch, plan, tests, code, quality gate, draft PR.
argument-hint: "[plan item ID or short description, e.g. P2]"
disable-model-invocation: true
---

Implement this work item: $ARGUMENTS

## 1. Understand (no edits yet)

- Find the item in the newest `docs/plans/v*-plan.md`. If it is not there, treat the
  arguments as the description and say that it is outside the plan.
- Read the code it touches and the related tests. List the files you expect to change.
- Check for conflicts with `CLAUDE.md` rules (dependencies, security model, compatibility).
- Present a short implementation plan in Vietnamese: approach, public API changes, files,
  tests to add, risks, and anything ambiguous. **Wait for approval before editing.**

## 2. Implement

1. Create a branch from the up-to-date base: `feat/<short-topic>` (or `fix/`, `chore/`).
2. Write the tests first for the new behaviour and run them to see them fail.
3. Implement the smallest change that makes them pass. Keep the public API typed and
   documented with docstrings. Keep new third-party dependencies in optional extras.
4. Update `README.md` for user-facing changes and add an entry under `## Unreleased` in
   `CHANGELOG.md`.
5. Run the full quality gate from `CLAUDE.md` until all four checks pass.
6. Commit in small, focused commits with English messages.

## 3. Hand over

- Push the branch and open a **draft** pull request using
  `.github/pull_request_template.md`, linking the plan item ID. Ask before pushing if you
  have not pushed in this session yet.
- In the plan file, do not tick the item yet; it is ticked after review and merge.

Report in Vietnamese: what changed (files and why), test results (count, coverage), what
was not verified, and the PR link.
