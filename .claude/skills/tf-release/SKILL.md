---
description: Run the TeiForge release checklist for a version and stop before every irreversible step (merge, tag, publish).
argument-hint: "[version, e.g. 0.3.0]"
disable-model-invocation: true
---

Prepare the release of TeiForge version $ARGUMENTS. Verify everything first; perform
irreversible actions only after explicit approval in the chat, one at a time.

## 1. Verify (read-only)

- `teiforge/_version.py` equals `$ARGUMENTS`.
- `CHANGELOG.md` has a `## $ARGUMENTS - <date>` section and `## Unreleased` is empty or
  intentionally carried over.
- All acceptance criteria in the matching `docs/plans/v<major.minor>-plan.md` are checked.
  List any unchecked item and stop if a blocking one remains (for example the real-API smoke
  test from `/tf-smoke-llm`).
- The quality gate passes locally (run the four commands from `CLAUDE.md`).
- The release pull request exists and its CI is green. Use the GitHub MCP server if it is
  connected, otherwise `gh pr checks`. Report each job.
- The package builds: `python -m pip install build` then `python -m build`, and the wheel
  installs in a fresh venv where `teiforge version` prints `$ARGUMENTS`. Remove `dist/` and
  the temporary venv afterwards.
- PyPI readiness: remind the maintainer that a Trusted Publisher for `teiforge` must exist
  (see the comment at the top of `.github/workflows/release.yml`). You cannot verify this.

Report the results as a checklist in Vietnamese and ask whether to continue.

## 2. Release (each step needs a separate "yes")

1. Merge the release pull request into `main` (squash or merge commit: ask which).
2. Update local `main`, then create the annotated tag `v$ARGUMENTS` and push it. This starts
   `release.yml`, which publishes to PyPI and creates the GitHub release.
3. Watch the `Release` workflow until it finishes and report the result, including the PyPI
   URL `https://pypi.org/project/teiforge/$ARGUMENTS/` once it is live.

If any step fails, stop, show the error and propose a fix. Never delete or move a pushed tag
without explicit approval.
