---
description: Draft the plan file for the next TeiForge version (docs/plans/vX.Y-plan.md) from the roadmap and the current code, then stop for approval.
argument-hint: "[version, e.g. 0.4] [optional focus]"
disable-model-invocation: true
---

Draft the plan for TeiForge version $0. Extra focus from the maintainer: $ARGUMENTS

Work in plan mode: read and analyse, write only the plan file, change no code.

## Inputs to read

- `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `SECURITY.md`.
- The previous plan in `docs/plans/` (structure, out-of-scope items, open risks).
- The relevant source in `teiforge/` and the open GitHub issues (via the GitHub MCP server if
  connected) for user-reported needs.

## Candidate scope (from the v0.3 plan, "out of scope")

1. Pluggable web-search providers (keep DuckDuckGo as default; add SearXNG/Brave/Tavily
   behind an interface and optional extras).
2. Async API and streaming (`Agent.arun`, token streaming events, async providers).
3. Local web UI as an optional extra (`teiforge[ui]`, `teiforge ui`, bound to 127.0.0.1),
   built only on the public API and the event stream.
4. Documentation site (mkdocs) published from CI.

Do not assume all four belong in $0. Recommend a scope based on value, risk and effort, and
say which items move to a later version and why.

## Plan file

Write `docs/plans/v$0-plan.md` in Vietnamese using the same structure as the previous plan:

1. Bối cảnh (with evidence: file and function names, issue numbers).
2. Mục tiêu and what is out of scope.
3. Thiết kế for each item: public API sketch, files touched, new optional dependencies,
   backward compatibility, security impact (workspace, network, prompt injection).
4. Chất lượng và hạ tầng (tests, CI changes).
5. Tiêu chí nghiệm thu as unchecked boxes that can be verified objectively.
6. Việc tác giả cần làm thủ công.
7. Rủi ro with mitigations.
8. Thứ tự triển khai: an ordered list of small work items, each sized for one pull request,
   each with an ID (for example `P1`, `P2`) that `/tf-feature` can reference.

Mark every assumption you could not verify. Then stop and ask the maintainer to review the
plan before any implementation starts.
