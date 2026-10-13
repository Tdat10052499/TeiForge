# Security Policy

TeiForge gives language models access to tools that read files, write files and
(optionally) run commands, so security reports are taken seriously.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.3.x | Yes |
| < 0.3 (AgentForge) | No |

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through
[GitHub Security Advisories](https://github.com/Tdat10052499/teiforge/security/advisories/new).

Include the TeiForge version, a description of the issue, steps to reproduce, and the
impact you expect. You should receive an acknowledgement within 7 days. Once a fix is
released, you will be credited in the changelog unless you prefer otherwise.

## Security model

- **Workspace sandbox.** `file_reader`, `file_writer` and `command_runner` only touch paths
  inside the agent's workspace (the `workspace` option, `TEIFORGE_WORKSPACE`, or the current
  directory). Paths that escape it through `..`, absolute paths or symbolic links are refused.
  Choose the narrowest workspace that works for your task.
- **No silent overwrites.** `file_writer` refuses to replace an existing file unless
  `overwrite=true` is passed.
- **Commands are off by default.** `command_runner` only works with
  `TEIFORGE_ENABLE_COMMANDS=1`, never uses a shell, and only supports `echo`, `pwd`,
  `ls [path]` and `python --version`.
- **Prompt injection.** Web pages and files can contain instructions aimed at the model.
  Treat tool output as untrusted, keep the toolset minimal, and review what an agent wrote
  before you rely on it.
- **Secrets.** API keys are read from environment variables and are never written to
  memory files or run results. Do not put secrets inside prompts or saved conversations.

Reports about bypassing the workspace sandbox or the command allow-list are in scope.
