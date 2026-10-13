# Changelog

Notable changes to AgentForge are documented here.

## Unreleased

### Added

- Added OpenAI-powered summaries for web search results when the optional client and API key are configured.
- Added DuckDuckGo HTML result parsing as a fallback when the Instant Answer API has no results.
- Added CLI options to save search results and disable LLM summarization.

### Changed

- Restricted command execution to a small set of built-in operations and Python version reporting; commands are never passed through a shell.
- Restored project documentation and corrected the environment and dependency example files.
- Made search-and-save failures explicit, prevented failed searches from being saved as successful results, and kept task history consistent.
- Connected `AgentConfig` to agent construction, memory sizing, built-in tool selection, and OpenAI summary settings.
- Aligned the package version at `0.2.0` and declared OpenAI as an optional dependency.
