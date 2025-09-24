# Model Capability Survey

Single-purpose tool to discover provider models and record whether each model is callable, supports function calling, and emits reasoning (“deep thinking”).

## Script

- `scripts/model_capability_survey.py`

## Requirements

- `~/.codex/config.toml` with a `model_provider` and `model_providers.<id>.base_url`
- `~/.codex/auth.json` containing `OPENAI_API_KEY`

## Usage

- Auto-discover models (via `/models`) and save a Markdown table:

  `python3 scripts/model_capability_survey.py --limit 20`

- Specify models explicitly:

  `python3 scripts/model_capability_survey.py gpt-4o gpt-5-codex`

- Options:
  - `--out PATH` (default `reports/model_capabilities.md`)
  - `--no-reasoning` (skip reasoning fields in request)
  - `--timeout SECS` per-model HTTP timeout (default 120)
  - `--echo CMD...` command array to request via tool call (default `echo hello-fc`)

## Notes

- Always uses the Responses API to consistently detect function calls and reasoning events.
- Declares both a generic `function` tool (`shell`) and the special `local_shell` for compatibility. The script only detects whether a function call is produced; it does not execute or return tool outputs.

