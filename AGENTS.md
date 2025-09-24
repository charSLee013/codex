# Repository Guidelines

## Project Structure & Module Organization
`codex-rs/` hosts the Rust workspace; every crate keeps the `codex-` prefix (for example `codex-core`, `codex-tui`) with sources in `src/` and integration suites in `tests/`. Consult `codex-rs/docs/` and `codex-rs/tui/styles.md` for design decisions and UI patterns. `codex-cli/` contains the Node-based distribution wrappers, while `docs/` covers user-facing guidance and `scripts/` gathers cross-language tooling. Place assets and fixtures beside the code that consumes them to keep review diffs focused.

## Build, Test, and Development Commands
- `just fmt` (run inside `codex-rs/`): enforces the workspace rustfmt configuration before commits.
- `just fix -p codex-<crate>`: applies scoped clippy fixes; reserve bare `just fix` for workspace-wide updates.
- `just codex`, `just exec`, `just tui`: run key binaries locally during feature work.
- `cargo test -p codex-<crate>`: execute targeted Rust suites; follow with `cargo test --all-features` or `just test` (cargo nextest) when touching shared crates.
- `pnpm install` (inside `codex-cli/`): restores CLI dependencies; use `pnpm exec prettier --write .` before publishing JS changes.

## Coding Style & Naming Conventions
Rust code is formatted by rustfmt with the repo’s `imports_granularity=Item` setting, so avoid manual import juggling. Prefer inline interpolation (`format!("{value}")`) and ratatui’s Stylize helpers (`"status".green()`) per `tui/styles.md`. New crates, binaries, and feature flags should retain the `codex-` prefix for easy discovery. JavaScript assets follow Prettier defaults; keep files ASCII unless an existing file already uses Unicode glyphs.

## Testing Guidelines
Unit tests live alongside modules; end-to-end and snapshot cases live in crate-level `tests/`. TUI snapshots rely on insta—after intentional updates run `cargo test -p codex-tui`, inspect with `cargo insta pending-snapshots -p codex-tui`, then accept via `cargo insta accept -p codex-tui`. Prefer `pretty_assertions::assert_eq` for diffable failures. Use `cargo nextest run --no-fail-fast` for comprehensive checks before release, and leave existing `CODEX_SANDBOX_*` guards untouched so restricted-environment skips continue to work.

## Commit & Pull Request Guidelines
History favors an optional scope, imperative summary, and PR reference—`[exec] add include-plan-tool flag (#3461)` is a good model. Describe intent, highlight user-visible changes, and list verification steps (`just fmt`, `just fix`, targeted tests) in every PR. Link issues or design docs when applicable and include screenshots or terminal captures for UI or CLI updates.

## Sandbox & Configuration Notes
Do not modify logic tied to `CODEX_SANDBOX_ENV_VAR` or `CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR`; several integration tests depend on their current meaning. Document any new environment requirements in `docs/config.md` and flag migrations early in the PR description so downstream agents can adapt.
