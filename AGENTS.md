# AGENTS.md

Purpose: repository-wide operating rules for AI coding agents working on this project.

## Core Policy

- Prefer forward-only changes.
- Do not add backward compatibility by default.
- Do not add fall backs, just fail.
- When schema/config formats change, migrate all project data forward in the same change.
- If backward compatibility is requested, implement it only when explicitly asked.
- Limit code file sizes to about 500 lines, if they go over that size, then ask about refactoring.
- Do not use brittle regex and string replaacments to edit AI model responses, improve the prompt instead to get the correct output.
- Do not write code for input or output from AI model requests or responses, prefer to write prompts to .skill files.
- Use `uv run ruff format` to automatically enforce formatting.

## Data and Schema Migrations

- Treat `metadata/*/render_settings.json` and related metadata as migratable assets.
- When renaming keys or structures:
  - update readers/writers in code,
  - run migration across existing metadata files,
  - remove old keys/paths unless explicitly told to keep them.
- Keep a single canonical schema in code and on disk.

## UI and Terminology

- Prefer precise naming that matches behavior.
- Current preferred term: `Gamma Correction` (not `Brightness`).
- Keep user-facing labels aligned with metadata/renderer semantics.

## Preview Render Behavior

- Preview should apply only the current wizard stage effects by default:
  - Review step: bad-frame repair only.
  - Gamma Correction step: gamma correction only.
  - Summary step: combined output.

## Engineering Defaults

- Make focused, minimal changes.
- Avoid introducing extra abstraction unless it reduces real maintenance cost.
- Update docs/help text when behavior or naming changes.

## Python Environment

- For Python commands in this repo, do not rely on PATH-resolved `python`.
- Use `uv sync` from the repo root to create or update the project environment.
- From the repo root, prefer `uv run ...` for test, lint, and validation commands.
- Validate changed Python modules with `uv run python -m py_compile` when possible.

## Git Hygiene

- Do not revert unrelated local changes.
- Commit only files related to the requested task unless asked otherwise.
- Use clear commit messages that describe behavior change.
- Git hooks are configured to run tests, don't use `--no-verify` to skip tests, instead fix the errors if there are any.

## If Unsure

- Ask for a decision only when truly ambiguous.
- Otherwise choose the simplest forward-moving implementation consistent with these rules.
- If there are lint errors that require many exceptions, ask what to do, we may want to releax the linter.

