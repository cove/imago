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
- Do not use Tesseract for OCR. All text extraction must be done by the AI vision model.
- Use `uv run ruff format` to automatically enforce formatting.

## Project Skills

- Project-local skills live under `skills/`.
- Check `skills/` before changing AI prompting, captioning, indexing, or other model-behavior workflows in this repo.
- The base skill is `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` (orchestration, shared rules, shared prompt sections).
- Travel album prompts live in `skills/CORDELL_PHOTO_ALBUMS_TRAVEL/SKILL.md`.
- Family album prompts live in `skills/CORDELL_PHOTO_ALBUMS_FAMILY/SKILL.md`.
- `_caption_prompts.py` loads `Preamble Describe` from the album-type skill; all other sections from the base skill.
- Supporting skill documentation may live next to a skill under `references/` or as additional markdown files in `skills/`.
- If code loads prompt sections from a `SKILL.md`, update the skill file rather than adding brittle response post-processing in Python.

## Data and Schema Migrations

- Treat `metadata/*/render_settings.json` and related metadata as migratable assets.
- When renaming keys or structures:
  - update readers/writers in code,
  - run migration across existing metadata files,
  - remove old keys/paths unless explicitly told to keep them.
- Keep a single canonical schema in code and on disk.
- Prefer standard XMP schema fields over custom namespaces for `imago`.

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

## Duplicate Code (Skylos)

When `check_skylos.py` reports a duplicate-code finding (SKY-C401):

- **Do not** make superficial edits (rename variables, reorder statements, split into slightly different forms) to make the code look different to the detector. That is evading the problem, not solving it.
- **Do** refactor the duplicated logic into a shared function, helper, or class that both call sites use. The goal is genuine reuse — one canonical implementation, multiple callers.
- If the duplicated code serves genuinely different purposes and sharing would introduce harmful coupling, explain why before leaving it as-is. This is the exception, not the default.

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

