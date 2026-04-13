# AGENTS.md

Purpose: repository-wide operating rules for AI coding agents working on this project.

## Core Policy

- You are a software engineer that excels at doing what is asked and not overgineering a solution that wasn't requested since you understand the scale of the project you're working and are an expert at balancing performance, complexity vs. the size of the project.
- You are also good at designing simple tight systems where the code is elegantly takes into consideration future bugs.
- You use `just dupes`, `just deadcode`, `just complexity`, and `just test` to validate the quality of the code changes.
- Prefer stateless and reconstructing state from ground truth and rather than storing data in a database when possible.
- Limit code file sizes to about 500 lines.
- Do not use brittle regex and string replaacments to edit AI model responses, improve the prompt instead to get the correct output in JSON.
- Do not use Tesseract for OCR.
- Always bubble up the underlining errors when error reporting, don't interpet the errors or discard low level errors, for example if you try to write to a file and you get a permission error, you would buble up the error to the user or write the log as: <intention or process failed due to>:<OS permission error output>.
- When troubleshooting bugs, consider adding better diagnostics to streamline the troubleshooting process.
- Scope: Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
- Documentation: Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident.
- Defensive coding: Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs).
- Abstractions: Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task.

## Project Skills

- Project-local skills live under `skills/`.
- Check `skills/` before changing AI prompting, captioning, indexing, or other model-behavior workflows in this repo.
- The base skill is `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` (orchestration, shared rules, shared prompt sections).
- `_caption_prompts.py` loads all prompt sections from `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`.
- Supporting skill documentation may live next to a skill under `references/` or as additional markdown files in `skills/`.
- If code loads prompt sections from a `SKILL.md`, update the skill file rather than adding brittle response post-processing in Python.

## Photo Album File Naming Convention

All photo album files use a structured naming scheme:

```
{Collection}_{Year}_B{book}_P{page}_{type}.{ext}
```

| Type token | Role | Archive ext | View ext |
|------------|------|-------------|----------|
| `_S##` | Raw scan | `.tif` | — |
| `_D##-##` | Derived image | `.tif` | — |
| `_V` | View page (any scan count) | — | `.jpg` |
| `_D##-##_V` | View derived image | — | `.jpg` |

Rules:
- `_V` always and only marks a view output. `_S##` always and only marks an archive scan.
- `_D##-##` identifies a derived image; append `_V` for the view JPEG.
- Archive files are `.tif` and `.png`; view files are `.jpg` — no exceptions.
- `dc:source` on any view file references the archive TIF scan(s) it was derived from.
- Pages are numbered starting at P01. P00 is not a valid page number.
- XMP sidecars share the same stem as their companion image file (`.xmp` extension).

## Data and Schema Migrations

- Treat `metadata/*/render_settings.json` and related metadata as migratable assets.
- When renaming keys or structures:
  - update readers/writers in code,
  - run migration across existing metadata files,
  - remove old keys/paths unless explicitly told to keep them.
- Keep a single canonical schema in code and on disk.
- Prefer standard XMP schema fields over custom namespaces for `imago`.
- Photo Albums are located in `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`

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

When `just dupes` reports a duplicate-code finding (SKY-C401):

- **Do not** make superficial edits (rename variables, reorder statements, split into slightly different forms) to make the code look different to the detector. That is evading the problem, not solving it.
- **Do** refactor the duplicated logic into a shared function, helper, or class that both call sites use. The goal is genuine reuse — one canonical implementation, multiple callers.
- If the duplicated code serves genuinely different purposes and sharing would introduce harmful coupling, explain why before leaving it as-is. This is the exception, not the default.

## Python Environment

- For Python commands in this repo, do not rely on PATH-resolved `python`.
- Use `uv sync` from the repo root to create or update the project environment.
- From the repo root use  `just ...` for test, lint, and validation commands.
- Validate changed Python modules with `uv run python -m py_compile` when possible.
- To run ad-hoc python scripts use `uv run python -c`

## Git Hygiene

- Do not revert unrelated local changes.
- Commit only files related to the requested task unless asked otherwise.
- Use clear commit messages that describe behavior change.
- Git hooks are configured to run tests, don't use `--no-verify` to skip tests, instead fix the errors if there are any.

## If Unsure

- Ask for a decision only when truly ambiguous.
- Otherwise choose the simplest forward-moving implementation consistent with these rules.
- If there are lint errors that require many exceptions, ask what to do, we may want to releax the linter.
