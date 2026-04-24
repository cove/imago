## Context

The existing `photoalbums/` pipeline already has a step graph for `ai-index` and prompt debug artifacts for model calls. Runtime prompt text, however, is still stored in `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` and loaded by exact markdown section names through `_prompt_skill.py`. Some prompts and many generation parameters also live inline in modules such as `_caption_prompts.py`, `ai_ocr.py`, `ai_date.py`, and `ai_verify_crops.py`.

This makes a prompt change cross-cutting even when the intended behavior belongs to one pipeline step. It also hides tuning parameters from the prompt author, because token limits, temperatures, image caps, and retry parameters are split between CLI defaults, constants, and hard-coded request payloads.

## Goals / Non-Goals

**Goals:**

- Make runtime prompt ownership match the existing `photoalbums/` pipeline steps.
- Move prompts out of `SKILL.md` and into `photoalbums/prompts/`.
- Put step-specific tunable parameters next to the prompt files they affect.
- Preserve the existing `photoalbums/` command flow, XMP outputs, and LM Studio model-selection defaults.
- Make prompt and parameter changes visible in prompt debug artifacts.
- Include prompt and parameter hashes in existing step input hashes so changed prompt assets trigger targeted reruns.
- Bubble up file read, TOML parse, JSON parse, and template rendering errors with their underlying exception details.

**Non-Goals:**

- No `photoalbums2/` work.
- No Dagster, Streamlit, or new pipeline framework.
- No replacement of LM Studio as the default inference path.
- No model alias migration out of `photoalbums/ai_models.toml`.
- No broad refactor of caption, OCR, date, location, or verification output semantics.
- No prompt editing UI.

## Decisions

### 1. Prompt assets live under `photoalbums/prompts/`

Prompt files will be organized by pipeline step and prompt variant:

```text
photoalbums/prompts/
  shared/
  ai-index/
    ocr/
    caption/
    people-count/
    locations/
    date-estimate/
  verify-crops/
    verification/
    retry/
    parameter-suggestion/
```

This mirrors the existing code path rather than introducing a new conceptual pipeline. `shared/` is allowed only for rules that are actually reused by multiple steps. Step prompts include shared text explicitly through the loader so the rendered prompt remains reconstructable.

Alternative considered: one large `prompts.toml`. Rejected because it recreates the current "many sections in one file" problem and makes per-step review harder.

### 2. Use simple file formats

Use Markdown or plain text files for prompts, TOML for tunable parameters, and JSON for response schemas where a schema is already externalized or easy to externalize. Python remains responsible for output dataclasses, parsers, and request construction.

Alternative considered: YAML prompt manifests. Rejected because TOML is already used in this project and Python 3.11 includes `tomllib` for read-only parsing.

### 3. Add a small prompt asset loader

Add a loader module, tentatively `photoalbums/lib/ai_prompt_assets.py`, responsible for:

- resolving prompt asset paths relative to `photoalbums/prompts/`
- reading text, TOML, and JSON files
- rendering runtime variables
- computing deterministic hashes for prompt text and parameter payloads
- returning provenance metadata for debug output and step hashing

The loader should be intentionally small. It should not become a plugin system, a prompt registry DSL, or a second orchestration layer.

Alternative considered: keep using `_prompt_skill.py` but point it at prompt markdown files. Rejected because the existing parser depends on exact markdown headings and would preserve the current section-coupling problem.

### 4. Keep model aliases separate from step parameters

`photoalbums/ai_models.toml` remains the source for model aliases and default model selection. Step parameter files own model-call knobs such as `max_tokens`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, retry rungs, timeout, and image edge limits.

This keeps "which model" separate from "how this step calls the model."

### 5. CLI and render-settings overrides remain explicit overrides

Existing overrides such as `--caption-prompt-file`, `--caption-max-tokens`, `--caption-temperature`, and render setting keys continue to work. The resolved prompt asset metadata must record that an override was used so debug artifacts distinguish prompt-tree defaults from per-run overrides.

### 6. Prompt and parameter hashes participate in step invalidation

Prompt and parameter hashes should be added only to the owning step's input hash. For example, a caption prompt edit should invalidate `caption` and downstream dependents through the current step graph, but it should not invalidate OCR or people. This follows the existing `ai_index_steps.py` design where each step hashes only relevant inputs.

### 7. Prompt debug becomes the audit surface

Existing prompt debug records should include source path, prompt hash, params path, params hash, rendered system prompt, rendered user prompt, resolved model-call parameters, and override source. The prompt debug artifact should be enough to answer "what did we send and why did this rerun?" without mentally executing the Python prompt assembly.

## Risks / Trade-offs

- **Risk: Prompt migration changes behavior accidentally** -> Migrate step by step and add snapshot-style tests for rendered prompt fragments before removing `SKILL.md` loading.
- **Risk: Hashing every file read adds overhead** -> Prompt files are small; cache by file mtime and path where useful, following the current `_prompt_skill.py` reload pattern.
- **Risk: CLI overrides obscure source of truth** -> Record override source in prompt debug and include resolved values in step hashes.
- **Risk: Shared prompt files create hidden coupling** -> Keep shared files narrow and require each step to include them explicitly.
- **Risk: Missing or malformed prompt files stop processing** -> This is desired for production prompt assets; errors must include the underlying OS or parser message.

## Migration Plan

1. Add the prompt asset loader and tests without changing runtime prompt text.
2. Create `photoalbums/prompts/` with current prompt text copied from `SKILL.md` and inline constants.
3. Switch OCR and date estimate to the prompt loader first because their prompt surfaces are narrow.
4. Switch caption, people count, and locations next.
5. Switch `verify-crops` prompt variants and tuning params last.
6. Add prompt and parameter hashes to step input hashes for each migrated step.
7. Update prompt debug metadata for migrated steps.
8. Remove production reads from `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`.
9. Update `AGENTS.md` and `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` so future prompt edits target `photoalbums/prompts/`.

Rollback is straightforward until the final cleanup: restore the old call sites to `_prompt_skill.py` and keep the prompt tree unused. After cleanup, rollback is still limited to restoring the previous prompt loader code and prompt sections from version control.

## Open Questions

- Whether prompt template rendering should stay with the current `{name}` substitution semantics or use a stricter minimal template syntax.
- Whether JSON response schema files should be introduced for every structured call now, or only for calls where the schema is already represented as data.
- Whether `caption_matching` prompt assets should be included in this change or handled in a follow-up because it belongs to region detection rather than `ai-index`.
