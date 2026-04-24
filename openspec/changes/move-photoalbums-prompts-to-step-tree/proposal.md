## Why

Runtime prompts for the existing `photoalbums/` AI pipeline are embedded in `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` and then reassembled across several Python modules. This makes prompt changes hard to review, hard to tune per step, and easy to forget when pipeline behavior changes.

## What Changes

- Add a `photoalbums/prompts/` directory tree that mirrors the existing `photoalbums/` pipeline steps and prompt variants.
- Move runtime prompt text out of `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` into version-controlled prompt files under `photoalbums/prompts/`.
- Add adjacent step-level parameter files for tunable model-call settings such as token limits, sampling parameters, image edge limits, and retry ladders.
- Add prompt asset loading that renders prompt templates, loads parameters, computes hashes, and surfaces underlying file/parse errors.
- Record prompt file paths, prompt hashes, parameter paths, parameter hashes, rendered prompts, and resolved parameters in prompt debug artifacts.
- Include prompt and parameter hashes in existing step staleness/input hashes so prompt or parameter edits rerun only affected steps and downstream dependents.
- Keep model alias selection in `photoalbums/ai_models.toml`; the new prompt tree owns step-specific prompt text and sampler/runtime parameters.
- Update operator documentation so `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` is documentation only and is no longer read by production prompt code.

## Capabilities

### New Capabilities

- `photoalbums-prompt-assets`: step-scoped prompt files, parameter files, rendering, provenance, and prompt/parameter invalidation for the existing `photoalbums/` pipeline.

### Modified Capabilities

- None.

## Impact

- **Affected code**: `photoalbums/lib/_prompt_skill.py`, `photoalbums/lib/_caption_prompts.py`, `photoalbums/lib/_caption_lmstudio.py`, `photoalbums/lib/ai_ocr.py`, `photoalbums/lib/ai_date.py`, `photoalbums/lib/ai_verify_crops.py`, `photoalbums/lib/ai_index_steps.py`, prompt debug helpers, and related tests.
- **Affected files**: new `photoalbums/prompts/` tree; updates to `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` and `AGENTS.md` documentation.
- **Configuration impact**: `photoalbums/ai_models.toml` remains the model alias source. Step-specific prompt and sampler settings move to prompt-tree files.
- **Runtime impact**: changing a prompt or parameter file invalidates only the owning step and downstream dependents through existing step hashes.
- **Out of scope**: `photoalbums2/`, Dagster, new pipeline frameworks, full pipeline redesign, and replacing LM Studio defaults.
