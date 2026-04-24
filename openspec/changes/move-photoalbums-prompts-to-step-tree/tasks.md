## 1. Prompt Asset Loader

- [x] 1.1 Add `photoalbums/lib/ai_prompt_assets.py` with prompt root resolution, text loading, TOML params loading, optional JSON schema loading, variable rendering, and deterministic hashing.
- [x] 1.2 Ensure loader errors include the affected path and underlying OS, TOML, JSON, or render exception details.
- [x] 1.3 Add unit tests for successful render, missing prompt file, invalid TOML, hash changes, and cache/reload behavior.

## 2. Prompt Tree

- [x] 2.1 Create `photoalbums/prompts/shared/` for genuinely shared rules used by more than one step.
- [x] 2.2 Create `photoalbums/prompts/ai-index/ocr/` with OCR system/user prompts and params copied from current runtime behavior.
- [x] 2.3 Create `photoalbums/prompts/ai-index/caption/` and `people-count/` with current caption and people-count prompt text and params.
- [x] 2.4 Create `photoalbums/prompts/ai-index/locations/` with current location and location-query prompt text and params.
- [x] 2.5 Create `photoalbums/prompts/ai-index/date-estimate/` with current date-estimate prompt text and params.
- [x] 2.6 Create `photoalbums/prompts/verify-crops/verification/`, `retry/`, and `parameter-suggestion/` with current prompt variants and params.

## 3. Runtime Migration

- [x] 3.1 Switch `ai_ocr.py` from `SKILL.md` sections and inline prompt constants to prompt assets while preserving rendered prompt behavior.
- [x] 3.2 Switch `ai_date.py` to prompt assets while preserving rendered prompt behavior and structured output parsing.
- [x] 3.3 Switch `_caption_prompts.py` and `_caption_lmstudio.py` to prompt assets for caption, people count, location, and location-shown prompts.
- [x] 3.4 Switch `ai_verify_crops.py` to prompt assets for verification, retry, and parameter-suggestion prompts.
- [x] 3.5 Preserve existing CLI and render-settings override precedence for caption prompt and caption tuning values.

## 4. Parameters and Model Calls

- [x] 4.1 Load step-specific defaults from adjacent `params.toml` files for migrated model calls.
- [x] 4.2 Keep model alias resolution in `photoalbums/ai_models.toml` and avoid moving model selection into prompt params.
- [x] 4.3 Thread resolved params into LM Studio request payloads without changing output semantics.
- [x] 4.4 Record override sources when CLI or render settings replace prompt-tree parameter defaults.

## 5. Provenance and Invalidation

- [x] 5.1 Extend prompt debug metadata to include prompt paths, prompt hashes, params paths, params hashes, resolved params, and override sources.
- [x] 5.2 Include prompt and params hashes in the owning `ai-index` step input hashes.
- [x] 5.3 Ensure caption prompt changes invalidate caption and downstream steps without invalidating OCR, people, or objects.
- [x] 5.4 Ensure verify-crops prompt or params changes affect verify-crops review runs without invalidating ai-index step records.

## 6. Documentation Cleanup

- [x] 6.1 Update `AGENTS.md` to identify `photoalbums/prompts/` as the runtime prompt source of truth.
- [x] 6.2 Update `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` so it remains operator documentation and points runtime prompt edits to `photoalbums/prompts/`.
- [x] 6.3 Remove production runtime dependencies on `_prompt_skill.py` and retire or narrow tests that assert `SKILL.md` prompt loading.

## 7. Validation

- [x] 7.1 Run `uv run python -m py_compile` on changed Python modules.
- [x] 7.2 Run focused prompt, caption, OCR, date, verify-crops, and ai-index step tests.
- [x] 7.3 Run `just test`.
- [x] 7.4 Run `just dupes`.
- [x] 7.5 Run `just deadcode`.
- [x] 7.6 Run `just complexity`.
