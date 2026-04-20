## 1. Caption Matching Library

- [x] 1.1 Create `photoalbums/lib/_caption_matching.py` with `sort_regions_reading_order(regions, img_height, row_tolerance_frac=0.10)` — coordinate-based scanline sort; group into rows by y-tolerance ratio, sort rows top-to-bottom, within each row left-to-right by x
- [x] 1.2 Add `call_lmstudio_caption_matching(image_path, base_url, model) -> dict[int, str]` — sends numbered-photo prompt to LM Studio, parses `{"photo-1": ..., "photo-2": ...}` JSON with regex fallback for wrapped responses; auto-discovers loaded model if `model` is empty
- [x] 1.3 Add `assign_captions_from_lmstudio(regions, captions) -> list` — maps 1-based photo indices from LM Studio response to sorted regions

## 2. Production Integration

- [x] 2.1 Wire `_apply_lmstudio_captions` into `_detect_regions_docling` in `ai_view_regions.py`; replaces `associate_captions` geometry matching for the docling path
- [x] 2.2 Add `caption_matching_model` to `ai_models.toml` and `default_caption_matching_model()` to `ai_model_settings.py` (same alias-table pattern as `view_region_model`)
- [x] 2.3 Set `do_ocr=False` on `PdfPipelineOptions` in `_docling_pipeline.py` — OCR output is no longer needed now that LM Studio handles caption assignment

## 3. Prompt

- [x] 3.1 Implement numbered-photo prompt: example JSON first with empty-string placeholders, field notes, context-carryover instruction (prepend adjacent-photo subject when caption refers to sibling), "Just return the JSON without any extra text or explanation"

## 4. Diagnostic Script

- [x] 4.1 Create `photoalbums/scripts/caption_matching_debug.py` — runs Docling layout, sorts regions, calls LM Studio, prints merged result; supports `--image`, `--lmstudio-url`, `--model`, `--debug-image`

## 5. Tests

- [x] 5.1 Unit tests for `sort_regions_reading_order`: single region, two-row layout, same-row tie-breaking by x
- [x] 5.2 Unit tests for `call_lmstudio_caption_matching` response parsing: clean JSON, JSON in code fence, malformed JSON (expect empty dict + no exception), LM Studio offline
- [x] 5.3 Integration test for merge step: mock docling returning 3 regions + mock LM Studio returning 3 captions → verify correct assignment after sort
- [x] 5.4 Test for LM Studio offline path: regions saved with empty captions, detection not aborted

## 6. Promotion Cleanup

- [x] 6.1 Rename diagnostic script from `poc_caption_gemma4.py` to `caption_matching_debug.py`; old path redirects for compatibility
- [x] 6.2 Replace `_gemma4_caption_matching.py` with re-export shim pointing to `_caption_matching.py`
- [x] 6.3 Update all internal references from Gemma4-specific names to LM Studio-generic names
