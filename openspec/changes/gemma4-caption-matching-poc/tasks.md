## 1. POC Script — Layout Detection

- [ ] 1.1 Create `photoalbums/scripts/poc_caption_gemma4.py` with CLI args: `--image` (default `Family_1980-1985_B08_P16_V.jpg`), `--lmstudio-url`, `--gemma-model`, `--debug-image`
- [ ] 1.2 Add standard Docling pipeline call (reuse `run_docling_pipeline` from `_docling_pipeline.py`) and print detected bounding boxes
- [ ] 1.3 Add heron pipeline call: attempt to load `docling-project/docling-layout-heron` locally; if unavailable print a clear skip message; if available print bounding boxes alongside standard pipeline output

## 2. POC Script — Gemma4 Caption Matching

- [ ] 2.1 Add `_sort_regions_reading_order(regions, img_height, row_tolerance_frac=0.10)` helper that applies a strict coordinate-based scanline sort: group regions into rows by y-tolerance band (ratio, not pixels), sort rows top-to-bottom, then sort within each row by x — required for both standard Docling and Heron output since neither preserves reading order
- [ ] 2.2 Add `_call_gemma4_caption_matching(image_path, base_url, model) -> dict[int, str]` that sends the numbered-photo prompt to LM Studio and parses `{"photo-1": ..., "photo-2": ...}` JSON (with regex fallback for wrapped responses)
- [ ] 2.3 Wire sort + Gemma4 call in POC script: sort standard-pipeline regions, call Gemma4, merge captions, print `(index, bbox, caption)` for each region
- [ ] 2.4 Implement `--debug-image` flag: draw bounding boxes with sort-order labels on the test image using Pillow and write PNG to `<image-stem>_debug.png`

## 3. Production Integration — Gemma4 Caption Step

- [ ] 3.1 Move `_sort_regions_reading_order` and `_call_gemma4_caption_matching` to `photoalbums/lib/ai_view_regions.py` (or a new `_gemma4_caption_matching.py` helper module)
- [ ] 3.2 Add `default_gemma4_caption_model()` to `ai_model_settings.py` (env var `IMAGO_GEMMA4_CAPTION_MODEL`, default `"gemma-3-27b-it"` or similar)
- [ ] 3.3 In `_detect_regions_docling`, after bounding boxes are returned, call `_call_gemma4_caption_matching` and assign captions via sort-order merge; replace the `associate_captions(regions, captions, img_w)` call for the docling path

## 4. Production Integration — Disable OCR

- [ ] 4.1 In `_docling_pipeline.py`, set `do_ocr=False` on `PdfPipelineOptions` for the standard image pipeline path
- [ ] 4.2 Verify existing tests still pass; update any tests that relied on OCR-derived caption_hint values being populated by the docling pipeline

## 5. Tests

- [ ] 5.1 Add unit tests for `_sort_regions_reading_order`: single region, two-row layout, same-row tie-breaking by x
- [ ] 5.2 Add unit tests for `_call_gemma4_caption_matching` response parsing: clean JSON, JSON in code fence, malformed JSON (expect empty dict + no exception)
- [ ] 5.3 Add integration-style test for the merge step: mock docling returning 3 regions + mock Gemma4 returning 3 captions → verify correct assignment after sort
- [ ] 5.4 Update `test_ai_view_regions.py` to cover LM Studio offline path (Gemma4 step skipped, regions saved with empty captions)
