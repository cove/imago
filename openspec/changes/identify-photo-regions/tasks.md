## 1. Config & Model Registration

- [x] 1.1 Add `view-region = "google/gemma-4-26b-a4b"` entry to `photoalbums/ai_models.toml`
- [x] 1.2 Update `ai_model_settings.py` to read and expose `view_region_model` alongside existing model keys

## 2. Vision API — Region Detection

- [x] 2.1 Create `photoalbums/lib/ai_view_regions.py` with `detect_regions(image_path, force=False) -> list[RegionResult]`
- [x] 2.2 Implement LM Studio vision call in `ai_view_regions.py`: build base64 image message, send to `/v1/chat/completions` with structured JSON prompt requesting `[{index, x, y, width, height, confidence, caption_hint}]`
- [x] 2.3 Add JSON parse + retry loop (up to 3 attempts with stricter prompt on failure)
- [x] 2.4 Add `RegionResult` dataclass with pixel bounds, confidence, and caption_hint fields
- [x] 2.5 Implement XMP region read-back in `ai_view_regions.py`: parse existing `mwg-rs:RegionList` from the XMP sidecar to serve as cache (skip model call on `force=False` when regions already present)

## 3. Coordinate Conversion & Caption Association

- [x] 3.1 Implement `pixel_to_mwgrs(x, y, w, h, img_w, img_h) -> (cx, cy, nw, nh)` utility (centre-point normalised coords)
- [x] 3.2 Add unit tests for coordinate conversion (edge cases: region at corner, full-image region)
- [x] 3.3 Implement `associate_captions(regions, captions, img_width) -> list[RegionWithCaption]` using nearest-centre heuristic with 10%-width ambiguity threshold

## 4. XMP Write-back

- [x] 4.1 Register MWG-RS namespace (`http://www.metadataworkinggroup.com/schemas/regions/`) and `stArea` namespace (`http://ns.adobe.com/xap/1.0/sType/Area#`) in `xmp_sidecar.py`
- [x] 4.2 Add `write_region_list(xmp_path, regions_with_captions, img_w, img_h)` function to `xmp_sidecar.py` that builds `mwg-rs:RegionInfo` / `mwg-rs:RegionList` structure and replaces any existing region list
- [x] 4.3 Verify round-trip with exiftool: written XMP can be read back with correct region coordinates

## 5. MCP Endpoints

- [x] 5.1 Add `photoalbums_detect_view_regions(album_id, page=None, force=False)` tool to `mcp_server.py` — enqueues async job via `JobRunner`, returns `job_id`
- [x] 5.2 Add `photoalbums_review_view_regions(album_id, page)` tool to `mcp_server.py` — reads `mwg-rs:RegionList` from the XMP sidecar and returns structured dict with image path, dimensions, regions (pixel + normalised coords), and `caption_ambiguous` flag; returns `{"regions": [], "status": "not_detected"}` if no region list exists
- [x] 5.3 Add `photoalbums_update_view_region(album_id, page, region_index, x, y, width, height)` tool to `mcp_server.py` — updates the specified `mwg-rs:RegionList` entry directly in the XMP sidecar, returns confirmation with new normalised coords

## 6. CLI Integration

- [x] 6.1 Add `detect-view-regions` subcommand to `photoalbums.py` CLI (`photoalbums.py detect-view-regions <album_id> [--page N] [--force]`)
- [x] 6.2 Wire CLI command through to `ai_view_regions.detect_regions` + `xmp_sidecar.write_region_list`

## 7. Tests

- [x] 7.1 Add unit tests for `detect_regions` with a mocked LM Studio response (happy path + malformed JSON retry)
- [x] 7.2 Add unit tests for `associate_captions` (unambiguous, ambiguous, no captions)
- [x] 7.3 Add integration test for `write_region_list` verifying XMP output contains correct MWG-RS structure

## 8. Prompt Improvements

- [x] 8.1 Switch model output from pixel coords to normalised 0.0–1.0 coords; convert to pixel in `_parse_region_response` using actual image dimensions (fixes scaling bug where boxes appeared at 1/4 size in upper-left)
- [x] 8.2 Include original image pixel dimensions in the user prompt (e.g. `"The full image is 3840×2880 pixels."`) to help the model reason about region boundaries at the correct scale
- [x] 8.3 Set model temperature to `0.0` for fully deterministic region output

## 9. Docling Engine — Config & Integration

- [ ] 9.1 Add `view_region_engine` key to `ai_models.toml` (values: `lmstudio` | `docling`; default: `lmstudio`) and expose it through `ai_model_settings.py`
- [ ] 9.2 Add `docling` to project dependencies (`pyproject.toml` / `requirements.txt`)
- [ ] 9.3 Create `photoalbums/lib/_docling_regions.py` with `detect_regions_docling(image_path, converter) -> list[RegionResult]` that calls `DocumentConverter`, extracts `PICTURE` elements, and converts docling bounding boxes to pixel `RegionResult` objects; direct docling's HF model downloads to `HF_MODEL_CACHE_DIR` (same as `ai_ocr.py`)
- [ ] 9.4 Implement `DocumentConverter` reuse: instantiate once per CLI/MCP run and pass it into `detect_regions_docling()`, mirroring the existing `caption_engine_cache` / `ocr_engine_cache` pattern
- [ ] 9.5 In `_docling_regions.py`, after extracting `PICTURE` elements and before caption association, merge any pair of regions that overlap by more than 15% of the smaller region's area into a single union bounding box; apply iteratively until no merging pairs remain (mirrors `_merge_boxes()` in `ai_page_layout.py`)
- [ ] 9.6 Implement docling caption extraction in `_docling_regions.py` (runs after merge): collect `TEXT` elements from the same docling pass and associate using two rules — (1) centered-page caption: TEXT whose horizontal centre is in the middle third of the page and not adjacent to any photo boundary → broadcast to all regions with `caption_ambiguous=True`; (2) proximity caption: TEXT within one text-line height of one or more photo boundaries → associate with the photo(s) whose horizontal span it overlaps, broadcasting if it overlaps more than one
- [ ] 9.7 Wire engine selection into `ai_view_regions.detect_regions_for_view()`: when `engine == "docling"` call `_docling_regions.detect_regions_docling`; skip the LM Studio retry loop entirely
- [ ] 9.8 Write `view_regions` pipeline step in all outcomes: `result: "regions_found"` + `model: "docling"` on success; `result: "no_regions"` when no `PICTURE` elements found; `result: "validation_failed"` when validation fails (prevents infinite re-runs since docling is deterministic; re-processing requires `--force`)

## 9a. Tests for Docling Integration Code

- [ ] 9a.1 Test `_merge_overlapping_regions()`: two regions overlapping >15% → merged into union box; two regions overlapping ≤5% → unchanged; three-way chain merge (A overlaps B, B overlaps C) → all three merged
- [ ] 9a.2 Test caption association rules: centered TEXT → broadcast to all regions; TEXT below one photo within threshold → assigned to that region only; TEXT below two photos → broadcast to both; TEXT outside all thresholds → ignored
- [ ] 9a.3 Test coordinate conversion: given a mock docling bounding box and page size, verify the output pixel `RegionResult` matches expected values (exercises our conversion code, not docling internals)

## 10. OCR Text Propagation to Crop Sidecars

- [ ] 10.1 In `ai_photo_crops._write_crop_sidecar()`: (1) set `dc:description` to the docling-extracted region caption; if that is empty, use the source view's `ocr_text` from `view_state` as the fallback; (2) always write the source view's `ocr_text` to `imago:OCRText` on the crop sidecar regardless of which value was used for `dc:description` (note: field name is `imago:OCRText`, matching `xmp_sidecar.py` line 826)
- [ ] 10.2 Add unit tests: (a) docling caption present → used as `dc:description`, OCR text written to `imago:OCRText`; (b) no docling caption, OCR text present → OCR text used as both `dc:description` and `imago:OCRText`; (c) both empty → both fields empty, no error

## 11. CLI & MCP Updates for Docling Engine

- [ ] 11.1 Expose `--engine lmstudio|docling` flag on the `detect-view-regions` CLI subcommand; default reads from `ai_models.toml`
- [ ] 11.2 Update `photoalbums_detect_view_regions` MCP tool to pass engine selection through to `detect_regions_for_view()`
