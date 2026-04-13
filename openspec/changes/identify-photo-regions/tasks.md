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

## 9. Docling Model — Parser & Integration

- [x] 9.1 Add a `docling` alias to `[models]` in `photoalbums/ai_models.toml` pointing to `granite-docling-258m`; set `view_region_model = "docling"` to activate it (the model alias name contains "docling", which triggers the docling code path by substring match)
- [x] 9.2 Create `photoalbums/lib/_docling_parser.py` with `parse_doctag_response(content, img_w, img_h) -> list[RegionResult]` — parses the `<doctag>` XML returned by the granite-docling model; each `<picture>` element yields one `RegionResult`; the four `<loc_X>` child tags give top/left/bottom/right on a 0–500 scale, converted to pixel coordinates as `loc / 500 * dimension`; a `<caption>` child element sets `caption_hint` on that region
- [x] 9.3 In `_docling_parser.py`, after building the initial region list, merge region pairs that overlap by more than 15% of the smaller area into a single union bounding box; repeat iteratively until no pairs exceed the threshold; merged regions carry no `caption_hint` (caption association runs after)
- [x] 9.4 In `_docling_parser.py`, after merging, associate `<paragraph>` elements as caption fallback for regions whose `caption_hint` is still empty: assign a paragraph whose horizontal span overlaps the region and sits within one text-line height of its boundary; if a paragraph overlaps multiple regions equally, set `caption_ambiguous=True` on all; a paragraph centred in the middle third of the page width that is not adjacent to any single region is also broadcast to all regions with `caption_ambiguous=True`
- [x] 9.5 In `ai_view_regions.detect_regions()`, before the existing LM Studio call, check if the resolved model name contains `"docling"` (case-insensitive); if so: send prompt `"Convert this page to docling."` with the image (no `response_format` schema), pass the raw text response to `parse_doctag_response()`, skip the repair-prompt retry loop, and handle outcomes as follows: regions pass validation → return them; no `<picture>` elements in response → write `view_regions` pipeline step `result: "no_regions"` and return `[]`; validation fails → write `view_regions` pipeline step `result: "validation_failed"` and return `[]` (the step prevents re-runs; `--force` clears it)

## 9a. Tests for Docling Parser Code

- [x] 9a.1 Test `parse_doctag_response()` using a doctag string with four `<picture>` elements (one with an embedded `<caption>`): verify pixel bounding boxes match expected conversions and `caption_hint` is extracted correctly
- [x] 9a.2 Test merge logic in isolation: two regions overlapping >15% → merged to union box; two regions overlapping ≤5% → unchanged; three-way chain (A overlaps B, B overlaps C) → all three merged into one
- [x] 9a.3 Test paragraph caption association: embedded `<caption>` takes priority over nearby `<paragraph>`; nearby `<paragraph>` fills empty `caption_hint`; centered paragraph broadcasts to all; paragraph outside threshold leaves `caption_hint` empty

## 10. OCR Text Propagation to Crop Sidecars

- [ ] 10.1 In `ai_photo_crops._write_crop_sidecar()`, `view_state` is already loaded from the source view's XMP sidecar before this function is called; replace the hardcoded `ocr_text=""` arg to `write_xmp_sidecar()` with `str(view_state.get("ocr_text") or "").strip()`; also update the `caption` local variable: if `resolve_region_caption()` returns empty and `ocr_text` is non-empty, set `caption = ocr_text` before passing to `write_xmp_sidecar()` (field is written as `imago:OCRText` per `xmp_sidecar.py` line 826)
- [ ] 10.2 Add unit tests: (a) caption present → used as `dc:description`, OCR text written to `imago:OCRText`; (b) no caption, OCR text present → OCR text used as both `dc:description` and `imago:OCRText`; (c) both empty → both fields empty, no error

## 11. CLI & MCP — No Changes Required

The detect-view-regions command in `commands.py` reads `model_name = default_view_region_model()` from config and passes it through to `detect_regions()`. Switching to the docling model only requires changing the active alias in `ai_models.toml` (task 9.1). No new CLI flags or MCP tool changes are needed.
