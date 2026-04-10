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
