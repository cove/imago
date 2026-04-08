## 1. Config & Model Registration

- [ ] 1.1 Add `view-region = "google/gemma-4-26b-a4b"` entry to `photoalbums/ai_models.toml`
- [ ] 1.2 Update `ai_model_settings.py` to read and expose `view_region_model` alongside existing model keys

## 2. Vision API — Region Detection

- [ ] 2.1 Create `photoalbums/lib/ai_view_regions.py` with `detect_regions(image_path, force=False) -> list[RegionResult]`
- [ ] 2.2 Implement LM Studio vision call in `ai_view_regions.py`: build base64 image message, send to `/v1/chat/completions` with structured JSON prompt requesting `[{index, x, y, width, height, confidence, caption_hint}]`
- [ ] 2.3 Add JSON parse + retry loop (up to 3 attempts with stricter prompt on failure)
- [ ] 2.4 Add `RegionResult` dataclass with pixel bounds, confidence, and caption_hint fields
- [ ] 2.5 Implement XMP region read-back in `ai_view_regions.py`: parse existing `mwg-rs:RegionList` from the XMP sidecar to serve as cache (skip model call on `force=False` when regions already present)

## 3. Coordinate Conversion & Caption Association

- [ ] 3.1 Implement `pixel_to_mwgrs(x, y, w, h, img_w, img_h) -> (cx, cy, nw, nh)` utility (centre-point normalised coords)
- [ ] 3.2 Add unit tests for coordinate conversion (edge cases: region at corner, full-image region)
- [ ] 3.3 Implement `associate_captions(regions, captions, img_width) -> list[RegionWithCaption]` using nearest-centre heuristic with 10%-width ambiguity threshold

## 4. XMP Write-back

- [ ] 4.1 Register MWG-RS namespace (`http://www.metadataworkinggroup.com/schemas/regions/`) and `stArea` namespace (`http://ns.adobe.com/xap/1.0/sType/Area#`) in `xmp_sidecar.py`
- [ ] 4.2 Add `write_region_list(xmp_path, regions_with_captions, img_w, img_h)` function to `xmp_sidecar.py` that builds `mwg-rs:RegionInfo` / `mwg-rs:RegionList` structure and replaces any existing region list
- [ ] 4.3 Verify round-trip with exiftool: written XMP can be read back with correct region coordinates

## 5. MCP Endpoints

- [ ] 5.1 Add `photoalbums_detect_view_regions(album_id, page=None, force=False)` tool to `mcp_server.py` — enqueues async job via `JobRunner`, returns `job_id`
- [ ] 5.2 Add `photoalbums_review_view_regions(album_id, page)` tool to `mcp_server.py` — reads `mwg-rs:RegionList` from the XMP sidecar and returns structured dict with image path, dimensions, regions (pixel + normalised coords), and `caption_ambiguous` flag; returns `{"regions": [], "status": "not_detected"}` if no region list exists
- [ ] 5.3 Add `photoalbums_update_view_region(album_id, page, region_index, x, y, width, height)` tool to `mcp_server.py` — updates the specified `mwg-rs:RegionList` entry directly in the XMP sidecar, returns confirmation with new normalised coords

## 6. CLI Integration

- [ ] 6.1 Add `detect-view-regions` subcommand to `photoalbums.py` CLI (`photoalbums.py detect-view-regions <album_id> [--page N] [--force]`)
- [ ] 6.2 Wire CLI command through to `ai_view_regions.detect_regions` + `xmp_sidecar.write_region_list`

## 7. Tests

- [ ] 7.1 Add unit tests for `detect_regions` with a mocked LM Studio response (happy path + malformed JSON retry)
- [ ] 7.2 Add unit tests for `associate_captions` (unambiguous, ambiguous, no captions)
- [ ] 7.3 Add integration test for `write_region_list` verifying XMP output contains correct MWG-RS structure
