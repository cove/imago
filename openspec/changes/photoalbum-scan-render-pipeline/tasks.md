## 1. Page Stitch & Render

- [x] 1.1 Remove `_apply_archive_ctm_if_present` calls from `stitch()`, `tif_to_jpg()`, and `derived_to_jpg()` in `stitch_oversized_pages.py`
- [x] 1.2 Confirm partial-panorama case raises with correct message (not silently producing a cropped output)
- [x] 1.3 Confirm derived-image quality reduction loop stops at quality 40 and does not loop indefinitely
- [x] 1.4 Confirm missing S01 scan raises before any render attempt
- [x] 1.5 Add a per-page lock helper used by `render-pipeline`; acquire before page work begins and release on both success and failure

## 2. Pipeline State Helpers

- [x] 2.1 Add `read_pipeline_state(xmp_path) -> dict` to `xmp_sidecar.py`: reads the `pipeline` subkey from `imago:Detections` JSON; returns `{}` if absent
- [x] 2.2 Add `write_pipeline_step(xmp_path, step, *, model=None, extra=None)` to `xmp_sidecar.py`: merges `{step: {"completed": <now>, "model": model, ...extra}}` into the `pipeline` subkey of `imago:Detections` without touching other keys in the blob
- [x] 2.3 Add `clear_pipeline_steps(xmp_path, steps)` to `xmp_sidecar.py`: removes the listed step keys from the `pipeline` subkey (used by `--force`)
- [x] 2.4 Add unit tests: `write_pipeline_step` preserves existing `location` and other `imago:Detections` keys; `extra` fields round-trip; `clear_pipeline_steps` removes only named keys; `read_pipeline_state` returns `{}` on missing field

## 3. CTM Colour Restoration

- [x] 3.1 Confirm `ctm review` returns `"ctm": null` (not an error) when no CTM is stored in the archive XMP
- [x] 3.2 Add pipeline state check to `run_ctm generate`: skip if `pipeline.ctm` is present in archive sidecar `imago:Detections` and `--force` is not set; print skip message with recorded timestamp
- [x] 3.3 Add pipeline state write to `run_ctm generate`: call `write_pipeline_step(archive_xmp, "ctm", model=model_name)` on success
- [x] 3.4 Add `--force` handling for `ctm generate`: call `clear_pipeline_steps(archive_xmp, ["ctm"])` before regenerating
- [x] 3.5 Add `apply_ctm_to_jpeg(jpeg_path, matrix)` function in `ai_ctm_restoration.py`: reads JPEG, applies 3x3 matrix, writes corrected pixels back in place
- [x] 3.6 Add `--per-photo` flag to `run_ctm` in `commands.py`: when set, iterates crop JPEGs in `_Photos/` instead of archive scans; stores result in each crop's XMP sidecar; checks/writes `pipeline.ctm` state on the crop sidecar
- [x] 3.7 Add `run_ctm_apply(*, album_id, photos_root, page, force)` to `commands.py`: applies page CTM (`crs:ColorMatrix1` from archive XMP) to `_V.jpg`; applies per-photo CTM (`crs:ColorMatrix1` from each crop XMP) to each crop in `_Photos/`; records `pipeline.ctm_applied` per file; skips silently if no CTM is stored for a given file
- [x] 3.8 Add `ctm-apply` standalone CLI subcommand in `photoalbums.py`
- [x] 3.9 Add unit tests for `apply_ctm_to_jpeg` (identity matrix leaves pixels unchanged; known matrix produces correct output)
- [x] 3.10 Add unit test for `run_ctm_apply`: page CTM applied to `_V.jpg`; per-photo CTM applied to crop; file with no CTM skipped; pipeline-state skip on second call without `--force`

## 4. Region Detection

- [x] 4.1 Add pipeline state check to `run_detect_view_regions`: skip if `pipeline.view_regions` is present in view sidecar `imago:Detections` and `--force` is not set; print skip message
- [x] 4.2 Add pipeline state write to `run_detect_view_regions`: call `write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})` on success with regions, and `extra={"result": "no_regions"}` on success with zero regions
- [x] 4.3 Add `--force` handling: call `clear_pipeline_steps(xmp_path, ["view_regions"])` before re-detecting
- [x] 4.4 Add `person_names: list[str]` field to `RegionResult` dataclass in `ai_view_regions.py`
- [x] 4.5 Add `person_names` array to `_REGION_RESPONSE_FORMAT` JSON schema (alongside existing `caption_hint`); update `_parse_region_response` to populate `RegionResult.person_names`
- [x] 4.6 Add `album_context`, `page_caption`, and `people_roster` optional parameters to `detect_regions` and `_call_vision_model`; include all three in the prompt when non-empty so the model can expand hyphenated name shorthand using the roster and identify people from caption context
- [x] 4.7 Add `read_people_roster(album_set) -> dict[str, str]` to `common.py` (or `album_sets.py`): reads `[sets.<album_set>.people]` from `album_sets.toml`, filters out entries with empty values, returns the remainder; returns `{}` if no table is present
- [x] 4.8 Update `run_detect_view_regions` in `commands.py` to read `dc:description` from the view sidecar and pass it as `page_caption`; derive `album_context` from `parse_album_filename`; load `people_roster` via `read_people_roster`
- [x] 4.9 Add result validation before XMP write: reject zero-area / negative-area / degenerate tiny boxes and resolve heavy overlaps deterministically
- [x] 4.10 Ensure pipeline invocation of region detection is limited to page `_V.jpg` images and never runs on `_D##-##_V.jpg` derived outputs
- [x] 4.11 Add unit test: `_parse_region_response` with `person_names` in JSON -> `RegionResult.person_names` populated; missing `person_names` key -> empty list
- [x] 4.12 Add unit test: `read_people_roster` returns filtered dict (empty values excluded); missing table returns `{}`
- [x] 4.13 Add unit test: `detect_regions` passes `album_context`, `page_caption`, and `people_roster` into the prompt payload
- [x] 4.14 Add unit tests for empty-result success (`P01` / title page behavior), invalid zero-area boxes rejected, and overlapping regions resolved before XMP write

## 5. Render-Time Face Refresh

- [x] 5.1 Add a narrow `refresh_face_regions(image_path, sidecar_path, *, force=False)` entrypoint in a new `ai_render_face_refresh.py`: checks `pipeline.face_refresh` state, loads Cast store, runs buffalo_l, replaces only face-type `ImageRegion` entries, collects identified person names and writes them to `Iptc4xmpExt:PersonInImage` (replacing previous values), writes `pipeline.face_refresh` state on success
- [x] 5.2 Implement face-only `ImageRegion` replacement in `xmp_sidecar.py`: remove existing entries whose `Iptc4xmpExt:RCtype` is a face type, write new ones, preserve all non-face `ImageRegion` entries and unrelated XMP fields
- [x] 5.3 Add `face-refresh` standalone CLI subcommand in `photoalbums.py` -> `run_face_refresh` in `commands.py`
- [x] 5.4 Structure the command so Cast/model state is loaded once and files are processed sequentially to bound memory during album runs with many crops
- [x] 5.5 Add unit tests for face-only replacement: mixed face + non-face -> only face entries replaced; all-non-face -> unchanged
- [x] 5.6 Add unit test for Cast unavailable: `refresh_face_regions` does not write pipeline state and sidecar is unchanged
- [x] 5.7 Add unit test for pipeline-state skip: second call without `--force` prints skip message and does not re-run buffalo_l
- [x] 5.8 Add unit test for `PersonInImage` write: two identified faces -> names in bag; no Cast matches -> bag cleared

## 6. xmpMM Provenance Metadata

- [x] 6.1 Create `photoalbums/lib/xmpmm_provenance.py` with `assign_document_id(xmp_path) -> str`: writes `xmpMM:DocumentID` as `xmp:uuid:{uuid4}` if not already present; returns the new or existing value
- [x] 6.2 Add `write_derived_from(xmp_path, source_document_id, source_relative_path)` to `xmpmm_provenance.py`
- [x] 6.3 Add `write_pantry_entry(xmp_path, document_id, relative_path)` to `xmpmm_provenance.py`: appends to `xmpMM:Pantry` bag, deduplicating by `documentID`
- [x] 6.4 Add `write_creation_provenance(xmp_path, *, derived_from, pantry_sources)` helper in `xmpmm_provenance.py`: updates `DerivedFrom` and `Pantry` in place while preserving unrelated sidecar fields
- [x] 6.5 Call `assign_document_id` on the archive scan sidecar inside `_ensure_archive_page_sidecar` in `stitch_oversized_pages.py`
- [x] 6.6 Call `assign_document_id` and `write_creation_provenance` on the rendered JPEG sidecar immediately after each JPEG is written in `tif_to_jpg`, `derived_to_jpg`, and `stitch`
- [x] 6.7 For stitched page `_V.jpg` outputs, write `DerivedFrom` pointing to `_S01.tif` and add one `Pantry` entry per contributing scan
- [x] 6.8 For crop sidecars, write `DocumentID`, `DerivedFrom`, and `Pantry` when the crop sidecar is created from the page `_V.jpg`
- [x] 6.9 Add unit tests for `assign_document_id` (idempotent), `write_derived_from` (overwrite on second call), `write_pantry_entry` (deduplication), stitched-output multi-source pantry writes, and preservation of unrelated XMP fields

## 7. Pipeline Command

- [x] 7.1 Add `run_render_pipeline(*, album_id, photos_root, page, force, skip_crops)` to `photoalbums/commands.py`: calls the same underlying functions as the standalone commands in order - render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply
- [x] 7.2 Wire `render-pipeline` subcommand in `photoalbums.py` CLI
- [x] 7.3 Add `photoalbums-render-pipeline` recipe to `justfile`
- [x] 7.4 Per-page error handling: catch and print exceptions per page including underlying error, do not write pipeline state for failed steps, continue to next page, and collect page/step failures for final reporting
- [x] 7.5 Add end-of-run failure summary output listing every failed page and the step that failed; exit non-zero if any page failed

## 8. Tests

- [x] 8.1 Add integration test for `run_render_pipeline`: verify pipeline state entries written for render-adjacent AI steps, render output exists, regions XMP present, face regions refreshed, CTM applied, and provenance exists immediately on created files
- [x] 8.2 Add test: second pipeline run without `--force` skips all previously completed steps and prints skip messages
- [x] 8.3 Add test: `--force` clears all pipeline states and re-runs all applicable steps
- [x] 8.4 Add test: title page with no detected regions records explicit no-regions state, writes no crops, and does not fail the pipeline
- [x] 8.5 Add test: later-step failure still leaves creation-time provenance on files that were already created before the failure
