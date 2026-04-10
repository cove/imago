## 1. Page Stitch & Render (update existing implementation)

- [ ] 1.1 Remove `_apply_archive_ctm_if_present` calls from `stitch()`, `tif_to_jpg()`, and `derived_to_jpg()` in `stitch_oversized_pages.py`
- [ ] 1.2 Confirm partial-panorama case raises with correct message (not silently producing a cropped output)
- [ ] 1.3 Confirm derived-image quality reduction loop stops at quality 40 and does not loop indefinitely
- [ ] 1.4 Confirm missing S01 scan raises before any render attempt

## 2. Pipeline State Helpers

- [ ] 2.1 Add `read_pipeline_state(xmp_path) -> dict` to `xmp_sidecar.py`: reads the `pipeline` subkey from `imago:Detections` JSON; returns `{}` if absent
- [ ] 2.2 Add `write_pipeline_step(xmp_path, step, *, model=None)` to `xmp_sidecar.py`: merges `{step: {"completed": <now>, "model": model}}` into the `pipeline` subkey of `imago:Detections` without touching other keys in the blob
- [ ] 2.3 Add `clear_pipeline_steps(xmp_path, steps)` to `xmp_sidecar.py`: removes the listed step keys from the `pipeline` subkey (used by `--force`)
- [ ] 2.4 Add unit tests: `write_pipeline_step` preserves existing `location` and other `imago:Detections` keys; `clear_pipeline_steps` removes only named keys; `read_pipeline_state` returns `{}` on missing field

## 3. CTM Colour Restoration (update existing implementation + add ctm-apply)

- [ ] 3.1 Confirm `ctm review` returns `"ctm": null` (not an error) when no CTM is stored in the archive XMP
- [ ] 3.2 Add pipeline state check to `run_ctm generate`: skip if `pipeline.ctm` present in archive sidecar `imago:Detections` and `--force` not set; print skip message with recorded timestamp
- [ ] 3.3 Add pipeline state write to `run_ctm generate`: call `write_pipeline_step(archive_xmp, "ctm", model=model_name)` on success
- [ ] 3.4 Add `--force` handling for `ctm generate`: call `clear_pipeline_steps(archive_xmp, ["ctm"])` before regenerating
- [ ] 3.5 Add `apply_ctm_to_jpeg(jpeg_path, matrix)` function in `ai_ctm_restoration.py`: reads JPEG, applies 3×3 matrix, writes corrected pixels back in-place
- [ ] 3.6 Add `--per-photo` flag to `run_ctm` in `commands.py`: when set, iterates crop JPEGs in `_Photos/` instead of archive scans; stores result in each crop's XMP sidecar; checks/writes `pipeline.ctm` state on the crop sidecar
- [ ] 3.7 Add `run_ctm_apply(*, album_id, photos_root, page, force)` to `commands.py`: applies page CTM (`crs:ColorMatrix1` from archive XMP) to `_V.jpg`; applies per-photo CTM (`crs:ColorMatrix1` from each crop XMP) to each crop in `_Photos/`; both in a single pass; checks/writes `pipeline.ctm_applied` per file; skips silently if no CTM stored for a given file
- [ ] 3.8 Add `ctm-apply` standalone CLI subcommand in `photoalbums.py`
- [ ] 3.9 Add unit tests for `apply_ctm_to_jpeg` (identity matrix leaves pixels unchanged; known matrix produces correct output)
- [ ] 3.10 Add unit test for `run_ctm_apply`: page CTM applied to `_V.jpg`; per-photo CTM applied to crop; file with no CTM skipped; pipeline state skip on second call without `--force`

## 4. Region Detection (update existing implementation)

- [ ] 4.1 Add pipeline state check to `run_detect_view_regions`: skip if `pipeline.view_regions` present in view sidecar `imago:Detections` and `--force` not set; print skip message
- [ ] 4.2 Add pipeline state write to `run_detect_view_regions`: call `write_pipeline_step(xmp_path, "view_regions", model=model_name)` on success
- [ ] 4.3 Add `--force` handling: call `clear_pipeline_steps(xmp_path, ["view_regions"])` before re-detecting
- [ ] 4.4 Add `person_names: list[str]` field to `RegionResult` dataclass in `ai_view_regions.py`
- [ ] 4.5 Add `person_names` array to `_REGION_RESPONSE_FORMAT` JSON schema (alongside existing `caption_hint`); update `_parse_region_response` to populate `RegionResult.person_names`
- [ ] 4.6 Add `album_context`, `page_caption`, and `people_roster` optional parameters to `detect_regions` and `_call_vision_model`; include all three in the system prompt when non-empty so the model can expand hyphenated name shorthand using the roster and identify people from caption context
- [ ] 4.7 Add `read_people_roster(album_set) -> dict[str, str]` to `common.py` (or `album_sets.py`): reads `[sets.<album_set>.people]` from `album_sets.toml`, filters out entries with empty values, returns the remainder; returns `{}` if no table present
- [ ] 4.8 (was 4.7) Update `run_detect_view_regions` in `commands.py` to read `dc:description` from the view sidecar and pass it as `page_caption`; derive `album_context` from `parse_album_filename`; load `people_roster` via `read_people_roster`
- [ ] 4.9 Add unit test: `_parse_region_response` with `person_names` in JSON → `RegionResult.person_names` populated; missing `person_names` key → empty list
- [ ] 4.10 Add unit test: `read_people_roster` returns filtered dict (empty values excluded); missing table returns `{}`
- [ ] 4.11 Add unit test: `detect_regions` passes `album_context`, `page_caption`, and `people_roster` into the prompt payload

## 5. Render-Time Face Refresh

- [ ] 5.1 Add a narrow `refresh_face_regions(image_path, sidecar_path, *, force=False)` entrypoint in a new `ai_render_face_refresh.py`: checks `pipeline.face_refresh` state, loads Cast store, runs buffalo_l, replaces only face-type `ImageRegion` entries, collects identified person names and writes them to `Iptc4xmpExt:PersonInImage` (replacing previous values), writes `pipeline.face_refresh` state on success
- [ ] 5.2 Implement face-only `ImageRegion` replacement in `xmp_sidecar.py`: remove existing entries whose `Iptc4xmpExt:RCtype` is a face type, write new ones, preserve all non-face `ImageRegion` entries
- [ ] 5.3 Add `face-refresh` standalone CLI subcommand in `photoalbums.py` → `run_face_refresh` in `commands.py`
- [ ] 5.4 Add unit tests for face-only replacement: mixed face + non-face → only face entries replaced; all-non-face → unchanged
- [ ] 5.5 Add unit test for Cast unavailable: `refresh_face_regions` does not write pipeline state and sidecar is unchanged
- [ ] 5.6 Add unit test for pipeline state skip: second call without `--force` prints skip message and does not re-run buffalo_l
- [ ] 5.7 Add unit test for `PersonInImage` write: two identified faces → names in bag; no Cast matches → bag cleared

## 6. xmpMM Provenance Metadata

- [ ] 6.1 Create `photoalbums/lib/xmpmm_provenance.py` with `assign_document_id(xmp_path) -> str`: writes `xmpMM:DocumentID` as `xmp:uuid:{uuid4}` if not already present; returns the (new or existing) value
- [ ] 6.2 Add `write_derived_from(xmp_path, source_document_id, source_relative_path)` to `xmpmm_provenance.py`
- [ ] 6.3 Add `write_pantry_entry(xmp_path, document_id, relative_path)` to `xmpmm_provenance.py`: appends to `xmpMM:Pantry` bag, deduplicating by `documentID`
- [ ] 6.4 Add `write_provenance(rendered_sidecar, archive_sidecar, *, force=False)` to `xmpmm_provenance.py`: checks `pipeline.provenance` state, writes DerivedFrom + Pantry, writes `pipeline.provenance` state on success
- [ ] 6.5 Add `write-provenance` standalone CLI subcommand in `photoalbums.py` → `run_write_provenance` in `commands.py`
- [ ] 6.6 Call `assign_document_id` on the archive scan sidecar inside `_ensure_archive_page_sidecar` in `stitch_oversized_pages.py`
- [ ] 6.7 Call `assign_document_id` on the rendered JPEG sidecar immediately after each JPEG is written in `tif_to_jpg`, `derived_to_jpg`, and `stitch`
- [ ] 6.8 Add unit tests for `assign_document_id` (idempotent), `write_derived_from` (overwrite on second call), `write_pantry_entry` (deduplication), pipeline state skip

## 7. Pipeline Command

- [ ] 7.1 Add `run_render_pipeline(*, album_id, photos_root, page, force)` to `photoalbums/commands.py`: calls the same underlying functions as the standalone commands in order — render → ctm-apply → detect-regions → face-refresh → write-provenance
- [ ] 7.2 Wire `render-pipeline` subcommand in `photoalbums.py` CLI
- [ ] 7.3 Add `photoalbums-render-pipeline` recipe to `justfile`
- [ ] 7.4 Per-page error handling: catch and print exceptions per page including underlying error, do not write pipeline state for failed steps, continue to next page; exit non-zero if any page had an error

## 8. Tests

- [ ] 8.1 Add integration test for `run_render_pipeline`: verify pipeline state entries written for all steps, render output exists, regions XMP present, face regions refreshed, provenance fields set
- [ ] 8.2 Add test: second pipeline run without `--force` skips all AI steps (all pipeline states already recorded) and prints skip messages
- [ ] 8.3 Add test: `--force` clears all pipeline states and re-runs all steps
