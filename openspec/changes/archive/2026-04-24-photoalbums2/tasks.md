## 1. Project skeleton

- [x] 1.1 Create `photoalbums2/` directory with `pyproject.toml`, `README.md`, and `__init__.py`
- [x] 1.2 Add Dagster, Pydantic, Pydantic AI (or `instructor` + `litellm`), Jinja2, Streamlit, and Pillow as `photoalbums2` project dependencies
- [x] 1.3 Add `photoalbums2/config.py` with album root, output mode (`dry-run`, `staging`, `promote`), debug output root `_debug/photoalbums2/`, provider selection, default sampler parameters, and concurrency limit defaulting to 1
- [x] 1.4 Confirm `photoalbums2` can import `photoalbums.lib.xmp_sidecar` and `cast.storage` read-only from the monorepo
- [x] 1.5 Decide which legacy helper code is imported read-only versus copied into `photoalbums2`; do not copy legacy orchestration modules wholesale

## 2. Work-unit identity and bootstrap import

- [x] 2.1 Define page work-unit keys as album/book/page identity
- [x] 2.2 Define photo work-unit keys as album/book/page/crop identity
- [x] 2.3 Implement discovery for page work units from archive/view folders
- [x] 2.4 Implement discovery for photo work units from crop outputs
- [x] 2.5 Implement a bootstrap/import or observe path that records existing canonical `.xmp` sidecars as starting materializations using deterministic content/input hashes
- [x] 2.6 Add tests for page/photo key parsing and legacy sidecar import without LLM calls

## 3. Prompt and schema files

- [x] 3.1 Create `photoalbums2/prompts/ocr.py` with system prompt, Jinja user template, Pydantic output schema, and retry ladder
- [x] 3.2 Create `photoalbums2/prompts/caption.py`
- [x] 3.3 Create `photoalbums2/prompts/location.py`
- [x] 3.4 Create `photoalbums2/prompts/date.py`
- [x] 3.5 Create `photoalbums2/prompts/semantic_review.py`
- [x] 3.6 Create `photoalbums2/prompts/people.py` for the Cast-embedding match step shape, even though it does not call an LLM
- [x] 3.7 Ensure no `photoalbums2` runtime code reads `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`
- [x] 3.8 Ensure the location prompt/schema produces both a primary GPS query and named shown-location queries for the split `location_queries` asset

## 4. Retry and validation

- [x] 4.1 Implement technical failure classification for provider connection errors, timeouts, HTTP 5xx, invalid response envelopes, file locks, and process crashes so Dagster retry handles them
- [x] 4.2 Implement conservative static validators for empty response, incomplete JSON, invalid JSON, schema mismatch, impossible date format, unexpected double quotes inside caption text, prompt echo, obvious truncation, and known duplicate-character spam
- [x] 4.3 Implement the AI retry/rewrite runner that invokes the next ladder rung or rewrite instruction on static validation failure
- [x] 4.4 Record every AI attempt (prompt, response, parameters, validator result, failure reason) to Dagster metadata
- [x] 4.5 Confirm no candidate XMP is written when all AI retry rungs fail
- [x] 4.6 Add focused tests for static validation, technical retry classification, and retry metadata

## 5. Dagster asset graph

- [x] 5.1 Implement `render_settings` as a non-partitioned observable/input asset
- [x] 5.2 Implement page assets: `stitch`, `regions`, and `crops`
- [x] 5.3 Implement photo assets: `people`, `ocr`, `caption`, `location`, `date`, and `semantic_review`
- [x] 5.4 Wire dependencies so `render_settings` can stale `stitch[page]`, and page output changes can stale downstream photo work units
- [x] 5.5 Configure asset-level Dagster retry policy for technical failures
- [x] 5.6 Confirm the graph can materialize one page and one photo serially with concurrency 1
- [x] 5.7 Split the existing photo-level `location` asset into explicit `location_queries`, `gps_location`, and `locations_shown` assets so GPS and IPTC LocationShown appear in the Dagster DAG
- [x] 5.8 Ensure `gps_location` writes primary EXIF GPS/scalar location candidate metadata
- [x] 5.9 Ensure `locations_shown` writes candidate IPTC `LocationShown` rows with optional Nominatim GPS/source metadata
- [x] 5.10 Record location-query and geocoder query/result metadata for both `gps_location` and `locations_shown`

## 6. Selective Cast face refresh

- [x] 6.1 Represent a newly recognized Cast face as a targeted invalidation event
- [x] 6.2 Implement plausible-match lookup from the newly recognized face embedding to existing crop/photo embeddings
- [x] 6.3 Mark only matching `people[photo]` work units stale
- [x] 6.4 Confirm `stitch`, `regions`, `crops`, `ocr`, and unrelated `people[photo]` work units remain fresh after one new face is recognized
- [x] 6.5 Confirm downstream `caption`, `location`, `date`, and `semantic_review` rerun only when `people[photo]` output actually changes
- [x] 6.6 Update face-change downstream invalidation to include `location_queries`, `gps_location`, and `locations_shown` after the location asset split

## 7. Output modes and XMP writing

- [x] 7.1 Implement dry-run output mode writing candidate XMP and diagnostics under `_debug/photoalbums2/`
- [x] 7.2 Implement staging output mode writing adjacent `<stem>.xmp.new`
- [x] 7.3 Implement explicit promote mode/action that can copy reviewed staged output to canonical `.xmp` atomically
- [x] 7.4 Render a candidate-vs-canonical XMP diff for Dagster metadata and the review UI
- [x] 7.5 Add tests that dry-run and staging never modify canonical `.xmp`

## 8. Vertical slice

- [x] 8.1 Run one page through `stitch`, `regions`, and `crops` in dry-run mode
- [x] 8.2 Run one photo through `people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` in dry-run mode
- [x] 8.3 Compare debug candidate XMP to existing canonical `.xmp` on overlapping fields and document deltas
- [ ] 8.4 Run a 50-100 photo slice serially by default
- [ ] 8.5 Capture per-step timings, retry-ladder usage statistics, and `needs_review` counts for the slice

## 9. Streamlit review UI

- [x] 9.1 Create `photoalbums2/ui/app.py` reading Dagster metadata/event log
- [x] 9.2 Render a photo-picker sidebar listing photos in the current slice
- [x] 9.3 For the selected photo, render page image, crop image, prompt sent, response received, sampler settings, and retry/validation history per step
- [x] 9.4 Render candidate-vs-canonical XMP diff
- [x] 9.5 Provide a rerun button for a single selected work unit
- [x] 9.6 Optionally provide a prompt editing/draft workflow that keeps prompt changes version-controlled

## 10. Documentation and hand-off

- [x] 10.1 Write `photoalbums2/README.md` covering setup, `dagster dev --host 0.0.0.0`, `streamlit run`, output modes, concurrency limit, and provider swap
- [x] 10.2 Document the parallel stance: `photoalbums/` still owns canonical `.xmp` until an explicit promote action and later cutover proposal
- [x] 10.3 Document which legacy code was imported versus copied and why
- [x] 10.4 Note open questions resolved during implementation in the design file so the later cutover proposal can reference them
