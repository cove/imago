## 1. XMP Pipeline Step Schema Migration

- [x] 1.1 Update `read_pipeline_state` to normalise legacy `{"completed": ts}` entries to `{"timestamp": ts, "result": "ok", "input_hash": ""}` on read
- [x] 1.2 Add `write_pipeline_steps(xmp_path, updates)` that merges new-schema step records into `imago:Detections["pipeline"]` and preserves existing keys
- [x] 1.3 Add `migrate-pipeline-records` sub-command that rewrites all legacy entries across a directory tree in place
- [x] 1.4 Update `detect-regions` / docling path to write via `write_pipeline_steps` with the new schema
- [x] 1.5 Write tests: legacy entry normalised on read, empty input_hash treated as stale, migration pass rewrites correctly, new writes use new schema

## 2. Locations Step (consolidates GPS + named locations)

- [x] 2.1 Add caption engine method `generate_location_queries(image, caption_text) -> LocationQueryResult` returning a primary query string and a list of named location query strings; prompt SHALL describe both Nominatim free-form and structured query modes
- [x] 2.2 Implement the `locations` step: call `generate_location_queries`, resolve all queries via Nominatim, write results to `imago:Detections["location"]`, `["locations_shown"]`, and `["location_shown_ran"]`
- [x] 2.3 Remove `_resolve_location_metadata` / `estimate_location` lat/lon path and standalone `_resolve_locations_shown` call from `_run_image_analysis`
- [x] 2.4 Write tests: primary + named queries resolved; no primary query → location absent; Nominatim unavailable → names stored without coordinates

## 3. Remove Fast Paths

- [x] 3.1 Delete `_process_people_update` and route its trigger condition (`cast_store_signature_changed`) into the step graph as a forced-stale `people` step
- [x] 3.2 Delete `_process_gps_update` and route its trigger conditions (`missing_location_shown`, `location_shown_ai_gps_stale`) into the step graph as a forced-stale `locations` step
- [x] 3.3 Verify that files previously handled by the fast paths now route through `_process_full` + `StepRunner` and skip unchanged steps correctly

## 4. Propagate-to-Crops Step

- [x] 4.1 Implement `_run_propagate_to_crops(page_xmp_path, locations_output, people_output)` that reads MWG-RS regions from the page XMP, finds each corresponding crop file, and writes GPS + person names to each crop's XMP sidecar
- [x] 4.2 Record `ai-index/propagate-to-crops` in each crop's `imago:Detections["pipeline"]` after writing
- [x] 4.3 Declare the step in the step graph with `depends_on: ["locations", "people"]` and an input hash covering locations output hash + people output hash + sorted crop paths
- [x] 4.4 Write tests: two crops updated when people reruns; GPS written to all crops when locations reruns; page with no crops records ok with zero updates; step skipped when neither upstream reran

## 5. Step Graph Definition

- [x] 5.1 Create `photoalbums/lib/ai_index_steps.py` with `StepDef` dataclass (name, depends_on, input_hash_fn, output_keys)
- [x] 5.2 Declare all seven steps (`ocr`, `people`, `caption`, `locations`, `objects`, `date-estimate`, `propagate-to-crops`) with correct dependency edges
- [x] 5.3 Implement per-step `input_hash_fn` functions covering only the relevant settings and upstream output fields per the spec table; `ocr` hash includes scan group signature for multi-scan pages
- [x] 5.4 Write unit tests confirming dependency order and that changing caption model does not alter OCR hash

## 6. StepRunner

- [x] 6.1 Implement `StepRunner` class: evaluates staleness (hash diff OR upstream reran), skips if fresh, calls step sub-function if stale, sets `reran` flag
- [x] 6.2 Wire `StepRunner` into `_run_image_analysis` in `ai_index_runner.py`, replacing the sequential sub-function calls
- [x] 6.3 Ensure step records are only written to XMP after the final payload write succeeds (not on intermediate steps or crashes)
- [x] 6.4 Write tests: upstream-reran forces downstream stale, hash-match skips step and reuses prior XMP output

## 7. CLI Step Targeting

- [x] 7.1 Add `--steps <name>[,<name>]` argument to the `ai-index` CLI command in `cli.py` / `commands.py`
- [x] 7.2 Pass forced-stale step set into `StepRunner` so listed steps run unconditionally
- [x] 7.3 Confirm that forcing a step stale propagates staleness to its declared downstream dependents
- [x] 7.4 Write an integration test: `--steps caption` reruns caption and forces locations/date-estimate/propagate-to-crops stale while skipping ocr and people if hashes match

## 8. End-to-End Validation

- [ ] 8.1 Run `migrate-pipeline-records` on existing data; confirm all legacy entries are rewritten and sidecars without `"pipeline"` are untouched
- [ ] 8.2 Run `ai-index` on a sample page with crops; confirm all seven step records appear in the page XMP and `ai-index/propagate-to-crops` appears in each crop XMP
- [ ] 8.3 Run again with no settings changes; confirm all steps report skipped
- [ ] 8.4 Change cast store signature; confirm `people`, `caption`, `locations`, `date-estimate`, `propagate-to-crops` rerun and `ocr`, `objects` are skipped; confirm crop person names are updated
- [ ] 8.5 Verify existing sidecars (no `"pipeline"` key) gain full step records after one run
