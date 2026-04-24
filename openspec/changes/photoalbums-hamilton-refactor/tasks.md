## 1. Setup & Dependencies

- [ ] 1.1 Add `hamilton>=1.23.0` to `pyproject.toml` dependencies
- [ ] 1.2 Create `photoalbums3/` package directory at `/home/user/imago/photoalbums3/`
- [ ] 1.3 Create `photoalbums3/__init__.py` with package exports
- [ ] 1.4 Create `photoalbums3/steps/` subdirectory for step modules
- [ ] 1.5 Verify `uv sync` resolves hamilton correctly
- [ ] 1.6 Run a minimal Hamilton test: `python -c "import hamilton; print(hamilton.__version__)"`

## 2. Core Runner & DAG Builder

- [ ] 2.1 Create `photoalbums3/runner.py` with `run_pipeline()` function signature matching `photoalbums/commands.py::run_process_pipeline()`
- [ ] 2.2 Implement page iteration loop: `for page_group in iter_pages(...): dr.execute()`
- [ ] 2.3 Implement error handling and exit code logic matching existing behavior
- [ ] 2.4 Implement progress reporting (print per-page, per-step status)
- [ ] 2.5 Create `photoalbums3/pipeline.py` as the main module for Hamilton to import steps from
- [ ] 2.6 Add counter tracking dict (match existing "run"/"skipped"/"failed" metrics)

## 3. Step Implementations

### 3.1 Render Step
- [ ] 3.1.1 Create `photoalbums3/steps/render.py`
- [ ] 3.1.2 Extract `stitch_oversized_pages` logic into `render_page(archive, view_dir, photos_dir, page_group, force)` function
- [ ] 3.1.3 Return dict with `{"view_path", "xmp_path", "status"}` 
- [ ] 3.1.4 Implement staleness check: skip if view JPEG is fresh and xmp exists
- [ ] 3.1.5 Write unit test: successful render, skip existing, handle missing archive

### 3.2 Propagate Metadata Step
- [ ] 3.2.1 Create `photoalbums3/steps/propagate_metadata.py`
- [ ] 3.2.2 Implement `propagate_metadata(render_page: dict, view_dir: Path, ...)` depending on render_page
- [ ] 3.2.3 Copy archive XMP fields to page sidecar
- [ ] 3.2.4 Return dict with `{"status": "done"|"skipped"}`
- [ ] 3.2.5 Implement staleness: skip if page xmp is fresh and archive sidecar hasn't changed
- [ ] 3.2.6 Write unit test: xmp fields copied, skip if unchanged

### 3.3 Detect Regions Step
- [ ] 3.3.1 Create `photoalbums3/steps/detect_regions.py`
- [ ] 3.3.2 Implement `detect_regions(render_page: dict, propagate_metadata: dict, ...)` 
- [ ] 3.3.3 Extract region detection logic from `ai_view_regions.detect_regions()`
- [ ] 3.3.4 Write XMP region metadata
- [ ] 3.3.5 Return dict with `{"regions": list, "status": "detected"|"failed"}`
- [ ] 3.3.6 Implement staleness: skip if xmp regions are fresh and upstream steps haven't rerun
- [ ] 3.3.7 Write unit test: regions detected, xmp written, skip on staleness

### 3.4 Crop Regions Step
- [ ] 3.4.1 Create `photoalbums3/steps/crop_regions.py`
- [ ] 3.4.2 Implement `crop_regions(detect_regions: dict, view_dir: Path, photos_dir: Path, ...)` 
- [ ] 3.4.3 Extract crop logic from `ai_photo_crops.crop_page_regions()`
- [ ] 3.4.4 Return dict with `{"crops_written": int, "status": "done"|"skipped"}`
- [ ] 3.4.5 Implement staleness: skip if crops on disk match xmp regions
- [ ] 3.4.6 Write unit test: crops extracted, disk state matches xmp

### 3.5 Face Refresh Step
- [ ] 3.5.1 Create `photoalbums3/steps/face_refresh.py`
- [ ] 3.5.2 Implement `face_refresh(crop_regions: dict, view_dir: Path, photos_dir: Path, ...)`
- [ ] 3.5.3 Extract face refresh logic from `ai_render_face_refresh.RenderFaceRefreshSession`
- [ ] 3.5.4 Update face regions on all rendered outputs
- [ ] 3.5.5 Return dict with `{"faces_updated": int, "status": "done"|"skipped"}`
- [ ] 3.5.6 Implement staleness: skip if cast store hash hasn't changed
- [ ] 3.5.7 Write unit test: face regions updated, staleness checks work

### 3.6 AI Index Step
- [ ] 3.6.1 Create `photoalbums3/steps/ai_index.py`
- [ ] 3.6.2 Implement `ai_index(crop_regions: dict, view_dir: Path, photos_dir: Path, ...)` 
- [ ] 3.6.3 Delegate to existing `photoalbums/lib/ai_index_runner.IndexRunner` (no reimplementation)
- [ ] 3.6.4 Return dict with `{"status": "indexed"|"skipped"}`
- [ ] 3.6.5 Implement staleness: reuse existing staleness logic from IndexRunner
- [ ] 3.6.6 Write unit test: integration with IndexRunner, staleness passed through

### 3.7 Verify Crops Step
- [ ] 3.7.1 Create `photoalbums3/steps/verify_crops.py`
- [ ] 3.7.2 Implement `verify_crops(ai_index: dict, view_dir: Path, photos_dir: Path, ...)` as final step
- [ ] 3.7.3 Extract verification logic from `ai_verify_crops.run_verify_crops_page()`
- [ ] 3.7.4 Return dict with `{"verified": bool, "status": "done"|"skipped"}`
- [ ] 3.7.5 Implement staleness: skip if all upstream steps were skipped
- [ ] 3.7.6 Write unit test: verification logic works, staleness cascades

## 4. Hamilton Integration

- [ ] 4.1 In `pipeline.py`: use `@hamilton.config` to conditionally skip steps based on CLI flags (`--skip`, `--redo`)
- [ ] 4.2 Implement `skip_ids` and `redo_ids` handling: wrap step functions to force execution based on flags
- [ ] 4.3 Test DAG resolution: verify Hamilton correctly orders steps based on function parameters
- [ ] 4.4 Test skip/redo: verify `--skip render` skips render and forces downstream to skip
- [ ] 4.5 Test forced re-run: verify `--redo ai-index` forces ai-index and downstream to rerun

## 5. CLI Integration

- [ ] 5.1 Update `photoalbums/cli.py` render subcommand: add `--use-hamilton` flag
- [ ] 5.2 Update render command handler: check flag; if true, call `photoalbums3.runner.run_pipeline()` instead of `commands.run_process_pipeline()`
- [ ] 5.3 Pass all CLI arguments through to runner (album_id, page, skip_ids, redo_ids, force, debug, etc.)
- [ ] 5.4 Test CLI: `python photoalbums.py render --use-hamilton --album "Cordell_2020"` works
- [ ] 5.5 Test backward compat: `python photoalbums.py render --album "Cordell_2020"` (no flag) still uses old code

## 6. Testing

- [ ] 6.1 Create `photoalbums3/tests/` directory with conftest and test utilities
- [ ] 6.2 Write unit tests for each step function (using mocked inputs from upstream)
- [ ] 6.3 Write integration test: run full DAG on a sample page, verify all outputs match existing pipeline
- [ ] 6.4 Write comparison test: run old pipeline vs. Hamilton pipeline on same data, verify identical XMP sidecars
- [ ] 6.5 Test edge cases: missing archive, corrupt xmp, missing photos_dir
- [ ] 6.6 Run `uv run pytest photoalbums3/` — all tests pass
- [ ] 6.7 Run `uv run python scripts/check_pyright.py` for type checking on photoalbums3/

## 7. Documentation & Examples

- [ ] 7.1 Add `## photoalbums3` section to README explaining Hamilton-based pipeline
- [ ] 7.2 Document CLI flag: `--use-hamilton` is experimental, feedback welcome
- [ ] 7.3 Document step architecture: how to add new steps, how dependency resolution works
- [ ] 7.4 Create example: "Running a single step with photoalbums3"
- [ ] 7.5 Create troubleshooting guide: common Hamilton DAG resolution issues

## 8. Code Quality & Cleanup

- [ ] 8.1 Run `uv run python scripts/check_ruff.py photoalbums3/` — no lint errors
- [ ] 8.2 Run `uv run skylos photoalbums3/` — no duplicate code
- [ ] 8.3 Run `uv run radon cc photoalbums3/ -a` — complexity <10 per function
- [ ] 8.4 Ensure xmp_sidecar.py split is not a blocker (can coexist with monolithic version)
- [ ] 8.5 Verify no new dependencies beyond hamilton

## 9. End-to-End Validation

- [ ] 9.1 Pick a test album (e.g., Cordell_2020, small, <10 pages)
- [ ] 9.2 Run old pipeline: `python photoalbums.py render --album Cordell_2020` — verify complete
- [ ] 9.3 Run new pipeline: `python photoalbums.py render --use-hamilton --album Cordell_2020` on same data
- [ ] 9.4 Compare outputs: diff all page XMP sidecars, verify identical
- [ ] 9.5 Spot-check: verify view JPEGs are byte-identical (same render quality, etc.)
- [ ] 9.6 Test skip: rerun with `--use-hamilton` (no flag changes) — verify all steps skip
- [ ] 9.7 Test forced rerun: `--use-hamilton --redo ai-index` — verify ai-index + downstream rerun, others skip
- [ ] 9.8 Test single page: `--use-hamilton --album Cordell_2020 --page 2` — verify only page 2 processed

## 10. Finalization

- [ ] 10.1 Update AGENTS.md with photoalbums3 conventions (if any differences)
- [ ] 10.2 Create a migration guide: "When to use --use-hamilton" (currently: experimental, for testing)
- [ ] 10.3 Decide: keep photoalbums/ long-term or deprecate after validation?
- [ ] 10.4 Tag release notes with Hamilton integration
- [ ] 10.5 Close this change as complete
