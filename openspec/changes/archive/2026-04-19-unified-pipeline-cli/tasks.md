## 1. Pipeline Step Registry

- [x] 1.1 Create `photoalbums/lib/pipeline.py` and define `PipelineStep` dataclass with fields: `id`, `label`, `run_fn`, `skip_check_fn`, `depends_on`, `redo_clears`; this file is the single source of truth for pipeline structure
- [x] 1.2 Define the ordered `PIPELINE_STEPS` list with 6 steps: render, propagate-metadata, detect-regions, crop-regions, face-refresh, ai-index
- [x] 1.3 Add step id validation: resolve `--skip`/`--redo`/`--step` values against the registry and exit code 2 with the valid id list on unknown input

## 2. Process Subparser (`cli.py`)

- [x] 2.1 Add `process` subparser to `build_parser()` with args: `--photos-root` (required), `--album` (fragment, default `""`), `--page`, `--skip` (append, multiple), `--redo` (append, multiple), `--step` (single step id), `--force`; enforce `--step` and `--skip` are mutually exclusive
- [x] 2.2 Forward per-step flags under `process`: `--debug`, `--no-validation`, `--skip-restoration`, `--force-restoration`
- [x] 2.3 Wire `args.group == "process"` dispatch in `main()` to call `commands.run_process_pipeline(...)`

## 3. Step Plan Display

- [x] 3.1 Implement `print_pipeline_plan(steps, skip_ids, redo_ids, album_label, page_count)` that prints the numbered step list with annotations before any page processing begins
- [x] 3.2 Ensure skipped steps are annotated `(skipped: --skip <id>)` and redo steps are annotated `(redo forced)` in the plan output

## 4. Per-Step Progress Output

- [x] 4.1 Implement step progress header line format `[N/T] <step-id> <page-name>` printed at the start of each step per page
- [x] 4.2 Print outcome suffix `... done`, `... skipped (already complete)`, or `... ERROR` at end of each step per page

## 5. `run_process_pipeline` Orchestrator (`commands.py`)

- [x] 5.1 Implement `run_process_pipeline(*, album_id, photos_root, page, skip_ids, redo_ids, step_id, force, debug, no_validation, skip_restoration, force_restoration)` in `commands.py`; when `step_id` is set, derive `skip_ids` as all step ids except `step_id`
- [x] 5.2 Collect archive directories and page groups (reuse logic from `run_render_pipeline`)
- [x] 5.3 For each page, iterate active steps (steps not in `skip_ids`), applying `--redo` clear logic before calling each step's `run_fn`
- [x] 5.4 Accumulate per-step counters (pages_run, pages_skipped, pages_failed) for the summary
- [x] 5.5 Handle step-level exceptions without aborting the full run (log error, increment failed counter, continue to next page)
- [x] 5.6 Integrate `--force` as shorthand for `--redo` on all steps

## 6. Completion Summary

- [x] 6.1 After all pages print `===== PIPELINE SUMMARY =====` block with one row per active step showing step name, run count, skipped count, failed count, and a detail column (e.g. `4 crops written`)
- [x] 6.2 Return exit code 1 if any step recorded any page failures, else 0

## 7. Pipeline Steps Listing

- [x] 7.1 Add `--list-steps` flag to the `process` subparser that prints the step registry and exits 0
- [x] 7.2 Add `photoalbums-steps` justfile target invoking `photoalbums.py process --list-steps`

## 8. GPS-Only AI Step

- [x] 8.1 Add `--gps-only` flag to the `process` subparser; when set and `ai-index` is active, forward `--reprocess-mode=gps` to `run_ai_index`
- [x] 8.2 Rewrite `photoalbums-ai-gps` justfile target to call `photoalbums.py process --step ai-index --gps-only --photos-root <root> {{args}}`

## 9. Step Dependency Declarations

- [x] 9.1 `depends_on: list[str]` is already part of `PipelineStep` from task 1.1; declare all dependencies in `PIPELINE_STEPS` in `pipeline.py`: propagate-metadata→render, detect-regions→render, crop-regions→detect-regions, face-refresh→crop-regions, ai-index→crop-regions

## 10. Dependency-Based Staleness Check (`xmp_sidecar.py`)

- [x] 10.1 Implement `is_step_stale(step_name, depends_on, pipeline_state) -> bool` in `xmp_sidecar.py`: parses ISO-8601 `completed` values from `pipeline_state` dict and returns `True` if the step has no `completed` entry or any dependency's `completed` is newer
- [x] 10.2 No changes to `write_pipeline_step` — existing `completed` timestamp recording is sufficient

## 11. Orchestrator Staleness Integration

- [x] 11.1 In `run_process_pipeline`, after reading pipeline state per page, call `is_step_stale` for each active step before deciding to skip; if stale, re-run the step and log `(re-run: <dep> updated)` in the progress line
- [x] 11.2 Staleness check applies to all steps uniformly — no special-casing needed since the logic is purely timestamp comparison within `imago:Detections`

## 12. Justfile Update

- [x] 12.1 Add `photoalbums-process *args:` target invoking `photoalbums.py process --photos-root <root> {{args}}`
- [x] 12.2 Rewrite `photoalbums-detect-regions`, `photoalbums-crop-regions`, `photoalbums-render-pipeline`, and `photoalbums-render` to call `photoalbums.py process --step <step-id> --photos-root <root> {{args}}`
- [x] 12.3 Rewrite `photoalbums-ai` to call `photoalbums.py process --step ai-index --photos-root <root> {{args}}`
- [x] 12.4 Remove `photoalbums-ctm-generate`, `photoalbums-ctm-apply`, and `photoalbums-repair-crop-source` from the justfile entirely
- [x] 12.5 Retain `photoalbums-map`, `photoalbums-watch`, `photoalbums-render-validate` unchanged

## 13. Tests

- [x] 13.1 Add unit tests for step id validation (unknown id → exit 2, valid ids accepted; `--step` + `--skip` → exit 2)
- [x] 13.2 Add unit tests for `print_pipeline_plan` output format (skip annotations, redo annotations, gps-only annotation on ai-index)
- [x] 13.3 Add unit tests for `is_step_stale`: step current, dependency newer, step never run
- [x] 13.4 Add integration smoke test for `run_process_pipeline` with all steps skipped (verifies orchestrator runs without error on a minimal fixture)
