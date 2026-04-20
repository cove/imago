## ADDED Requirements

### Requirement: Process command entry point
The system SHALL expose a `process` subcommand on `photoalbums.py` accepting `--photos-root`, optional `--album` (fragment match), optional `--page`, zero or more `--skip <step-id>` and `--redo <step-id>` flags, an optional `--step <step-id>` flag (run exactly one step, skip all others), and `--force` (equivalent to `--redo` every step). `--step` and `--skip` are mutually exclusive. Unknown step ids in any flag SHALL cause exit code 2 with the valid step id list printed.

#### Scenario: Full run
- **WHEN** user runs `photoalbums.py process --photos-root <root>`
- **THEN** all pipeline steps execute in order for all matching albums and pages

#### Scenario: Skip a step
- **WHEN** user runs `photoalbums.py process --photos-root <root> --skip crop-regions`
- **THEN** crop-regions is skipped for all pages and the plan annotates it `(skipped: --skip crop-regions)`

#### Scenario: Redo a step
- **WHEN** user runs `photoalbums.py process --photos-root <root> --redo detect-regions`
- **THEN** detect-regions clears its pipeline state and reruns; all other steps use their normal skip logic

#### Scenario: Single-step mode
- **WHEN** user runs `photoalbums.py process --photos-root <root> --step detect-regions`
- **THEN** only detect-regions executes; all other steps are annotated `(skipped: --step)` in the plan

#### Scenario: Invalid step id or conflicting flags
- **WHEN** user supplies an unknown step id or both `--step` and `--skip`
- **THEN** the command exits code 2 with an error message and the valid step id list

### Requirement: Step plan and progress output
Before processing any page the system SHALL print a numbered plan of all steps with skip/redo annotations. As each step runs per page it SHALL print `[N/T] <step-id> <page-name> ... <outcome>`.

#### Scenario: Plan with mixed step states
- **WHEN** user runs with `--skip face-refresh --redo ai-index`
- **THEN** the plan shows all steps; `face-refresh` annotated `(skipped: --skip face-refresh)`, `ai-index` annotated `(redo forced)`

#### Scenario: Step outcomes
- **WHEN** a step completes, is skipped, or errors on a page
- **THEN** the progress line ends with `done`, `skipped (already complete)`, `(re-run: <dep> updated)`, or `ERROR` respectively

### Requirement: Completion summary
After all pages the system SHALL print `===== PIPELINE SUMMARY =====` with one row per active step showing pages run, skipped, and failed. Exit code SHALL be 1 if any step recorded failures.

#### Scenario: Partial failure
- **WHEN** one or more pages fail in a step
- **THEN** the summary row shows the failed count and the command exits 1

### Requirement: Step-level flags
All existing per-step flags SHALL be available under `process`: `--debug` (detect-regions, ai-index), `--no-validation` (detect-regions), `--skip-restoration` and `--force-restoration` (crop-regions), `--gps-only` (ai-index, forwards `--reprocess-mode=gps`).

#### Scenario: GPS-only AI reindex
- **WHEN** user runs `photoalbums.py process --photos-root <root> --step ai-index --gps-only`
- **THEN** only ai-index runs with GPS reprocessing mode

### Requirement: Pipeline step ordering
Steps SHALL execute in this fixed order:
1. `render` — stitch/convert archive scans to page view JPEGs
2. `propagate-metadata` — copy safe archive XMP fields to page sidecar
3. `detect-regions` — detect photo bounding boxes and write MWG-RS XMP regions
4. `crop-regions` — crop detected regions to `_Photos/` directory
5. `face-refresh` — update face region metadata on rendered outputs
6. `ai-index` — run AI pipeline (OCR, caption, GPS, XMP write)

### Requirement: Pipeline steps listing
`photoalbums.py process --list-steps` SHALL print the ordered step registry (number, id, description) and exit 0 without processing any files. A `photoalbums-steps` justfile target SHALL invoke it.

#### Scenario: Listing steps
- **WHEN** user runs `photoalbums.py process --list-steps`
- **THEN** all 6 steps are printed in order and the command exits 0

### Requirement: Step dependency declarations
Each `PipelineStep` in `photoalbums/lib/pipeline.py` SHALL declare `depends_on` step ids. The dependency graph is:
- `propagate-metadata` depends on `render`
- `detect-regions` depends on `render`
- `crop-regions` depends on `detect-regions`
- `face-refresh` depends on `crop-regions`
- `ai-index` depends on `crop-regions`

### Requirement: Dependency-based staleness check
`is_step_stale(step_name, depends_on, pipeline_state) -> bool` in `xmp_sidecar.py` SHALL return `True` if the step has no `completed` entry or any dependency's `completed` timestamp is newer than the step's own. No file mtime or external sources are consulted.

#### Scenario: Step is current
- **WHEN** all dependency `completed` timestamps are older than the step's own
- **THEN** `is_step_stale` returns `False`

#### Scenario: Dependency was re-run after this step
- **WHEN** a dependency's `completed` timestamp is newer than the step's own
- **THEN** `is_step_stale` returns `True`

#### Scenario: Step has never run
- **WHEN** the step has no `completed` entry
- **THEN** `is_step_stale` returns `True`

### Requirement: Cascading re-runs via dependency staleness
The orchestrator SHALL call `is_step_stale` per step per page. If stale, the step SHALL re-run and log `(re-run: <dep> updated)`.

#### Scenario: Cascade after detect-regions update
- **WHEN** `detect-regions` reruns within the same pipeline pass
- **THEN** `crop-regions`, `face-refresh`, and `ai-index` automatically re-run in turn because each sees a newer dependency timestamp

### Requirement: Justfile consolidation
The justfile SHALL be updated so all pipeline steps are driven through `photoalbums.py process`. Targets removed: `photoalbums-ctm-generate`, `photoalbums-ctm-apply`, `photoalbums-repair-crop-source`. Targets rewritten to `process --step <id>`: `photoalbums-render`, `photoalbums-detect-regions`, `photoalbums-crop-regions`, `photoalbums-render-pipeline`, `photoalbums-ai`, `photoalbums-ai-gps`. Retained unchanged: `photoalbums-map`, `photoalbums-watch`, `photoalbums-render-validate`.

#### Scenario: Full pipeline via justfile
- **WHEN** user runs `just photoalbums-process`
- **THEN** `photoalbums.py process --photos-root <configured-root>` is invoked
