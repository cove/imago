## ADDED Requirements

### Requirement: render-pipeline CLI command runs all render steps in order
The system SHALL provide a `render-pipeline` subcommand in `photoalbums.py` that executes render, region detection, face refresh, and provenance steps in sequence for a matching page, album, or album set.

#### Scenario: Full pipeline run for a single album
- **WHEN** `photoalbums render-pipeline --album-id Egypt_1975 --photos-root <root>` is run
- **THEN** the system processes every page in the album in order: render → detect-regions → crop-regions → ctm-apply → face-refresh → provenance, printing per-page progress

#### Scenario: Single-page pipeline run
- **WHEN** `photoalbums render-pipeline --album-id Egypt_1975 --page 26 --photos-root <root>` is run
- **THEN** only the page matching `P26` is processed through all pipeline steps

#### Scenario: Archive scan and rendered outputs receive DocumentID during the render step
- **WHEN** the render step runs for a page
- **THEN** the archive scan sidecar receives a `xmpMM:DocumentID` before stitching begins, and each rendered JPEG sidecar receives a `xmpMM:DocumentID` immediately after that JPEG is written

#### Scenario: Stored CTM is applied automatically during render step
- **WHEN** the archive XMP for a page contains `crs:ColorMatrix1` and the pipeline's render step runs
- **THEN** the CTM is applied to the stitched image before the view JPEG is written, without any extra flag required

#### Scenario: AI steps are skipped when pipeline state records completion
- **WHEN** the pipeline runs and the `imago:Detections` `pipeline` record for a step is already present and `--force` is not set
- **THEN** that step is skipped and a skip message is printed; downstream steps that depend only on data (not re-execution) continue normally

#### Scenario: --force flag re-runs all steps and clears pipeline state
- **WHEN** `--force` is passed to render-pipeline
- **THEN** render outputs are regenerated, all pipeline state records are cleared, and every step re-runs and re-records its completion

#### Scenario: Pipeline step failure is reported but does not abort other pages
- **WHEN** one page fails during face refresh (e.g. Cast store unavailable)
- **THEN** the error is printed including the underlying error message, the pipeline state for that step is NOT recorded, the page is skipped, and the pipeline continues with the next page; exit code is non-zero if any page had an error

### Requirement: Pipeline step completion is tracked in imago:Detections under a pipeline key
The system SHALL record each completed AI pipeline step in the `imago:Detections` JSON blob on the relevant XMP sidecar under a `"pipeline"` key. Each entry SHALL store at minimum the step name, completion timestamp, and model identifier used. This record is the authoritative source for whether a step needs to re-run.

#### Scenario: Completed step recorded in imago:Detections
- **WHEN** region detection completes successfully for a view JPEG
- **THEN** the view sidecar's `imago:Detections` JSON contains `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

#### Scenario: Step with no model records completion without model field
- **WHEN** the provenance step completes
- **THEN** the rendered sidecar's `imago:Detections` JSON contains `{"pipeline": {"provenance": {"completed": "<iso-timestamp>"}}, ...}`

#### Scenario: Archive sidecar records page CTM generation completion
- **WHEN** page-level CTM generation completes
- **THEN** the archive scan sidecar's `imago:Detections` contains `{"pipeline": {"ctm": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

#### Scenario: Crop sidecar records per-photo CTM generation completion
- **WHEN** per-photo CTM generation completes for a crop
- **THEN** the crop's XMP sidecar `imago:Detections` contains `{"pipeline": {"ctm": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

### Requirement: Each AI pipeline step is also runnable as a standalone CLI command
The system SHALL allow each AI-backed pipeline step to be invoked independently via CLI so that a single slow step can be run in isolation without re-executing the full pipeline.

#### Scenario: Standalone CTM apply
- **WHEN** `photoalbums ctm-apply --album-id X --photos-root <root>` is run
- **THEN** only CTM application runs against already-rendered JPEGs; pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone region detection
- **WHEN** `photoalbums detect-view-regions --album-id X --photos-root <root>` is run
- **THEN** only region detection runs; pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone face refresh
- **WHEN** `photoalbums face-refresh --album-id X --photos-root <root>` is run
- **THEN** only face refresh runs; pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone provenance write
- **WHEN** `photoalbums write-provenance --album-id X --photos-root <root>` is run
- **THEN** only DerivedFrom and Pantry are written; pipeline state is checked and updated exactly as when run inside `render-pipeline`
