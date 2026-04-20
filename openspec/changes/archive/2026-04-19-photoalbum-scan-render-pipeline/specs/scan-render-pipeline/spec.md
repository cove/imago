## ADDED Requirements

### Requirement: render-pipeline CLI command runs all render steps in order
The system SHALL provide a `render-pipeline` subcommand in `photoalbums.py` that executes render, region detection, crop generation, face refresh, and CTM application in sequence for a matching page, album, or album set.

#### Scenario: Full pipeline run for a single album
- **WHEN** `photoalbums render-pipeline --album-id Egypt_1975 --photos-root <root>` is run
- **THEN** the system processes every page in the album in order: render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply, printing per-page progress

#### Scenario: Single-page pipeline run
- **WHEN** `photoalbums render-pipeline --album-id Egypt_1975 --page 26 --photos-root <root>` is run
- **THEN** only the page matching `P26` is processed through the pipeline steps for that page

#### Scenario: Provenance is written when outputs are created
- **WHEN** the render step writes a new page `_V.jpg`, derived `_D##-##_V.jpg`, or crop `_D##-00_V.jpg`
- **THEN** that output's sidecar receives `xmpMM:DocumentID` immediately
- **AND** its initial `xmpMM:DerivedFrom` and `xmpMM:Pantry` are written as soon as the source set for that file is known

#### Scenario: AI steps are skipped when pipeline state records completion
- **WHEN** the pipeline runs and the `imago:Detections` `pipeline` record for a step is already present and `--force` is not set
- **THEN** that step is skipped and a skip message is printed
- **AND** downstream steps that depend only on the existing data, not on rerunning the skipped step, continue normally

#### Scenario: --force flag re-runs all pipeline steps
- **WHEN** `--force` is passed to `render-pipeline`
- **THEN** the relevant pipeline-state records are cleared and each applicable step re-runs for the targeted page(s)

#### Scenario: Pipeline step failure is reported immediately and summarized after completion
- **WHEN** one page fails during `face-refresh`
- **THEN** the failing page id, step name, and underlying error are printed immediately
- **AND** the pipeline continues with the next page
- **AND** the failed page appears in the end-of-run failure summary
- **AND** the command exits non-zero if any page failed

### Requirement: Pipeline XMP updates preserve unrelated sidecar fields
The system SHALL update the canonical XMP sidecar in place for each step and SHALL preserve unrelated fields, including manual edits, when rerunning a step that owns only a subset of the sidecar.

#### Scenario: Region detection rerun preserves manual fields
- **WHEN** `detect-view-regions --force` reruns for a page whose sidecar already contains manual `dc:description` and location fields
- **THEN** the step updates only the region-detection-owned fields and pipeline state
- **AND** the unrelated sidecar fields remain unchanged

### Requirement: Pipeline step completion is tracked in imago:Detections under a pipeline key
The system SHALL record each completed AI-backed pipeline step in the `imago:Detections` JSON blob on the relevant XMP sidecar under a `"pipeline"` key. Each entry SHALL store at minimum the completion timestamp and model identifier where applicable, and MAY include step-specific metadata.

#### Scenario: Completed region-detection step recorded in imago:Detections
- **WHEN** region detection completes successfully for a view JPEG and regions are found
- **THEN** the view sidecar's `imago:Detections` JSON contains `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>", "result": "regions_found"}}, ...}`

#### Scenario: Completed no-regions step recorded explicitly
- **WHEN** region detection completes successfully for a title page and finds no regions
- **THEN** the view sidecar's `imago:Detections` JSON contains `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>", "result": "no_regions"}}, ...}`

#### Scenario: Archive sidecar records page CTM generation completion
- **WHEN** page-level CTM generation completes
- **THEN** the archive scan sidecar's `imago:Detections` contains `{"pipeline": {"ctm": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

#### Scenario: Crop sidecar records per-photo CTM generation completion
- **WHEN** per-photo CTM generation completes for a crop
- **THEN** the crop's XMP sidecar `imago:Detections` contains `{"pipeline": {"ctm": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

### Requirement: Each pipeline step is also runnable through the same underlying command path
The system SHALL allow each pipeline step to be invoked independently via CLI so that a single slow or review-driven step can be run in isolation without re-executing the full pipeline.

#### Scenario: Standalone CTM apply
- **WHEN** `photoalbums ctm-apply --album-id X --photos-root <root>` is run
- **THEN** only CTM application runs against already-rendered JPEGs
- **AND** pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone region detection
- **WHEN** `photoalbums detect-view-regions --album-id X --photos-root <root>` is run
- **THEN** only region detection runs
- **AND** pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone crop-regions
- **WHEN** `photoalbums crop-regions --album-id X --photos-root <root>` is run
- **THEN** only crop generation runs
- **AND** pipeline state is checked and updated exactly as when run inside `render-pipeline`

#### Scenario: Standalone face refresh
- **WHEN** `photoalbums face-refresh --album-id X --photos-root <root>` is run
- **THEN** only face refresh runs
- **AND** pipeline state is checked and updated exactly as when run inside `render-pipeline`
