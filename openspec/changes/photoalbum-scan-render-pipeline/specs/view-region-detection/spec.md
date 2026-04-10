## MODIFIED Requirements

### Requirement: Region detection is invoked as a pipeline step after render
The system SHALL support region detection being called as a step within the `render-pipeline` command, running after the render step has completed for each page. All existing detection behavior (model call, XMP write-back, cache semantics, retry logic) is unchanged.

#### Scenario: Detection runs on the raw rendered view JPEG
- **WHEN** `render-pipeline` reaches the detect-regions step for a page
- **THEN** detection is run against the raw `_V.jpg` from the render step; CTM has not yet been applied at this stage, which is intentional — region boundaries are geometry-based and stable across colour corrections

#### Scenario: Detection skipped when pipeline state records completion
- **WHEN** the view sidecar's `imago:Detections` contains `pipeline.view_regions.completed` and `--force` is not set
- **THEN** the detection step is skipped and existing regions are preserved

### Requirement: Region detection records completion in imago:Detections pipeline state
The system SHALL write a `pipeline.view_regions` record to the view sidecar's `imago:Detections` JSON when detection succeeds, and SHALL skip detection when that record is already present and `--force` is not set.

#### Scenario: Successful detection writes pipeline state
- **WHEN** region detection completes and writes `mwg-rs:RegionList` to the view sidecar
- **THEN** the view sidecar's `imago:Detections` is updated to include `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

#### Scenario: Pipeline state skips detection on re-run
- **WHEN** `detect-view-regions` is run and `pipeline.view_regions.completed` is already present in the view sidecar's `imago:Detections` and `--force` is not set
- **THEN** the system skips the LM Studio vision call and prints a skip message
