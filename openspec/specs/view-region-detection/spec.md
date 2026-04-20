## MODIFIED Requirements

### Requirement: Region detection is invoked as a pipeline step after render for page views only
The system SHALL support region detection being called as a step within the `render-pipeline` command, running after the render step has completed for each page. Detection runs only on page `_V.jpg` images. It does not run on render-produced `_D##-##_V.jpg` derived outputs.

#### Scenario: Detection runs on the raw rendered page view JPEG
- **WHEN** `render-pipeline` reaches the detect-regions step for a page
- **THEN** detection is run against the raw `_V.jpg` from the render step
- **AND** face refresh and CTM application have not yet been applied to that page view

#### Scenario: Derived outputs are not region-detected
- **WHEN** `render-pipeline` creates `_D01-02_V.jpg` as a derived render output
- **THEN** the detect-regions step does not run on that derived JPEG

#### Scenario: Detection skipped when pipeline state records completion
- **WHEN** the view sidecar's `imago:Detections` contains `pipeline.view_regions.completed` and `--force` is not set
- **THEN** the detection step is skipped and existing regions are preserved

### Requirement: Region detection records explicit success state in imago:Detections pipeline state
The system SHALL write a `pipeline.view_regions` record to the view sidecar's `imago:Detections` JSON when detection succeeds, including an explicit result value that distinguishes "regions found" from "no regions found".

#### Scenario: Successful detection with regions writes pipeline state
- **WHEN** region detection completes and writes `mwg-rs:RegionList` to the view sidecar
- **THEN** the view sidecar's `imago:Detections` is updated to include `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>", "result": "regions_found"}}, ...}`

#### Scenario: Successful detection with no regions writes explicit no-regions state
- **WHEN** region detection runs successfully for a title page and finds no regions
- **THEN** the view sidecar's `imago:Detections` is updated to include `{"pipeline": {"view_regions": {"completed": "<iso-timestamp>", "model": "<model-id>", "result": "no_regions"}}, ...}`

#### Scenario: Pipeline state skips detection on re-run
- **WHEN** `detect-view-regions` is run and `pipeline.view_regions.completed` is already present in the view sidecar's `imago:Detections` and `--force` is not set
- **THEN** the system skips the LM Studio vision call and prints a skip message

### Requirement: Non-empty region results are validated before XMP write
The system SHALL validate non-empty AI region results before writing `mwg-rs:RegionList` to XMP.

#### Scenario: Zero-area region is rejected
- **WHEN** the model returns a region whose width or height is zero or negative
- **THEN** that region is not written to XMP

#### Scenario: Heavy overlap is resolved deterministically
- **WHEN** the model returns two heavily overlapping regions for what appears to be the same photo
- **THEN** the system resolves the conflict deterministically before writing the final region list
