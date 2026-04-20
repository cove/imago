## ADDED Requirements

### Requirement: CTM is estimated from a stitched view image using Gemma 4 via LM Studio
The system SHALL call the LM Studio OpenAI-compatible endpoint with the rendered view JPEG and a structured prompt, and receive back a JSON object containing a 3×3 colour transformation matrix, a confidence score, and optional warnings.

#### Scenario: Successful CTM estimation
- **WHEN** `photoalbums ctm generate --album-id X --page N --photos-root <root>` is run and LM Studio is available
- **THEN** the system returns a CTM result with `matrix` (9 floats), `confidence` (0.0–1.0), `warnings` (list, may be empty), and `reasoning_summary` (string)

#### Scenario: Model returns malformed JSON
- **WHEN** the LM Studio response cannot be parsed as a valid CTM JSON object
- **THEN** the system retries up to 3 times with a stricter prompt; if all retries fail, it raises a `CTMValidationError` with the underlying parse failure message

#### Scenario: Returned matrix has fewer than 9 finite numeric coefficients
- **WHEN** the parsed JSON contains a `matrix` field that does not contain exactly 9 finite numbers
- **THEN** the system raises `CTMValidationError` and does not store the result

### Requirement: CTM is stored only in the `_Archive/` XMP sidecar using `crs:ColorMatrix1`
The system SHALL write the validated CTM to the `_Archive/` XMP sidecar for the stitched page as `crs:ColorMatrix1` and SHALL NOT write CTM metadata to `_View/` or derived sidecars.

#### Scenario: CTM stored in archive sidecar
- **WHEN** a valid CTM is generated for page P12 of album Egypt_1975
- **THEN** the value is written to `Egypt_1975_Archive/<stem>_S01.xmp` under `crs:ColorMatrix1` and the `_View/` sidecar is unchanged

#### Scenario: Existing CTM not overwritten without --force
- **WHEN** `ctm generate` is run for a page whose archive XMP already contains `crs:ColorMatrix1` and `--force` is not passed
- **THEN** the system skips generation and reports the existing CTM values

#### Scenario: Existing CTM overwritten with --force
- **WHEN** `ctm generate` is run with `--force` on a page with an existing CTM
- **THEN** the `pipeline.ctm` record is cleared, the old matrix is replaced with the newly generated result, and a new `pipeline.ctm` record is written

### Requirement: CTM generation records completion in imago:Detections pipeline state
The system SHALL write a `pipeline.ctm` record to the archive sidecar's `imago:Detections` JSON when CTM generation succeeds, and SHALL skip generation (printing the recorded timestamp) when that record is already present and `--force` is not set.

#### Scenario: Successful generation writes pipeline state
- **WHEN** CTM generation succeeds
- **THEN** the archive sidecar's `imago:Detections` contains `{"pipeline": {"ctm": {"completed": "<iso-timestamp>", "model": "<model-id>"}}, ...}`

#### Scenario: Pipeline state causes generation to be skipped
- **WHEN** `ctm generate` is run and `pipeline.ctm.completed` is present in the archive sidecar's `imago:Detections` and `--force` is not set
- **THEN** the system skips the LM Studio call and prints a skip message with the recorded completion timestamp

### Requirement: CTM generation supports both page-level and per-photo-level modes
The system SHALL allow `ctm generate` to run against either the page-level `_V.jpg` (storing the result in the `_Archive/` XMP) or individual crop JPEGs in `_Photos/` (storing the result in each crop's own XMP sidecar). A `--per-photo` flag selects per-photo mode.

#### Scenario: Page-level CTM generation
- **WHEN** `photoalbums ctm generate --album-id X --photos-root <root>` is run without `--per-photo`
- **THEN** the Gemma 4 model analyses the `_V.jpg` and the result is stored in the `_Archive/` XMP as `crs:ColorMatrix1`

#### Scenario: Per-photo CTM generation
- **WHEN** `photoalbums ctm generate --album-id X --photos-root <root> --per-photo` is run
- **THEN** the Gemma 4 model analyses each crop JPEG in `_Photos/` individually, and each result is stored in that crop's own XMP sidecar as `crs:ColorMatrix1`

#### Scenario: Per-photo and page-level CTMs are independent
- **WHEN** a page has both a page CTM in `_Archive/` XMP and per-photo CTMs in crop sidecars
- **THEN** `ctm-apply` applies the page CTM to `_V.jpg` and each per-photo CTM to its respective crop; neither overrides the other

### Requirement: ctm-apply corrects both the page view and individual photo crops in a single pass
The system SHALL provide a `ctm-apply` pipeline step (and standalone CLI command) that reads `crs:ColorMatrix1` from the archive XMP and applies it to `_V.jpg`, AND reads `crs:ColorMatrix1` from each crop's own XMP sidecar and applies it to that crop JPEG. Both levels run in a single invocation. The render and crop-regions steps SHALL NOT apply any CTM.

#### Scenario: Page CTM applied to view JPEG
- **WHEN** `ctm-apply` runs and `crs:ColorMatrix1` is stored in the archive XMP
- **THEN** `_V.jpg` pixels are corrected in-place and `pipeline.ctm_applied` is recorded on its sidecar

#### Scenario: Per-photo CTM applied to individual crop
- **WHEN** `ctm-apply` runs and a crop's XMP sidecar contains `crs:ColorMatrix1`
- **THEN** that crop JPEG is corrected in-place and `pipeline.ctm_applied` is recorded on that crop's sidecar

#### Scenario: ctm-apply skips silently per-file when no CTM is stored
- **WHEN** a JPEG (page or crop) has no `crs:ColorMatrix1` in its source XMP
- **THEN** that file is not modified and the pipeline continues without error

#### Scenario: Render and crop-regions steps produce uncorrected pixels
- **WHEN** the render step writes `_V.jpg` or crop-regions writes a crop JPEG
- **THEN** no colour matrix is applied; pixels reflect the raw stitched or cropped data

### Requirement: ctm-apply records completion in imago:Detections pipeline state
The system SHALL write a `pipeline.ctm_applied` record to the rendered JPEG's sidecar `imago:Detections` when the transform is applied, and SHALL skip application when that record is already present and `--force` is not set.

#### Scenario: Successful ctm-apply writes pipeline state
- **WHEN** CTM is applied to a rendered JPEG
- **THEN** the sidecar's `imago:Detections` contains `{"pipeline": {"ctm_applied": {"completed": "<iso-timestamp>", "model": "<ctm-model-id>"}}, ...}`

#### Scenario: Pipeline state skips ctm-apply on re-run
- **WHEN** `ctm-apply` is run and `pipeline.ctm_applied` is already present in the sidecar's `imago:Detections` and `--force` is not set
- **THEN** the JPEG is not re-processed and a skip message is printed

#### Scenario: --force clears state and re-applies
- **WHEN** `ctm-apply --force` is run on an already-corrected JPEG
- **THEN** `pipeline.ctm_applied` is cleared, the matrix is re-applied, and a new `pipeline.ctm_applied` record is written

### Requirement: CTM review command returns stored matrix values
The system SHALL provide a `ctm review` subcommand that reads and prints the stored `crs:ColorMatrix1` from the archive XMP alongside confidence and warnings.

#### Scenario: Review displays stored CTM
- **WHEN** `photoalbums ctm review --album-id X --page N` is run and a CTM is stored
- **THEN** the output is a JSON line with `image`, `archive_xmp`, `ctm.matrix`, `ctm.confidence`, and `ctm.warnings`

#### Scenario: Review reports missing CTM
- **WHEN** no CTM is stored for the requested page
- **THEN** the output JSON contains `"ctm": null` and exit code is 0
