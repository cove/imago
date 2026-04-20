## ADDED Requirements

### Requirement: Multi-scan pages are stitched into a single view JPEG
The system SHALL stitch all archive TIF scans for a page into a single composite image and write it as `_V.jpg` in the `_View/` directory. The stitcher SHALL try multiple affine configurations in sequence and fall back to a linear-overlap algorithm for two-scan pages.

#### Scenario: Successful multi-scan stitch
- **WHEN** a page has two or more archive TIF scans (e.g. `_S01.tif`, `_S02.tif`)
- **THEN** the system produces a single `_V.jpg` that includes all scans stitched together and `_result_expands_canvas` confirms no scan was dropped

#### Scenario: Linear-pair fallback for two-scan pages
- **WHEN** all affine stitch configurations fail for a two-scan page
- **THEN** the system falls back to `_stitch_linear_pair_images` using overlap detection and produces a valid `_V.jpg`

#### Scenario: Partial panorama raises an error
- **WHEN** the stitcher warns that not all scans were included in the panorama
- **THEN** the system raises `RuntimeError("Stitching produced a partial panorama (not all scans were included)")` and does not write output

#### Scenario: All stitching attempts fail
- **WHEN** every affine attempt and the linear-pair fallback raise exceptions
- **THEN** the system raises `RuntimeError("All stitching attempts failed")` with the last underlying exception chained

### Requirement: Single-scan pages are converted to JPEG without stitching
The system SHALL read a single archive TIF and write it directly as `_V.jpg`, skipping the stitching step.

#### Scenario: Single scan renders to view JPEG
- **WHEN** a page has exactly one archive TIF scan (`_S01.tif`)
- **THEN** the system reads the TIF and writes `_V.jpg` without invoking the stitcher

### Requirement: Render step produces uncorrected JPEG; CTM is applied separately
The system SHALL NOT apply any colour matrix during stitching or JPEG writing. The render step writes the JPEG from raw stitched pixels; colour correction is applied by the subsequent `ctm-apply` step.

#### Scenario: Render writes JPEG without CTM
- **WHEN** the render step runs for a page that has `crs:ColorMatrix1` stored in the archive XMP
- **THEN** the written JPEG reflects the raw stitched pixels; the CTM is not applied during render

### Requirement: Derived media files are rendered to JPEG
The system SHALL convert `_D##-##` derived TIF or media files to `_D##-##_V.jpg` in the `_View/` directory. Output JPEG quality SHALL be reduced iteratively if the output file size would exceed the source file size.

#### Scenario: Derived TIF rendered to derived view JPEG
- **WHEN** a `_D##-##.tif` file is present in the archive directory
- **THEN** the system produces `_D##-##_V.jpg` in the `_View/` directory

#### Scenario: Quality reduced to keep derived JPEG under source size
- **WHEN** the initial JPEG write at quality 80 produces a file larger than the source
- **THEN** the system retries at decreasing quality steps (80 → 70 → … → 40) until the output is smaller than the source

### Requirement: Existing render outputs are skipped unless forced
The system SHALL skip writing a render output if the target JPEG already exists and is valid, printing a skip message. The `--force` flag (or equivalent pipeline flag) SHALL cause the output to be regenerated.

#### Scenario: Existing output skipped
- **WHEN** `_V.jpg` already exists for a page and `--force` is not set
- **THEN** the render step prints a skip message and returns without overwriting the file

### Requirement: Every render output is validated after writing
The system SHALL open the written JPEG with Pillow after writing and raise `RuntimeError` if the file cannot be read or is corrupt.

#### Scenario: Output validation failure raises error
- **WHEN** a written JPEG cannot be opened by Pillow
- **THEN** the system raises `RuntimeError("Output validation failed: <path>")`

### Requirement: S01 scan is required for every page
The system SHALL identify the `_S01.tif` as the primary scan for each page. If no `_S01.tif` is found, the system SHALL raise `RuntimeError` before attempting any render.

#### Scenario: Missing S01 raises error
- **WHEN** a page directory contains scans but none is numbered `_S01`
- **THEN** the system raises `RuntimeError` naming the scans that were found
