## ADDED Requirements

### Requirement: POC script runs docling-layout-heron and standard pipeline on test image and compares output
The system SHALL provide a standalone POC script that processes `Family_1980-1985_B08_P16_V.jpg` using both the current standard Docling pipeline and the docling-layout-heron model (when locally available), prints detected bounding boxes from each, runs Gemma4 caption matching against the standard-pipeline boxes, and prints the merged result.

#### Scenario: Heron model weights are available locally
- **WHEN** the POC script runs and the heron model weights are present on disk
- **THEN** the script runs both pipelines, prints bounding boxes from each side-by-side, and reports region counts

#### Scenario: Heron model weights are not available
- **WHEN** the POC script runs and the heron model is not found locally
- **THEN** the script prints a clear message indicating heron is unavailable, runs only the standard pipeline, and continues to the Gemma4 caption step

#### Scenario: POC script completes end-to-end
- **WHEN** the script finishes successfully
- **THEN** it prints the final merged list of (bounding box, caption) pairs for the test image in a human-readable format

### Requirement: POC script optionally writes a debug image with numbered bounding boxes overlaid
The system SHALL support a `--debug-image` flag on the POC script that, when set, writes a copy of the test image with each detected bounding box drawn and labelled with its sort-order number, to aid visual validation of the sort-order convention.

#### Scenario: Debug image flag enabled
- **WHEN** the POC script is run with `--debug-image`
- **THEN** a PNG is written alongside the script output showing labelled bounding boxes over the test image

#### Scenario: Debug image flag not set
- **WHEN** the POC script is run without `--debug-image`
- **THEN** no debug image is written
