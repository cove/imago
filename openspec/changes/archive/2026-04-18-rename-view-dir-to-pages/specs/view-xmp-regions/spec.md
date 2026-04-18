## ADDED Requirements

### Requirement: Crop metadata references rendered pages under `_Pages`
The system SHALL treat `*_Pages` as the rendered page location when crop-side metadata references a page JPEG or page-side sidecar.

#### Scenario: Crop-side metadata resolves the page sidecar location
- **WHEN** the system derives the page-side XMP candidate for a crop stored under `Egypt_1975_B00_Photos`
- **THEN** the candidate page sidecar is resolved under `Egypt_1975_B00_Pages`
- **AND** no candidate under `Egypt_1975_B00_View` is treated as canonical

#### Scenario: Page-relative metadata is written after the directory rename
- **WHEN** the system writes or refreshes crop metadata that references `Egypt_1975_B00_P26_V.jpg`
- **THEN** the stored page location uses the `Egypt_1975_B00_Pages` directory
