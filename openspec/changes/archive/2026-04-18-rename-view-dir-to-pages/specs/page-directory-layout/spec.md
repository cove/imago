## ADDED Requirements

### Requirement: Album directory naming uses a shared canonical layout helper
The system SHALL define the canonical album directory suffixes and sibling-derivation rules in one shared helper surface rather than reimplementing `_Archive`, `_Pages`, and `_Photos` string logic independently across modules.

#### Scenario: Runtime code derives a pages sibling
- **WHEN** runtime code needs the rendered page sibling for `Egypt_1975_B00_Archive`
- **THEN** it obtains `Egypt_1975_B00_Pages` through the shared layout helper
- **AND** it does not need to hard-code `_View` or `_Pages` string replacement locally

### Requirement: Rendered page directories use the `_Pages` suffix
The system SHALL treat `*_Pages` as the only canonical directory name for rendered album page JPEGs. Any runtime helper that derives, scans, or validates rendered page directories SHALL use `_Pages` rather than `_View`.

#### Scenario: Derive rendered page directory from an archive directory
- **WHEN** the system derives the rendered page directory sibling for `Egypt_1975_B00_Archive`
- **THEN** it resolves the rendered page directory as `Egypt_1975_B00_Pages`
- **AND** it does not derive `Egypt_1975_B00_View`

#### Scenario: Cast ingest uses page directories by default
- **WHEN** a Cast bulk photo ingest request uses the default album folder glob
- **THEN** the default glob is `*_Pages`
- **AND** the ingest scan targets JPEGs under `*_Pages`

### Requirement: Runtime page lookups resolve `_Pages` siblings
The system SHALL resolve relationships among `_Archive`, `_Pages`, and `_Photos` using `_Pages` as the rendered page directory suffix.

#### Scenario: Find archive directory from a page image path
- **WHEN** the system processes an image located under `Egypt_1975_B00_Pages`
- **THEN** archive-side sibling lookup resolves `Egypt_1975_B00_Archive`

#### Scenario: Find page-side sidecar candidates from a photo crop
- **WHEN** the system derives page-side sidecar candidates for a crop stored under `Egypt_1975_B00_Photos`
- **THEN** it searches under `Egypt_1975_B00_Pages`
- **AND** it does not generate page-side candidates under `Egypt_1975_B00_View`
