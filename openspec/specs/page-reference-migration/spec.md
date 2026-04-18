# page-reference-migration Specification

## Purpose
Define how XMP metadata writes and migrates rendered page references from the legacy `_View` directory name to `_Pages`.
## Requirements
### Requirement: New XMP page references use `_Pages`
The system SHALL write page-JPEG XMP references using `*_Pages` whenever a sidecar stores a relative path to a rendered page JPEG.

#### Scenario: Crop sidecar references its source page
- **WHEN** the system writes a crop sidecar whose source page JPEG is `Egypt_1975_B00_P26_V.jpg`
- **THEN** the stored relative page reference points at `../Egypt_1975_B00_Pages/Egypt_1975_B00_P26_V.jpg`
- **AND** it does not write `../Egypt_1975_B00_View/Egypt_1975_B00_P26_V.jpg`

### Requirement: Existing XMP page references are migrated from `_View` to `_Pages`
The system SHALL provide a migration that rewrites existing XMP page references from `*_View` to `*_Pages` without changing unrelated XMP fields.

#### Scenario: Migrate a sidecar with a `_View` page reference
- **WHEN** the migration processes an XMP sidecar containing `../Europe_1985_B02_View/Europe_1985_B02_P18_V.jpg`
- **THEN** the resulting sidecar contains `../Europe_1985_B02_Pages/Europe_1985_B02_P18_V.jpg`
- **AND** unrelated fields in the sidecar remain unchanged

#### Scenario: Leave sidecars without `_View` page references untouched
- **WHEN** the migration processes an XMP sidecar that contains no `_View` page reference
- **THEN** the sidecar content is left unchanged
