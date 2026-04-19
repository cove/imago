## ADDED Requirements

### Requirement: Legacy caption-layout sidecars can be verified before migration
The system SHALL provide a verification flow that scans existing album `.xmp` sidecars and reports files that still use the legacy crop-caption layout or legacy custom `dc:description` alt-text entries.

#### Scenario: Verification reports legacy sidecars
- **WHEN** verification scans a photo album root containing sidecars with `xml:lang="x-caption"` or crop-side inherited page OCR stored as `imago:OCRText`
- **THEN** it reports those sidecars as needing migration

#### Scenario: Verification passes after migration
- **WHEN** verification scans a photo album root whose sidecars already match the new caption contract
- **THEN** it reports no caption-layout migration work remaining

### Requirement: Caption-layout migration rewrites sidecars in place
The system SHALL provide a migration that rewrites existing sidecars from the legacy caption and OCR layout to the new contract without regenerating image files and without changing unrelated XMP fields.

#### Scenario: Migrate a legacy crop sidecar in place
- **WHEN** migration processes a crop sidecar whose `dc:description` uses a legacy `x-caption` entry and whose inherited page OCR is stored in `imago:OCRText`
- **THEN** the sidecar is rewritten so the crop caption becomes `dc:description` `x-default`
- **AND** the inherited page OCR is stored in `imago:ParentOCRText`
- **AND** unrelated fields in the sidecar remain unchanged

#### Scenario: Leave already-migrated sidecars untouched
- **WHEN** migration processes a sidecar that already matches the new caption and parent-OCR contract
- **THEN** the sidecar content is left unchanged

### Requirement: Migration resolves crop captions from authoritative sources
When rewriting a legacy crop sidecar, the migration SHALL recover the crop caption from the strongest available source before falling back to page-level text.

The caption resolution priority SHALL be:
1. The parent page region's `mwg-rs:Name`
2. The crop sidecar's legacy `dc:description` `x-caption` value
3. The crop sidecar's existing logical description if it already represents a crop caption
4. The parent page sidecar's `dc:description`

#### Scenario: Parent region caption wins during migration
- **WHEN** the parent page region has a non-empty `mwg-rs:Name` and the crop sidecar also contains a legacy `x-caption`
- **THEN** migration uses the parent region caption as the migrated crop caption

#### Scenario: Page description is used only as final fallback
- **WHEN** migration cannot recover any per-crop caption from the parent region or crop sidecar
- **AND** the parent page sidecar has a non-empty `dc:description`
- **THEN** migration uses the parent page `dc:description` as the migrated crop description fallback
