## Requirements

### Requirement: View-image XMP metadata can store photo regions
Imago MUST support storing detected photo regions for stitched view images in XMP sidecars and MUST preserve clean separation from CTM restoration metadata stored in the `_Archive/` XMP.

#### Scenario: Preserve region metadata while CTM lives in `_Archive/` XMP
- **WHEN** a stitched view image has photo-region metadata and a corresponding stitched image has CTM restoration metadata
- **THEN** Imago preserves the existing region metadata structure for the view-image XMP
- **AND** stores CTM metadata only in the `_Archive/` XMP
- **AND** does not require archive-manifest ingredient or homography fields for this CTM workflow

### Requirement: Crop generation reads boundary boxes from the page view XMP sidecar
The system SHALL treat the `mwg-rs:RegionList` stored in the page view JPEG's XMP sidecar as the source of truth for crop boundaries. The crop step SHALL read those regions from XMP, convert each region into a pixel rectangle, and crop the corresponding area from the page `_V.jpg`.

#### Scenario: Crop step reads stored region list
- **WHEN** `crop-regions` runs for a page whose view sidecar contains a `mwg-rs:RegionList`
- **THEN** the crop step reads the stored regions from XMP, converts them to pixel rectangles, and writes one crop JPEG per region

#### Scenario: No region list in XMP
- **WHEN** the page view sidecar has no `mwg-rs:RegionList`
- **THEN** the crop step writes no crop JPEGs and leaves the page unresolved

### Requirement: Region captions use mwg-rs:Name
The system SHALL store each region's caption in the region's `mwg-rs:Name` field inside the `mwg-rs:RegionList`. If an internal implementation hint is retained, it MAY also store the same caption in `imago:CaptionHint`, but the crop step SHALL treat `mwg-rs:Name` as the primary caption field.

#### Scenario: Region caption is present
- **WHEN** a detected region has a caption from Docling
- **THEN** the serialized region stores that caption in `mwg-rs:Name`, and the crop step uses it when writing the crop sidecar

#### Scenario: Internal hint omitted
- **WHEN** the implementation does not write `imago:CaptionHint`
- **THEN** the crop step still works because it reads the standard `mwg-rs:Name` field

### Requirement: Page view dc:description contains OCR and scene text
The system SHALL write the page view JPEG's top-level `dc:description` as a human-readable summary that includes the page OCR text and Gemma scene text. The raw strings SHALL still be written separately to `imago:OCRText` and `imago:SceneText`.

When both OCR text and scene text are present, `dc:description` SHALL include both in that order, with a blank line between them. Empty parts SHALL be omitted from the combined description. This page-level `dc:description` is for searchable page text and SHALL NOT be reused as the per-region caption store.

#### Scenario: Page has OCR text and scene text
- **WHEN** the page has both OCR text and Gemma scene text
- **THEN** the page sidecar's `dc:description` contains both texts, while `imago:OCRText` and `imago:SceneText` remain populated separately

#### Scenario: Page has only OCR text
- **WHEN** the page has OCR text but no scene text
- **THEN** the page sidecar's `dc:description` contains only the OCR text and `imago:SceneText` remains empty

#### Scenario: Page has no OCR or scene text
- **WHEN** the page has neither OCR text nor scene text
- **THEN** the page sidecar's `dc:description` is empty or omitted, and the raw text fields remain empty

### Requirement: Crop sidecars use the stored region caption
The system SHALL treat the region caption stored in `mwg-rs:Name` as the caption associated with that photo when writing crop-side metadata.

#### Scenario: Crop uses stored region caption
- **WHEN** a stored region has a non-empty `mwg-rs:Name`
- **THEN** the crop step uses that text as the region caption for the crop sidecar
