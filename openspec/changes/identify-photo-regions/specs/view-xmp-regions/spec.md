## ADDED Requirements

### Requirement: Crop generation reads boundary boxes from the page view XMP sidecar
The system SHALL treat the `mwg-rs:RegionList` stored in the page view JPEG's XMP sidecar as the source of truth for crop boundaries. The crop step SHALL read those regions from XMP, convert each region into a pixel rectangle, and crop the corresponding area from the page `_V.jpg`.

#### Scenario: Crop step reads stored region list
- **WHEN** `crop-regions` runs for a page whose view sidecar contains a `mwg-rs:RegionList`
- **THEN** the crop step reads the stored regions from XMP, converts them to pixel rectangles, and writes one crop JPEG per region

#### Scenario: No region list in XMP
- **WHEN** the page view sidecar has no `mwg-rs:RegionList`
- **THEN** the crop step writes no crop JPEGs and leaves the page unresolved

### Requirement: Region captions use dc:description
The system SHALL store each region's caption in the region's standard `dc:description` field inside the `mwg-rs:RegionList`. If an internal implementation hint is retained, it MAY also store the same caption in `imago:CaptionHint`, but the crop step SHALL treat `dc:description` as the primary caption field.

#### Scenario: Region caption is present
- **WHEN** a detected region has a caption from Docling
- **THEN** the serialized region stores that caption in `dc:description`, and the crop step uses it when writing the crop sidecar

#### Scenario: Internal hint omitted
- **WHEN** the implementation does not write `imago:CaptionHint`
- **THEN** the crop step still works because it reads the standard `dc:description` field

### Requirement: Page view dc:description contains OCR and scene text
The system SHALL write the page view JPEG's top-level `dc:description` as a human-readable summary that includes the page OCR text and Gemma scene text. The raw strings SHALL still be written separately to `imago:OCRText` and `imago:SceneText`.

When both OCR text and scene text are present, `dc:description` SHALL include both in that order, with a blank line between them. Empty parts SHALL be omitted from the combined description.

#### Scenario: Page has OCR text and scene text
- **WHEN** the page has both OCR text and Gemma scene text
- **THEN** the page sidecar's `dc:description` contains both texts, while `imago:OCRText` and `imago:SceneText` remain populated separately

#### Scenario: Page has only OCR text
- **WHEN** the page has OCR text but no scene text
- **THEN** the page sidecar's `dc:description` contains only the OCR text and `imago:SceneText` remains empty

#### Scenario: Page has no OCR or scene text
- **WHEN** the page has neither OCR text nor scene text
- **THEN** the page sidecar's `dc:description` is empty or omitted, and the raw text fields remain empty

### Requirement: Derived render outputs publish their own visible text in dc:description
The system SHALL write every render-produced `_D##-##_V.jpg` sidecar with a top-level `dc:description` that summarizes the visible text in that image. The summary SHALL follow the same OCR-plus-scene-text composition rule as the page view output.

#### Scenario: Derived render output has OCR and scene text
- **WHEN** a derived `_D##-##_V.jpg` image has OCR text and Gemma scene text
- **THEN** its sidecar `dc:description` contains both texts, and the raw `imago:OCRText` and `imago:SceneText` fields remain separately populated

#### Scenario: Empty text on an output image
- **WHEN** a render-produced output image has no visible text summary
- **THEN** the output sidecar may leave `dc:description` empty, but it SHALL still preserve any separate raw text fields that were produced

### Requirement: Crop sidecar dc:description combines the region caption with crop-local OCR text
The system SHALL write each `_Photos/_D##-00_V.jpg` crop sidecar with a top-level `dc:description` that contains the region's caption and the OCR text recognized from the crop image itself. The crop description SHALL be caption-first, then crop OCR text, with a blank line between them when both are present. The crop sidecar SHALL also write the crop's OCR text separately to `imago:OCRText`.

#### Scenario: Crop has caption and crop OCR text
- **WHEN** a crop has a region caption and OCR text recognized from that crop image
- **THEN** the crop sidecar's `dc:description` contains both texts, and `imago:OCRText` stores the crop OCR text

#### Scenario: Crop has caption only
- **WHEN** a crop has a region caption but no OCR text
- **THEN** the crop sidecar's `dc:description` contains only the caption text

#### Scenario: Crop has OCR text only
- **WHEN** a crop has OCR text but no region caption
- **THEN** the crop sidecar's `dc:description` contains only the crop OCR text, and `imago:OCRText` stores the same text

