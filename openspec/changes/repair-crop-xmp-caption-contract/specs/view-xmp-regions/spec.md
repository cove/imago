## ADDED Requirements

### Requirement: XMP descriptions do not use custom non-language alt entries
The system SHALL NOT write custom `dc:description` alt-text entries such as `x-caption`, `x-author`, or `x-scene` on page or crop sidecars. OCR- and scene-derived text SHALL be stored in dedicated XMP fields instead of custom description languages.

#### Scenario: Page sidecar description omits legacy custom alt entries
- **WHEN** the system writes or rewrites a page sidecar with OCR text, scene text, or both
- **THEN** the resulting `dc:description` contains only standard Lang Alt entries needed for the visible description
- **AND** it does not contain `xml:lang="x-caption"`, `xml:lang="x-author"`, or `xml:lang="x-scene"`

#### Scenario: Crop sidecar description omits legacy custom alt entries
- **WHEN** the system writes or rewrites a crop sidecar with a resolved region caption
- **THEN** the resulting `dc:description` contains only standard Lang Alt entries needed for the crop caption
- **AND** it does not contain `xml:lang="x-caption"`, `xml:lang="x-author"`, or `xml:lang="x-scene"`

### Requirement: Crop sidecars store inherited page OCR as parent context
When crop-side metadata inherits OCR text from the parent page, the system SHALL store that text in `imago:ParentOCRText` rather than `imago:OCRText`.

#### Scenario: Crop inherits page OCR context
- **WHEN** a crop sidecar is written for a page whose sidecar has non-empty OCR text
- **THEN** the crop sidecar contains that inherited text in `imago:ParentOCRText`
- **AND** it does not write the inherited page OCR to `imago:OCRText`

## MODIFIED Requirements

### Requirement: Page view dc:description contains OCR and scene text
The system SHALL write the page view JPEG's top-level `dc:description` as a human-readable summary that includes the page OCR text and Gemma scene text. The raw strings SHALL still be written separately to `imago:OCRText` and `imago:SceneText`.

When both OCR text and scene text are present, `dc:description` SHALL include both in that order using labeled sections separated by a blank line:

```text
OCR:
<ocr text>

Scene Text:
<scene text>
```

Empty parts SHALL be omitted from the combined description. This page-level `dc:description` is for searchable page text and SHALL NOT be reused as the per-region caption store.

#### Scenario: Page has OCR text and scene text
- **WHEN** the page has both OCR text and Gemma scene text
- **THEN** the page sidecar's `dc:description` contains an `OCR:` section followed by a `Scene Text:` section
- **AND** `imago:OCRText` and `imago:SceneText` remain populated separately

#### Scenario: Page has only OCR text
- **WHEN** the page has OCR text but no scene text
- **THEN** the page sidecar's `dc:description` contains only the OCR-derived section
- **AND** `imago:SceneText` remains empty

#### Scenario: Page has no OCR or scene text
- **WHEN** the page has neither OCR text nor scene text
- **THEN** the page sidecar's `dc:description` is empty or omitted
- **AND** the raw text fields remain empty

### Requirement: Crop sidecars use the stored region caption
The system SHALL treat the region caption stored in `mwg-rs:Name` as the caption associated with that photo when writing crop-side metadata.

When a non-empty region caption is available, the crop sidecar's `dc:description` SHALL use that caption as its `x-default` value. Inherited page OCR text SHALL NOT replace or outrank a non-empty crop caption.

#### Scenario: Crop uses stored region caption
- **WHEN** a stored region has a non-empty `mwg-rs:Name`
- **THEN** the crop step uses that text as the crop sidecar caption
- **AND** the crop sidecar writes that caption as `dc:description` `x-default`

#### Scenario: Page OCR does not override crop caption
- **WHEN** a crop has a non-empty region caption and the parent page sidecar also has non-empty OCR text
- **THEN** the crop sidecar's default `dc:description` remains the region caption
- **AND** the inherited page OCR is stored only as parent context metadata

#### Scenario: Page description remains the fallback when no crop caption exists
- **WHEN** a crop has no `mwg-rs:Name` caption and no other per-region caption source, but the parent page sidecar has a non-empty `dc:description`
- **THEN** the crop sidecar may fall back to the page-side description for `dc:description`
