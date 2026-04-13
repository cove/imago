## ADDED Requirements

### Requirement: Docling detection uses the existing LM Studio infrastructure
When a view-region model whose name contains the substring `"docling"` (case-insensitive) is active, the system SHALL send the image to LM Studio using the same HTTP infrastructure as the existing Gemma4 path, but with a docling conversion prompt and a `<doctag>` response parser instead of the JSON bounding-box prompt and JSON parser. No new Python dependencies are required.

#### Scenario: Docling model selected in ai_models.toml
- **WHEN** the active `view_region_model` resolves to a model name containing `"docling"` (e.g. `granite-docling-258m`)
- **THEN** the system uses the docling prompt and `<doctag>` parser; the LM Studio `response_format` JSON schema is NOT sent (the model returns plain text)

#### Scenario: Non-docling model selected
- **WHEN** the active model name does not contain `"docling"`
- **THEN** the system behaves identically to the existing implementation (JSON bounding-box prompt, JSON parser)

---

### Requirement: The docling prompt requests a doctag conversion
The system SHALL send the prompt `"Convert this page to docling."` as the user message (with the image attached) when the docling path is active. No system prompt is required; the model is prompted only with the image and this single instruction.

---

### Requirement: The doctag response is parsed into RegionResult objects
The system SHALL parse the `<doctag>…</doctag>` XML returned by the model. Each `<picture>` element produces one `RegionResult`. Coordinates are carried as four consecutive `<loc_X>` child tags in order: **top, left, bottom, right** (ymin, xmin, ymax, xmax) on a 0–500 normalized scale (where 500 represents the full image dimension). Example: `<loc_68><loc_21><loc_250><loc_199>` = top=68, left=21, bottom=250, right=199. The system SHALL convert these to pixel coordinates using:

```
left_px   = round(loc_left   / 500 * img_w)
top_px    = round(loc_top    / 500 * img_h)
right_px  = round(loc_right  / 500 * img_w)
bottom_px = round(loc_bottom / 500 * img_h)

x      = left_px
y      = top_px
width  = right_px - left_px
height = bottom_px - top_px
```

If a `<picture>` contains a `<caption>` child element, the caption's text content is used as `caption_hint` for that region.

#### Scenario: Four pictures, one with an embedded caption
- **WHEN** the model returns four `<picture>` elements and one has a `<caption>` child
- **THEN** the system produces four `RegionResult` objects; the region with the caption has a non-empty `caption_hint`; the others have empty `caption_hint`

#### Scenario: Malformed or missing doctag wrapper
- **WHEN** the response does not contain a `<doctag>` element or contains no `<picture>` elements
- **THEN** the system returns an empty region list and logs a WARNING

---

### Requirement: Paragraph elements provide additional caption hints
`<paragraph>` elements in the `<doctag>` response that are within one text-line height of a `<picture>` boundary and whose horizontal span overlaps the picture SHALL be associated with that picture as `caption_hint` (if the picture has no embedded `<caption>`). If a paragraph overlaps two or more pictures equally, it is broadcast to all with `caption_ambiguous = True`.

A centered paragraph (horizontal centre within the middle third of the page width) that is not adjacent to any single picture is broadcast to all regions with `caption_ambiguous = True`.

---

### Requirement: Overlapping picture elements are merged before validation
When the parsed region list contains two or more `RegionResult` objects whose bounding boxes overlap by more than 15% of the smaller region's area, the system SHALL merge them into a single union bounding box. Merging is applied iteratively until no pairs exceed the threshold. The merged region carries no `caption_hint`; caption association runs on the final merged list.

#### Scenario: Two photos overlapping artistically
- **WHEN** two `<picture>` elements overlap by more than 15% of the smaller area
- **THEN** they are merged into one `RegionResult` whose bounds are the union; validation sees one region

#### Scenario: Minor bounding-box imprecision
- **WHEN** two regions overlap by 5% or less
- **THEN** no merge occurs; the existing overlap validation passes normally

---

### Requirement: Validation and pipeline step follow the same contract as the lmstudio engine
The system SHALL run `validate_regions()` on the docling result list. Pipeline step outcomes:

| Outcome | Pipeline step written |
|---|---|
| Regions found, validation passes | `result: "regions_found"`, `model: <model-name>` |
| No `<picture>` elements in response | `result: "no_regions"` |
| Validation fails | `result: "validation_failed"` (prevents infinite re-runs; `--force` required to retry) |

The docling path does NOT retry with a repair prompt — if validation fails, the result is recorded and the user must intervene.

---

### Requirement: Crop XMP sidecars always carry page OCR text, and use it as caption fallback
The existing `_write_crop_sidecar()` in `ai_photo_crops.py` currently passes `ocr_text=""`. The system SHALL instead:

1. Always write the source view's `ocr_text` (from `view_state`) into `imago:OCRText` on the derived crop's XMP sidecar.
2. Set `dc:description` to the region caption (from `resolve_region_caption()`); if that is empty, fall back to the `ocr_text` value.

Caption priority for `dc:description`:
1. Region caption from `resolve_region_caption()` (non-empty)
2. Source view `ocr_text` (non-empty)
3. `""` (empty)

#### Scenario: Region has a docling caption
- **WHEN** the region's `caption_hint` resolves to a non-empty caption
- **THEN** `dc:description` is set to that caption; `imago:OCRText` is written with the page OCR text

#### Scenario: No caption, source view has OCR text
- **WHEN** `resolve_region_caption()` returns empty and source `ocr_text` is non-empty
- **THEN** `dc:description` is set to the OCR text; `imago:OCRText` is written with the same text

#### Scenario: No caption and no OCR text
- **WHEN** both are empty
- **THEN** both fields are empty; no error is raised
