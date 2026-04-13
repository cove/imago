## ADDED Requirements

### Requirement: Docling is selectable as the view-region detection engine
The system SHALL support a `view_region_engine` config key in `ai_models.toml` with accepted values `lmstudio` (existing default) and `docling`. When `docling` is selected, the LM Studio vision API is not called; instead, docling's Python API processes the view image locally.

#### Scenario: Engine defaults to lmstudio when not configured
- **WHEN** `ai_models.toml` has no `view_region_engine` key
- **THEN** the system behaves identically to the existing implementation (Gemma4 via LM Studio)

#### Scenario: Docling engine selected
- **WHEN** `ai_models.toml` contains `view_region_engine = "docling"`
- **THEN** region detection uses docling's Python API and does not make any LM Studio HTTP calls

---

### Requirement: Docling detects photo regions via layout analysis
The system SHALL call docling's `DocumentConverter` on the view image and extract all items with label `DocItemLabel.PICTURE` as the detected photo regions. The resulting regions are written as a `mwg-rs:RegionList` to the view image's XMP sidecar using the existing `write_region_list()` path — identical output format to the LM Studio engine. The existing `crop_page_regions()` function reads from that XMP sidecar without any knowledge of which engine produced the regions.

#### Scenario: Multiple photos on a page
- **WHEN** docling processes a view image containing N distinct photograph prints
- **THEN** the system returns N `RegionResult` objects, one per `PICTURE` element, with pixel bounding boxes derived from docling's provenance data

#### Scenario: No pictures detected
- **WHEN** docling finds no `PICTURE` elements
- **THEN** the system returns an empty region list, logs a warning, writes no `mwg-rs:RegionList`, and writes the `view_regions` pipeline step with `result: "no_regions"` so subsequent runs skip detection and the cropper exits cleanly

#### Scenario: Coordinate extraction
- **WHEN** converting docling bounding boxes to pixel coordinates
- **THEN** the system uses docling's page size metadata to convert normalised or point coordinates to integer pixel values consistent with the source view image dimensions

---

### Requirement: Overlapping PICTURE elements are merged before validation
Album pages sometimes contain photos arranged in an artistic overlapping style. When docling returns two or more `PICTURE` elements whose bounding boxes overlap by more than 15% of the smaller element's area, the system SHALL merge them into a single bounding box (union of all merged elements) before passing to `validate_regions()`. This prevents the 5% overlap validation check from incorrectly rejecting intentional artistic arrangements.

Merging is applied iteratively until no remaining pairs exceed the threshold (matching the existing `_merge_boxes()` approach in `ai_page_layout.py`). The merged region carries no `caption_hint` from either source element; caption association runs after merging on the final region list.

#### Scenario: Two photos overlapping artistically
- **WHEN** docling detects two `PICTURE` elements that overlap by more than 15% of the smaller element's area
- **THEN** they are merged into a single `RegionResult` whose bounds are the union of both bounding boxes, and validation sees one region, not two

#### Scenario: Two photos with minor bounding-box imprecision
- **WHEN** two `PICTURE` elements overlap by 5% or less
- **THEN** no merge occurs; the overlap validation check in `validate_regions()` passes normally

---

### Requirement: Validation runs on docling output but does not trigger a retry
The system SHALL run the existing `validate_regions()` logic (overlap, full-page, zero-area, insufficient coverage checks) on the docling region list. If validation fails, the failures are logged at ERROR level and the regions are NOT written to XMP. No retry or fallback to the LM Studio engine occurs.

#### Scenario: Regions found and pass validation
- **WHEN** docling output passes all validation checks
- **THEN** regions are written to `mwg-rs:RegionList` and the `view_regions` pipeline step is written with `result: "regions_found"` and `model: "docling"`

#### Scenario: Validation failure
- **WHEN** docling output fails one or more validation checks
- **THEN** failures are logged with region index and reason; the region list is NOT written to the XMP sidecar; the `view_regions` pipeline step is written with `result: "validation_failed"` so subsequent runs skip re-detection (preventing an infinite retry loop since docling is deterministic); re-processing requires `--force`

---

### Requirement: Caption text is extracted from docling Text elements
The system SHALL collect all items with label `DocItemLabel.TEXT` from the docling output and associate each with photo regions using the following spatial rules. This logic is entirely application-level — docling provides only bounding boxes; caption-to-photo grouping is our own heuristic.

**Rule 1 — Centered page caption (broadcast):** A `TEXT` element whose horizontal centre falls within the middle third of the page width AND that is not within one text-line height of any single `PICTURE` boundary is treated as a page-level caption and broadcast to all regions with `caption_ambiguous = True`.

**Rule 2 — Grouped caption (proximity association):** A `TEXT` element that is within one text-line height of one or more `PICTURE` elements' boundaries is associated with the photo(s) whose horizontal span it overlaps. If it overlaps two or more photos equally, it is broadcast to those photos with `caption_ambiguous = True`.

**Rule 3 — No match:** A `TEXT` element that does not satisfy either rule is ignored.

Associated text is passed as `caption_hint` on each `RegionResult` so the existing `write_region_list()` stores it in the region XMP and `crop_page_regions()` picks it up via `resolve_region_caption()` without modification.

#### Scenario: Caption centered on the page (applies to all photos)
- **WHEN** a `TEXT` element is horizontally centred on the page and not adjacent to any single photo boundary
- **THEN** it is broadcast to all regions as `caption_hint` with `caption_ambiguous = True`

#### Scenario: Caption immediately below one photo
- **WHEN** a `TEXT` element is within one text-line height below a single `PICTURE` element and its horizontal span overlaps only that photo
- **THEN** it is assigned as `caption_hint` for that region only

#### Scenario: Caption below a group of photos sharing a row
- **WHEN** a `TEXT` element is within one text-line height below multiple `PICTURE` elements and its horizontal span overlaps all of them
- **THEN** it is broadcast to those regions with `caption_ambiguous = True`

#### Scenario: No qualifying text for a region
- **WHEN** no `TEXT` element satisfies either rule for a given `PICTURE` region
- **THEN** the region's `caption_hint` is set to an empty string

---

### Requirement: Crop XMP sidecars always carry page OCR text, and use it as caption fallback
The existing `_write_crop_sidecar()` in `ai_photo_crops.py` currently passes `ocr_text=""`. The system SHALL instead:

1. Always write the source view's `ocr_text` (from `view_state`) into `imago:OCRText` on the derived crop's XMP sidecar, giving downstream indexing full page context regardless of whether a per-region caption exists.
2. Use `imago:OCRText` as the caption (`dc:description`) fallback when the docling-extracted region caption is empty. Caption priority for the crop:
   1. docling `caption_hint` (non-empty)
   2. `imago:OCRText` from the source view (non-empty)
   3. `""` (empty)

This fallback is applied in `_write_crop_sidecar()` after `resolve_region_caption()` produces an empty result.

#### Scenario: Docling extracts a caption for the region
- **WHEN** the region has a non-empty `caption_hint` from docling
- **THEN** `dc:description` on the crop is set to that caption; `imago:OCRText` is written separately with the full page OCR text

#### Scenario: No docling caption, source view has OCR text
- **WHEN** `caption_hint` is empty and the source view XMP contains non-empty `ocr_text`
- **THEN** `dc:description` on the crop is set to the OCR text; `imago:OCRText` is also written with that same text

#### Scenario: No docling caption and no OCR text
- **WHEN** both `caption_hint` and source `ocr_text` are empty
- **THEN** `dc:description` is empty; `imago:OCRText` is empty; no error is raised
