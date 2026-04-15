## MODIFIED Requirements

### Requirement: Docling detection uses the Docling library's VlmPipeline
When a view-region model whose name contains the substring `"docling"` (case-insensitive) is active, the system SHALL process the album page image using the Docling Python library's `DocumentConverter` configured with `VlmPipeline` and the selected preset's local model assets. The raw HTTP call to LM Studio and the custom `<doctag>` XML parser are not part of this path.

#### Scenario: Docling model selected, local assets available
- **WHEN** the active `view_region_model` resolves to a model name containing `"docling"` and the local model assets for the selected preset are available
- **THEN** the system calls `DocumentConverter.convert(image_path)` with a local Docling VLM pipeline and returns `RegionResult` objects derived from the `DoclingDocument` output

#### Scenario: Non-docling model selected
- **WHEN** the active model name does not contain `"docling"`
- **THEN** the system uses the existing JSON bounding-box path unchanged; no Docling library call is made

---

### Requirement: Pipeline is configured using a Docling preset name
The system SHALL configure the VLM pipeline via `VlmConvertOptions.from_preset(preset_name)` where `preset_name` comes from `ai_models.toml`. The selected preset SHALL resolve without contacting LM Studio or Hugging Face Hub during region detection; the required model assets are expected to already be available locally.

#### Scenario: Preset name resolved from config
- **WHEN** the docling model config specifies `preset = "granite_docling"`
- **THEN** the pipeline is created with `VlmConvertOptions.from_preset("granite_docling")` and the correct prompt format and response parser for that model are applied automatically

#### Scenario: Local assets already present
- **WHEN** the machine is offline and the Docling model assets for the selected preset are already present locally
- **THEN** the pipeline still runs without a Hugging Face Hub fetch

---

### Requirement: DoclingDocument picture items are mapped to RegionResult objects
The system SHALL iterate the items in the `DoclingDocument` returned by the converter, select all items with label `DocItemLabel.PICTURE`, and convert each item's bounding box to a `RegionResult`. Bounding boxes are accessed via the item's `prov` attribute in page-relative normalized coordinates (0-1 scale); the system SHALL convert to pixel coordinates using:

```
x      = round(bbox.l * img_w)
y      = round(bbox.t * img_h)
width  = round((bbox.r - bbox.l) * img_w)
height = round((bbox.b - bbox.t) * img_h)
```

If a picture item carries a caption text (via associated caption item), the caption text SHALL be stored as the region's standard `dc:description`. If the implementation retains an internal hint for compatibility, it MAY mirror the same text into `imago:CaptionHint`.

#### Scenario: Two pictures detected
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE`
- **THEN** the system returns two `RegionResult` objects with pixel coordinates derived from each item's bounding box

#### Scenario: Picture with associated caption
- **WHEN** a picture item has an associated caption item with non-empty text
- **THEN** the resulting `RegionResult` has a non-empty `caption_hint` containing that text, and the serialized region metadata stores the same text in `dc:description`

#### Scenario: No picture items in document
- **WHEN** the `DoclingDocument` contains no items with `DocItemLabel.PICTURE`
- **THEN** the system returns an empty list and logs a WARNING

