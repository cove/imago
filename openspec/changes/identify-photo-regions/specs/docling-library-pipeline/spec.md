## MODIFIED Requirements

### Requirement: Docling detection uses the Docling library's VlmPipeline
The system SHALL process the album page image using the Docling Python library's `DocumentConverter` configured with `VlmPipeline` and the selected preset's local model assets. The raw HTTP call to LM Studio, the Gemma region-detection path, and the custom `<doctag>` XML parser are not part of this path.

#### Scenario: Local Docling pipeline selected
- **WHEN** page photo region detection runs and the local model assets for the selected preset are available
- **THEN** the system calls `DocumentConverter.convert(image_path)` with a local Docling VLM pipeline and returns `RegionResult` objects derived from the `DoclingDocument` output

---

### Requirement: Pipeline is configured using a Docling preset name
The system SHALL configure the VLM pipeline via `VlmConvertOptions.from_preset(preset_name)` where `preset_name` comes from `ai_models.toml`. The selected preset SHALL resolve without contacting LM Studio or Hugging Face Hub during region detection; the required model assets are expected to already be available locally. The canonical target for this change is Windows with automatic GPU/CPU backend selection.

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

If a picture item carries a caption text (via associated caption item), the caption text SHALL be stored as the region's `mwg-rs:Name` value. If the implementation retains an internal hint for compatibility, it MAY mirror the same text into `imago:CaptionHint`. Top-level page `dc:description` remains the searchable page-text summary and SHALL NOT be repurposed as the per-region caption store.

#### Scenario: Two pictures detected
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE`
- **THEN** the system returns two `RegionResult` objects with pixel coordinates derived from each item's bounding box

#### Scenario: Picture with associated caption
- **WHEN** a picture item has an associated caption item with non-empty text
- **THEN** the resulting `RegionResult` has a non-empty `caption_hint` containing that text, and the serialized region metadata stores the same text in `mwg-rs:Name`

#### Scenario: No picture items in document
- **WHEN** the `DoclingDocument` contains no items with `DocItemLabel.PICTURE`
- **THEN** the system returns an empty list and logs a WARNING

---

### Requirement: Raw Docling output can be written as a separate debug JSON artifact
When view-region debug output is enabled, the system SHALL serialize the raw Docling result into a separate per-image debug JSON artifact outside the XMP sidecar. This artifact is for debugging and provenance only and SHALL NOT become part of the crop contract or replace the standard XMP region fields.

The debug artifact SHALL use the existing per-image debug area and be named distinctly from the prompt debug artifact so both can coexist for the same page.

#### Scenario: Debug output enabled for a Docling page
- **WHEN** the docling region-detection path runs with debug artifact output enabled
- **THEN** the system writes a separate JSON artifact containing the raw Docling result in the page's `_debug` area, and still writes only the canonical region/caption data to XMP

#### Scenario: Debug output disabled
- **WHEN** the docling region-detection path runs without debug artifact output enabled
- **THEN** the system does not write the raw Docling JSON artifact
