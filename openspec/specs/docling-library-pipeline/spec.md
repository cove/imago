## Purpose
Describe the Docling-backed page photo region pipeline that follows the Docling CLI standard image path and converts detected picture items into crop-ready page regions.

## Requirements

### Requirement: Docling detection uses the Docling library's standard image pipeline
The system SHALL process the album page image using the same Docling standard image pipeline that the `docling` command line tool uses for `--from image`. The runtime SHALL configure `DocumentConverter` for `InputFormat.IMAGE` with `PdfPipelineOptions` and `ImageDocumentBackend`. The raw HTTP call to LM Studio, the Gemma region-detection path, the custom `<doctag>` XML parser, and the Docling `VlmPipeline` are not part of this path.

#### Scenario: Standard image pipeline selected
- **WHEN** page photo region detection runs
- **THEN** the system calls `DocumentConverter.convert(image_path)` with the standard Docling image pipeline and returns `RegionResult` objects derived from the `DoclingDocument` output

### Requirement: Pipeline follows the CLI-standard image configuration
The system SHALL follow the Docling CLI standard image path rather than a custom VLM-specific configuration. Region detection SHALL use `PdfPipelineOptions` with Windows automatic GPU/CPU backend selection through Docling's normal accelerator settings. The required local OCR/layout assets are expected to already be available locally, and region detection SHALL NOT contact LM Studio or Hugging Face Hub at runtime.

#### Scenario: CLI-equivalent image path used
- **WHEN** the page region detector is initialized
- **THEN** it builds the same Docling image-processing path as the CLI standard pipeline instead of using `VlmConvertOptions.from_preset(...)`

#### Scenario: Local assets already present
- **WHEN** the machine is offline and the Docling OCR/layout assets are already present locally
- **THEN** the pipeline still runs without a Hugging Face Hub fetch

### Requirement: DoclingDocument picture items are mapped to RegionResult objects
The system SHALL iterate the items in the `DoclingDocument` returned by the converter, select all items with label `DocItemLabel.PICTURE`, and convert each item's bounding box to a `RegionResult`. Bounding boxes are accessed via the item's `prov` attribute in page pixel coordinates and MAY use `CoordOrigin.BOTTOMLEFT`; the system SHALL convert each bbox to top-left origin with `bbox.to_top_left_origin(page_height)` before building the pixel rectangle:

```text
left, top, right, bottom = bbox.to_top_left_origin(page_height).as_tuple()
x      = round(left)
y      = round(top)
width  = round(right - left)
height = round(bottom - top)
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

### Requirement: Docling and RapidOCR info logs are suppressed during region detection
The system SHALL suppress Docling-side `INFO` logging during page region detection so routine runs do not emit Docling or RapidOCR progress noise. Warnings and errors SHALL still be visible.

#### Scenario: Standard region detection runs successfully
- **WHEN** page photo region detection executes through the Docling standard image pipeline
- **THEN** Docling and RapidOCR `INFO` logs are not emitted by the region-detection wrapper

#### Scenario: Docling emits a warning or error
- **WHEN** the Docling region-detection path encounters a warning or error
- **THEN** that warning or error remains visible to the caller

### Requirement: Raw Docling output can be written as a separate debug JSON artifact
When view-region debug output is enabled, the system SHALL serialize the raw Docling result into a separate per-image debug JSON artifact outside the XMP sidecar. This artifact is for debugging and provenance only and SHALL NOT become part of the crop contract or replace the standard XMP region fields.

The debug artifact SHALL use the existing per-image debug area and be named distinctly from the prompt debug artifact so both can coexist for the same page.

#### Scenario: Debug output enabled for a Docling page
- **WHEN** the docling region-detection path runs with debug artifact output enabled
- **THEN** the system writes a separate JSON artifact containing the raw Docling result in the page's `_debug` area, and still writes only the canonical region/caption data to XMP

#### Scenario: Debug output disabled
- **WHEN** the docling region-detection path runs without debug artifact output enabled
- **THEN** the system does not write the raw Docling JSON artifact
