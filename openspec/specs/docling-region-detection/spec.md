## Purpose
Define the runtime contract for Docling-based page photo region detection, including offline execution, validation outcomes, and failure behavior.

## Requirements

### Requirement: Docling detection runs from local model assets without runtime fetches
The system SHALL process the album page image with the Docling Python library directly using the standard image pipeline assets that are already available locally. Page photo region detection SHALL NOT require LM Studio or Hugging Face Hub access at runtime for the layout step. LM Studio caption matching is the integrated caption assignment step for this pipeline; if LM Studio is unavailable at runtime, the system degrades gracefully by writing regions with empty captions rather than failing layout detection.

#### Scenario: Local Docling assets are available
- **WHEN** page photo region detection runs and the required Docling layout assets are already available locally
- **THEN** the system returns `RegionResult` objects derived from the `DoclingDocument` output without making a network request for model weights

#### Scenario: LM Studio unavailable
- **WHEN** LM Studio is offline or no caption-matching model is configured
- **THEN** the docling layout step still runs and regions are written with empty captions; layout detection is not blocked

### Requirement: Docling pipeline runs with OCR disabled
The system SHALL configure `PdfPipelineOptions` with `do_ocr=False` for the standard image pipeline path. OCR output is no longer needed — captions are assigned by the LM Studio caption-matching step — and running OCR adds latency with no benefit.

#### Scenario: Standard image pipeline runs
- **WHEN** the Docling standard image pipeline processes an album page
- **THEN** the pipeline runs with `do_ocr=False` and RapidOCR is not invoked

### Requirement: DoclingDocument picture items are mapped to RegionResult objects
The system SHALL iterate the items in the `DoclingDocument` returned by the converter, select all items with label `DocItemLabel.PICTURE`, and convert each item's bounding box to a `RegionResult`. Bounding boxes are accessed via the item's `prov` attribute in page pixel coordinates and MAY use `CoordOrigin.BOTTOMLEFT`; the system SHALL convert each bbox to top-left origin with `bbox.to_top_left_origin(page_height)` before building the pixel rectangle:

```text
left, top, right, bottom = bbox.to_top_left_origin(page_height).as_tuple()
x      = round(left)
y      = round(top)
width  = round(right - left)
height = round(bottom - top)
```

Caption text for each region SHALL be populated by the LM Studio caption-matching step (see `gemma4-caption-matching` spec) rather than from Docling caption items. The `mwg-rs:Name` field on each region SHALL contain the LM Studio-assigned caption. The `imago:CaptionHint` field MAY mirror the same value for compatibility.

#### Scenario: Two pictures detected, LM Studio assigns captions
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE` and the configured LM Studio model returns captions for both
- **THEN** the system returns two `RegionResult` objects with pixel coordinates derived from each item's bounding box and captions assigned by the model

#### Scenario: Two pictures detected, LM Studio offline
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE` and LM Studio is unavailable
- **THEN** the system returns two `RegionResult` objects with empty `caption_hint` fields

#### Scenario: No picture items in document
- **WHEN** the `DoclingDocument` contains no items with `DocItemLabel.PICTURE`
- **THEN** the system returns an empty list and logs a WARNING

### Requirement: Validation and pipeline step follow the same contract as the local Docling engine
The system SHALL run `validate_regions()` on the docling result list. Pipeline step outcomes:

| Outcome | Pipeline step written |
|---|---|
| Regions found, validation passes | `result: "regions_found"`, `model: <model-name>` |
| No picture items in document | `result: "no_regions"` |
| Validation fails | `result: "validation_failed"` |
| Runtime failure retries exhausted | `result: "failed"` |

The docling path retries only runtime failures. It does NOT retry with a repair prompt, and it does NOT retry validation failures. If validation fails, the result is recorded and the user must intervene.

The docling path SHALL follow the standard Docling image pipeline behavior that the CLI uses, including its multi-picture extraction and page-space bbox conventions, before the local validation gate is applied.

#### Scenario: Valid regions returned by Docling pipeline
- **WHEN** the `DoclingDocument` contains valid picture items that pass `validate_regions()`
- **THEN** the pipeline step records `result: "regions_found"` and the `RegionResult` list is written to XMP

#### Scenario: Docling pipeline returns empty document
- **WHEN** the converter returns a `DoclingDocument` with no picture items
- **THEN** the pipeline step records `result: "no_regions"` and no crops are written

#### Scenario: Runtime failure eventually succeeds
- **WHEN** the Docling pipeline raises a runtime error on an early attempt but succeeds on a later retry
- **THEN** the system continues and records the eventual successful outcome instead of `failed`

#### Scenario: Runtime failures exhaust retries
- **WHEN** the Docling pipeline keeps failing at runtime after the configured retry limit
- **THEN** the pipeline step records `result: "failed"`, no regions are written, and the underlying error is preserved in logs/debug output
