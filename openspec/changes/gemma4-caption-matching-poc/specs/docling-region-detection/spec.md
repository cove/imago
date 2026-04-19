## MODIFIED Requirements

### Requirement: Docling detection runs from local model assets without runtime fetches
The system SHALL process the album page image with the Docling Python library directly using the standard image pipeline assets that are already available locally. Page photo region detection SHALL NOT require LM Studio or Hugging Face Hub access at runtime for the layout step. The Gemma4/LM Studio path is used only for the caption-assignment step that follows layout detection and is optional — if LM Studio is unavailable, regions are written without captions.

#### Scenario: Local Docling assets are available
- **WHEN** page photo region detection runs and the required Docling layout assets are already available locally
- **THEN** the system returns `RegionResult` objects derived from the `DoclingDocument` output without making a network request for model weights

#### Scenario: LM Studio unavailable
- **WHEN** LM Studio is offline or not configured
- **THEN** the docling layout step still runs and regions are written; Gemma4 caption assignment is skipped and regions are saved with empty captions

## MODIFIED Requirements

### Requirement: DoclingDocument picture items are mapped to RegionResult objects
The system SHALL iterate the items in the `DoclingDocument` returned by the converter, select all items with label `DocItemLabel.PICTURE`, and convert each item's bounding box to a `RegionResult`. Bounding boxes are accessed via the item's `prov` attribute in page pixel coordinates and MAY use `CoordOrigin.BOTTOMLEFT`; the system SHALL convert each bbox to top-left origin with `bbox.to_top_left_origin(page_height)` before building the pixel rectangle:

```text
left, top, right, bottom = bbox.to_top_left_origin(page_height).as_tuple()
x      = round(left)
y      = round(top)
width  = round(right - left)
height = round(bottom - top)
```

Caption text for each region SHALL be populated by the Gemma4 caption-matching step (see `gemma4-caption-matching` spec) rather than from Docling caption items. The `mwg-rs:Name` field on each region SHALL contain the Gemma4-assigned caption. The `imago:CaptionHint` field MAY mirror the same value for compatibility.

#### Scenario: Two pictures detected, Gemma4 assigns captions
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE` and Gemma4 returns captions for both
- **THEN** the system returns two `RegionResult` objects with pixel coordinates derived from each item's bounding box and captions assigned by Gemma4

#### Scenario: Two pictures detected, LM Studio offline
- **WHEN** the `DoclingDocument` contains two items with `DocItemLabel.PICTURE` and LM Studio is unavailable
- **THEN** the system returns two `RegionResult` objects with empty `caption_hint` fields

#### Scenario: No picture items in document
- **WHEN** the `DoclingDocument` contains no items with `DocItemLabel.PICTURE`
- **THEN** the system returns an empty list and logs a WARNING

## REMOVED Requirements

### Requirement: Docling and RapidOCR info logs are suppressed during region detection
**Reason**: OCR is disabled on the docling path (`do_ocr=False`), so RapidOCR no longer runs during region detection. The log-suppression requirement for RapidOCR is moot. Docling INFO suppression remains in effect but no longer needs to specifically call out RapidOCR.
**Migration**: The log-suppression wrapper remains for Docling INFO logs; the RapidOCR-specific note is removed from the requirement. No consumer action required.
