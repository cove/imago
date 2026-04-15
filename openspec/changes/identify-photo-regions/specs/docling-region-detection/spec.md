## REMOVED Requirements

### Requirement: Docling detection uses the existing LM Studio infrastructure
**Reason**: Page-region detection now runs directly through Docling from local model assets, so LM Studio is no longer part of the boundary-finding contract.
**Migration**: Use the Docling library path directly for region detection; no LM Studio server setup is required for this branch.

---

### Requirement: The docling prompt requests a doctag conversion
**Reason**: The Docling library constructs and manages its own prompt internally based on the selected preset. The caller no longer specifies a prompt string.
**Migration**: No action required; the preset continues to define the Docling prompt internally.

---

### Requirement: The doctag response is parsed into RegionResult objects
**Reason**: `_docling_parser.py` and all `<doctag>` XML parsing logic are removed. The Docling library returns a structured `DoclingDocument`; bounding boxes are read from item `prov` attributes directly.
**Migration**: See `docling-library-pipeline` spec for the `DoclingDocument` → `RegionResult` mapping.

---

### Requirement: Paragraph elements provide additional caption hints
**Reason**: The paragraph proximity algorithm in `_docling_parser.py` is removed along with the parser. Caption hints now come from Docling's native caption item association, and downstream crop metadata consumes those hints directly.
**Migration**: No additional paragraph pass is required for this branch.

---

### Requirement: Overlapping picture elements are merged before validation
**Reason**: The merge pass compensated for model output artifacts in the raw HTTP path. The Docling library pipeline is expected to produce crop-ready picture items directly.
**Migration**: If overlapping regions are observed from the pipeline output, re-add the merge pass as a separate post-processing step in `_docling_pipeline.py`.

---

## ADDED Requirements

### Requirement: Docling detection runs from local model assets without runtime fetches
When a view-region model whose name contains the substring `"docling"` (case-insensitive) is active, the system SHALL process the album page image with the Docling Python library directly using model assets that are already available locally. The page-region branch SHALL NOT require LM Studio or Hugging Face Hub access at runtime.

#### Scenario: Local Docling assets are available
- **WHEN** the active `view_region_model` resolves to a model name containing `"docling"` and the selected preset's model assets are already available locally
- **THEN** the system returns `RegionResult` objects derived from the `DoclingDocument` output without making a network request for model weights

#### Scenario: LM Studio unavailable
- **WHEN** LM Studio is offline or not configured
- **THEN** the docling branch still runs because it does not depend on LM Studio connectivity

---

### Requirement: Validation and pipeline step follow the same contract as the local Docling engine
The system SHALL run `validate_regions()` on the docling result list. Pipeline step outcomes:

| Outcome | Pipeline step written |
|---|---|
| Regions found, validation passes | `result: "regions_found"`, `model: <model-name>` |
| No picture items in document | `result: "no_regions"` |
| Validation fails | `result: "validation_failed"` (prevents infinite re-runs; `--force` required to retry) |

The docling path does NOT retry with a repair prompt. If validation fails, the result is recorded and the user must intervene.

#### Scenario: Valid regions returned by Docling pipeline
- **WHEN** the `DoclingDocument` contains valid picture items that pass `validate_regions()`
- **THEN** the pipeline step records `result: "regions_found"` and the `RegionResult` list is written to XMP

#### Scenario: Docling pipeline returns empty document
- **WHEN** the converter returns a `DoclingDocument` with no picture items
- **THEN** the pipeline step records `result: "no_regions"` and no crops are written

