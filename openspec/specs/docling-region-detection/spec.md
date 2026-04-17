## Purpose
Define the runtime contract for Docling-based page photo region detection, including offline execution, validation outcomes, and failure behavior.

## Requirements

### Requirement: Docling detection runs from local model assets without runtime fetches
The system SHALL process the album page image with the Docling Python library directly using the standard image pipeline assets that are already available locally. Page photo region detection SHALL NOT require LM Studio or Hugging Face Hub access at runtime, and the Gemma/LM Studio path is removed for this workflow.

#### Scenario: Local Docling assets are available
- **WHEN** page photo region detection runs and the required Docling OCR/layout assets are already available locally
- **THEN** the system returns `RegionResult` objects derived from the `DoclingDocument` output without making a network request for model weights

#### Scenario: LM Studio unavailable
- **WHEN** LM Studio is offline or not configured
- **THEN** the docling branch still runs because it does not depend on LM Studio connectivity

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
