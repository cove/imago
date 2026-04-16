## Why

The current page-region path doesn't work reliably for photo boundaries. Docling produces better photo boundary detection and caption extraction than the Gemma-based path, making it the right model for this layout work. The page-region stage still needs to run from local model assets and not depend on Hugging Face or LM Studio at runtime.

## What Changes

- Use Docling for page-level photo region detection and caption extraction.
- Remove the Gemma/LM Studio region-detection path for page photo boundaries.
- Require the Docling branch to run from local model assets, without LM Studio or Hugging Face Hub calls during region detection.
- Run the Docling backend on Windows with automatic GPU/CPU selection.
- Retry Docling only for runtime failures, and record a distinct `failed` pipeline result if those retries are exhausted.
- Write the raw Docling output as a separate per-image debug JSON artifact when region-debug output is enabled.
- Preserve the existing XMP pipeline-step contract and validation behavior for Docling results.
- Keep downstream photo-text OCR as a separate concern; LM Studio is no longer part of region detection.
- Store per-region photo captions in `mwg-rs:Name`, while keeping the page-level top-level `dc:description` focused on searchable OCR and scene text for the page view.

## Capabilities

### New Capabilities
- None. This change repairs and extends existing region-detection capabilities rather than adding a separate top-level workflow or tool.

### Modified Capabilities
- `docling-library-pipeline`: change the Docling region pipeline to use local model assets through the Docling library directly, and optionally emit a separate raw Docling debug artifact.
- `docling-region-detection`: update the Docling-specific detection contract to reflect the offline/local library pipeline and its output.
- `view-xmp-regions`: keep the page-side XMP region list as the single source of truth for crop boundaries consumed by the crop step.

## Impact

- Affected code: `photoalbums/lib/_docling_pipeline.py`, `photoalbums/lib/ai_view_regions.py`, `photoalbums/lib/xmp_sidecar.py`, `photoalbums/lib/ai_photo_crops.py`, related tests, and the Docling-related AI model configuration.
- Dependency impact: Docling remains required, but the region-detection path must no longer depend on LM Studio or Hugging Face Hub access at runtime; the canonical backend target is Windows with automatic GPU/CPU selection.
- Behavior impact: Docling region detection becomes the only supported source of crop-ready page photo regions. Optional debug runs also preserve the raw Docling output in a separate JSON artifact outside XMP, region captions are stored in `mwg-rs:Name`, and the page view's top-level `dc:description` remains a searchable OCR/scene-text summary.
