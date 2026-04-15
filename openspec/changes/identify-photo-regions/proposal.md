## Why

The current page-region path doesn't work reliably for photo boundaries. Docling produces better photo boundary detection and caption extraction than the Gemma-based path, making it the right model for this layout work. The page-region stage still needs to run from local model assets and not depend on Hugging Face or LM Studio at runtime.

## What Changes

- Use Docling for page-level photo region detection and caption extraction.
- Require the Docling branch to run from local model assets, without LM Studio or Hugging Face Hub calls during region detection.
- Keep the non-Docling scene text detection path unchanged.
- Preserve the existing XMP pipeline-step contract and validation behavior for Docling results.
- Keep downstream photo-text OCR as a separate concern; LM Studio is no longer part of region detection.

## Capabilities

### New Capabilities
- None

### Modified Capabilities
- `docling-library-pipeline`: change the Docling region pipeline to use local model assets through the Docling library directly.
- `docling-region-detection`: update the Docling-specific detection contract to reflect the offline/local library pipeline and its output.
- `view-xmp-regions`: keep the page-side XMP region list as the single source of truth for crop boundaries consumed by the crop step.

## Impact

- Affected code: `photoalbums/lib/_docling_pipeline.py`, `photoalbums/lib/ai_view_regions.py`, related tests, and the Docling-related AI model configuration.
- Dependency impact: Docling remains required, but the Docling region-detection path must not depend on a running LM Studio server or Hugging Face Hub access at runtime.
- Behavior impact: Docling region detection becomes the local source of truth for crop-ready photo regions and caption hints before crop generation.
