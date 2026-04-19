## Why

Docling's OCR-based caption association is unreliable for photo album pages — it misidentifies which caption belongs to which photo. Gemma4 via LM Studio has proven far more accurate at this task, and the docling-layout-heron model may deliver better bounding boxes than the current standard pipeline, making this a good moment to POC both improvements together on a known test image.

## What Changes

- **New**: POC script that runs docling-layout-heron for layout segmentation on a single test image (`Family_1980-1985_B08_P16_V.jpg`) and logs detected bounding boxes
- **New**: POC script step that calls Gemma4 via LM Studio with the numbered-photo caption-matching prompt and parses the JSON response
- **New**: Merge step that sorts docling-heron bounding boxes left-to-right/top-to-bottom (matching Gemma4's numbering convention) and assigns Gemma4's captions to the corresponding regions
- **Modified**: `docling-region-detection` spec — OCR/caption-proximity association is replaced by Gemma4 caption matching; docling's role narrows to layout/bounding-box detection only

## Capabilities

### New Capabilities
- `gemma4-caption-matching`: Sends album page image to Gemma4 via LM Studio with numbered-photo prompt; parses `{"photo-1": "caption", ...}` JSON response; assigns captions to regions by matching left-to-right/top-to-bottom sort order with docling's detected bounding boxes
- `heron-layout-poc`: Loads docling-layout-heron model locally and runs it against the test image to evaluate bounding box quality vs. current standard Docling pipeline

### Modified Capabilities
- `docling-region-detection`: OCR pass and caption proximity association (`associate_captions` geometry matching) are removed from the docling path; docling is used only for picture-item bounding box detection

## Impact

- `photoalbums/lib/ai_view_regions.py` — `associate_captions` geometry matching is replaced by Gemma4 caption assignment for the docling code path
- `photoalbums/lib/_docling_pipeline.py` — OCR pipeline options may be stripped to layout-only
- `photoalbums/lib/_caption_lmstudio.py` — new prompt skill for numbered caption matching
- `photoalbums/lib/ai_model_settings.py` — new setting for Gemma4 caption-matching model
- New POC script (not wired into main pipeline) to validate heron model + Gemma4 merge on test image
- LM Studio dependency is already present; docling-layout-heron requires a model weight download
