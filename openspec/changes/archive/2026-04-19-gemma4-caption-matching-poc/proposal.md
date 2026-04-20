## Why

Docling's OCR-based caption association is unreliable for photo album pages — it misidentifies which caption belongs to which photo. A vision-capable LM Studio model is far more accurate at this task, and the standard Docling pipeline (without OCR) produces reliable bounding boxes, making it straightforward to combine the two.

## What Changes

- **New**: `_caption_matching.py` library module — sorts detected regions into reading order (coordinate-based scanline sort) and calls a configured LM Studio model to assign captions via a structured numbered-photo prompt; parses `{"photo-1": "caption", ...}` JSON response
- **New**: Diagnostic script (`caption_matching_debug.py`) that runs the standard Docling pipeline on a single image, calls LM Studio for caption matching, and prints the merged result with optional debug image output
- **New**: `caption_matching_model` key in `ai_models.toml` — selects which configured model alias to use for caption matching
- **Modified**: `docling-region-detection` spec — OCR pass removed (`do_ocr=False`); caption proximity association (`associate_captions`) replaced by LM Studio caption matching; docling's role narrows to layout/bounding-box detection only

## Capabilities

### New Capabilities
- `lmstudio-caption-matching`: Sends album page image to a configured LM Studio model with a numbered-photo prompt; parses `{"photo-1": "caption", ...}` JSON response; assigns captions to regions by matching coordinate-based reading-order sort with docling's detected bounding boxes; handles context-carryover (adjacent photo subjects prepended when a caption refers to a sibling)

### Modified Capabilities
- `docling-region-detection`: OCR pass and caption proximity association (`associate_captions` geometry matching) are removed from the docling path; docling is used only for picture-item bounding box detection

## Impact

- `photoalbums/lib/_caption_matching.py` — new module: `sort_regions_reading_order`, `call_lmstudio_caption_matching`, `assign_captions_from_lmstudio`
- `photoalbums/lib/ai_view_regions.py` — `associate_captions` geometry matching replaced by LM Studio caption assignment for the docling code path
- `photoalbums/lib/_docling_pipeline.py` — `do_ocr=False` added to `PdfPipelineOptions`
- `photoalbums/lib/ai_model_settings.py` — `caption_matching_model` loaded from `ai_models.toml`; `default_caption_matching_model()` added
- `photoalbums/ai_models.toml` — `caption_matching_model` key added
- `photoalbums/scripts/caption_matching_debug.py` — diagnostic script for inspecting caption matching output on a specific image
