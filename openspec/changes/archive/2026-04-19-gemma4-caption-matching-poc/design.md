## Context

The current region-detection pipeline uses Docling's standard image pipeline (layout + OCR) and then runs `associate_captions` — a geometry-based function that assigns caption text to photo regions by proximity. This association step frequently mis-assigns captions on complex album pages where photos and captions don't follow a simple nearest-neighbour layout.

A vision-capable LM Studio model has shown strong understanding of album page layout. The prompt instructs the model to number photos left-to-right/top-to-bottom, then assign captions to photo numbers and return a JSON object — producing reliable caption assignments. The standard Docling pipeline (without OCR) produces accurate bounding boxes and is retained for layout detection.

## Goals / Non-Goals

**Goals:**
- Implement the LM Studio caption-matching step as a reusable module that sorts bounding boxes into reading order, calls the configured model, and returns captions keyed by sort-order position
- Wire the caption-matching module into the existing `_detect_regions_docling` path so the geometry-based `associate_captions` is no longer used for caption assignment
- Configure the caption-matching model via `ai_models.toml` (same pattern as `view_region_model`)
- Strip the OCR sub-pipeline from the docling path since it is no longer needed

**Non-Goals:**
- Replacing `associate_captions` for non-docling region paths
- Changing the XMP output schema; `mwg-rs:Name` still holds the caption

## Decisions

### Decision: Merge strategy — coordinate-based sort maps to model's photo numbers

The LM Studio model is prompted to number photos left-to-right then top-to-bottom. Docling bounding boxes do not come out in reading order — they are in internal model representation order. A coordinate-based scanline sort is therefore always applied: group boxes into rows using a y-tolerance band (ratio 0.0–1.0 of image height, default 0.10), sort rows top-to-bottom and boxes within each row left-to-right by x. This produces the same 1-based index the model uses, so `photo-N` maps directly to the Nth sorted box.

**Alternative considered**: Pass bounding box coordinates to the model and ask it to assign captions to box IDs. Rejected — more prompt complexity; the sort-order convention is simpler and sufficient.

### Decision: Model configured via `ai_models.toml`, not env var

Caption-matching model selection uses the same `ai_models.toml` alias pattern as `view_region_model`. A new `caption_matching_model` key points to a model alias defined in the `[models]` table. This keeps all model configuration in one place and avoids runtime env-var dependency.

**Alternative considered**: Env var `IMAGO_GEMMA4_CAPTION_MODEL`. Rejected — inconsistent with how other models are configured; tightly coupled to a specific model name.

### Decision: Prompt structure — example JSON first, then field notes

The prompt opens with a backtick-fenced JSON example using empty-string placeholders, followed by bullet-point field notes, and closes with "Just return the JSON without any extra text or explanation." This structure matches the established SKILL.md output-format pattern and prevents the model from wrapping the response in prose.

The prompt also instructs the model to prepend subject context when a caption refers to subjects shown in an adjacent photo (e.g. "GILBERT & HELEN — Their new home"), so captions remain self-contained when viewed in isolation.

### Decision: OCR pass stripped from docling path

Docling's OCR output was only used to produce caption text that `associate_captions` pulled from. With LM Studio handling caption assignment, the OCR sub-pipeline adds latency with no benefit. `PdfPipelineOptions` is configured with `do_ocr=False`.

## Risks / Trade-offs

- [LM Studio JSON parse failure] → Mitigation: fall back to empty captions (regions saved without captions) and log a WARNING; region detection is not aborted
- [Sort-order mismatch on unusual layouts] → Mitigation: row-tolerance band handles slight vertical misalignment; value is configurable
- [LM Studio offline] → Mitigation: caption step degrades gracefully; regions are written with empty captions so layout detection is never blocked by caption assignment availability
- [Model returns more/fewer captions than regions] → Mitigation: surplus keys ignored; missing keys produce empty caption string

## Open Questions

- What row-tolerance (pixels or fraction of image height) is appropriate for grouping photos into rows before left-to-right sort? Answer: ratio 0.0–1.0, default 0.10.
- Should the diagnostic script write a debug image with numbered bounding boxes overlaid to aid visual validation? Answer: Yes, via `--debug-image` flag.
