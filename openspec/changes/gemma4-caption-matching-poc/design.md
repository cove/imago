## Context

The current region-detection pipeline uses Docling's standard image pipeline (layout + OCR) and then runs `associate_captions` — a geometry-based function that assigns caption text to photo regions by proximity. This association step frequently mis-assigns captions on complex album pages where photos and captions don't follow a simple nearest-neighbour layout.

Gemma4 running in LM Studio has shown strong understanding of album page layout. The prompt "Number the photos left-to-right, top-to-bottom; determine which caption goes with which photo; output `{photo-1: caption, ...}`" produces reliable caption assignments. The gap is that Gemma4's bounding boxes are imprecise, while Docling's boxes are accurate.

The docling-layout-heron model (`docling-project/docling-layout-heron` on Hugging Face) is a newer layout segmentation model that may improve bounding box precision over the current standard pipeline and is worth benchmarking on this test image before wiring anything into production.

## Goals / Non-Goals

**Goals:**
- Validate that the left-to-right/top-to-bottom sort order of Docling bounding boxes matches Gemma4's photo numbering convention, enabling a clean merge
- Validate docling-layout-heron bounding box quality vs. the current standard pipeline on `Family_1980-1985_B08_P16_V.jpg`
- Implement the Gemma4 caption-matching step as a reusable function that accepts a list of bounding boxes and an image, calls LM Studio, and returns `{photo_index: caption}` keyed by sort-order position
- Wire the new caption-matching function into the existing `_detect_regions_docling` path so the geometry-based `associate_captions` is no longer used for caption assignment

**Non-Goals:**
- Replacing `associate_captions` for non-docling region paths
- Full production rollout — this is a POC; the heron model evaluation result gates whether heron replaces the standard pipeline
- Changing the XMP output schema; `mwg-rs:Name` still holds the caption

## Decisions

### Decision: Merge strategy — sort Docling boxes, map to Gemma4 photo numbers

Gemma4 is prompted to number photos left-to-right then top-to-bottom. Neither the standard Docling pipeline nor Heron outputs bounding boxes in reading order — both return boxes in internal model representation order. A coordinate-based sort is therefore always required: group boxes into rows using a y-tolerance band (ratio 0.0–1.0 of image height), then sort rows top-to-bottom and boxes within each row left-to-right by x coordinate. This produces the same 1-based index Gemma4 uses, so `photo-N` maps directly to the Nth sorted box. No coordinate matching is needed.

**Alternative considered**: Pass Docling bounding boxes as context to Gemma4 and ask it to assign captions to box IDs. Rejected — this increases prompt complexity and LM Studio context; the sort-order convention is simpler and sufficient.

### Decision: POC script first, then integrate

A standalone POC script (`photoalbums/scripts/poc_caption_gemma4.py`) runs against the single test image, prints bounding boxes from both the standard pipeline and heron (if available), calls Gemma4, and shows the merged result. This lets us visually validate accuracy before touching the production code path.

**Alternative considered**: Integrate directly and add a feature flag. Rejected for POC scope — a script is faster to iterate and easier to discard if results are poor.

### Decision: Heron model is evaluated but not assumed

The POC tests heron if its weights are available locally; if not, it falls back to the standard pipeline for bounding boxes. The production integration decision (heron vs. standard) is deferred to after POC results are reviewed.

### Decision: OCR pass stripped from docling path once caption-matching is wired

Docling's OCR output is only used today to produce the caption text that `associate_captions` pulls from. Once Gemma4 handles caption assignment, the OCR sub-pipeline adds latency with no benefit. `PdfPipelineOptions` will be configured to disable OCR (`do_ocr=False`) on the docling path.

## Risks / Trade-offs

- [Gemma4 JSON parse failure] → Mitigation: fall back to empty captions (regions saved without captions) and log a warning; do not abort region detection
- [Heron model weights not available locally] → Mitigation: POC detects availability and logs a clear message; production path is unaffected
- [Sort-order mismatch on unusual layouts] → Mitigation: row-tolerance band handles slight vertical misalignment; ambiguous cases are flagged in the `caption_ambiguous` field already present on `RegionWithCaption`
- [LM Studio offline] → Mitigation: Gemma4 caption step is skipped (same pattern as existing LM Studio guard); regions are written without captions

## Open Questions

- What row-tolerance (pixels or fraction of image height) is appropriate for grouping photos into rows before left-to-right sort? Needs empirical check on test image. Answer: It should be a ratio number between 0.0 - 1.0, not pixels.
- Does heron require GPU or run acceptably on CPU? Needs measurement during POC. Answer: assume CPU.
- Should the POC script write a debug image with numbered bounding boxes overlaid to aid visual validation? Answer: Yes
