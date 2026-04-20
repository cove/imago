## Purpose
Record the evaluation outcome of the docling-layout-heron POC. Heron integration was explored as an alternative layout detection approach alongside the standard Docling pipeline.

## Requirements

### Requirement: POC evaluation concluded — heron integration not pursued
The heron layout approach (docling-layout-heron) was evaluated during the gemma4-caption-matching POC. It did not produce results superior to the standard Docling pipeline without OCR, and heron integration is not pursued in the production pipeline.

The standard Docling pipeline with `do_ocr=False` produces accurate bounding boxes and is the selected approach. The `--debug-image` flag on `poc_caption_gemma4.py` continues to work against standard-pipeline regions.

#### Scenario: Standard pipeline selected
- **WHEN** region detection runs in production
- **THEN** the standard Docling image pipeline (not heron) is used for bounding box extraction
