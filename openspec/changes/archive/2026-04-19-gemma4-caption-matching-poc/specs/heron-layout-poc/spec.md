## REMOVED Requirements

### Requirement: POC script runs docling-layout-heron and standard pipeline on test image and compares output
**Reason**: docling-layout-heron did not work out during POC evaluation. The standard Docling pipeline without OCR produces accurate bounding boxes and is sufficient. Heron integration is not pursued.
**Migration**: No migration needed — heron was never wired into the production pipeline.

### Requirement: POC script optionally writes a debug image with numbered bounding boxes overlaid
**Reason**: Superseded — the `--debug-image` flag was retained in the POC script but is now backed solely by standard-pipeline bounding boxes, not heron.
**Migration**: The `--debug-image` flag on `poc_caption_gemma4.py` continues to work against standard-pipeline regions. No action required.
