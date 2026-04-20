## Why

The scan-to-viewable-JPEG workflow spans several ad-hoc steps - render, photo-region detection, crop generation, face-region refresh, CTM colour correction, and provenance metadata - that currently have no single entry point, no stable ordering, and no explicit contract for retries, empty AI results, or page-level failure reporting. This change unifies those steps into one ordered CLI pipeline, moves provenance writes to file-creation time so lineage is never lost if a later step fails, and makes XMP mutations preserve unrelated sidecar fields instead of treating sidecars as disposable outputs.

## What Changes

- New `render-pipeline` CLI command that runs the full scan-to-view pipeline in order for a page, album, or album set
- Pipeline order defined as: render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply
- Provenance (`xmpMM:DocumentID`, `xmpMM:DerivedFrom`, `xmpMM:Pantry`) written at the earliest file-creation point for rendered views, derived renders, and crop sidecars
- Photo-region detection remains scoped to page `_V.jpg` view images; `_D##` derived outputs are never region-detected or cropped
- Empty AI region detection is a valid success case with an explicit "no regions found" signal in pipeline state
- End-of-run failure summary reports which pages and steps failed after the pipeline completes

## Capabilities

### New Capabilities

- `scan-render-pipeline`: Top-level CLI pipeline that runs stitch/render -> region detection -> crop-regions -> face refresh -> CTM apply in the correct order for a page, album, or album set
- `page-stitch-render`: Stitch one or more archive TIF scans into a single view JPEG (or convert a single scan / derived media file directly), assign provenance at creation time, validate output, and skip existing outputs
- `ctm-color-restoration`: Estimate a CTM from a stitched view image via Gemma 4, persist `crs:ColorMatrix1` in `_Archive/` XMP, and apply the matrix later with `ctm-apply` so colour correction can be re-run without re-stitching, re-cropping, or re-running face refresh
- `render-face-region-refresh`: Run Cast-backed `buffalo_l` face identification against raw rendered page/crop pixels, replace only person-identifying `ImageRegion` entries, and preserve unrelated XMP fields
- `xmpmm-provenance`: Write `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` when each new output file is created so archival lineage exists even if a later pipeline step fails

### Modified Capabilities

- `view-region-detection`: Wire the existing detect-regions step into `scan-render-pipeline` after render and before crop generation; detection runs only on page `_V.jpg` images and records an explicit no-regions result when appropriate

## Impact

- New `render-pipeline` subcommand added to `photoalbums.py` CLI
- New `photoalbums/lib/xmpmm_provenance.py` for DocumentID, DerivedFrom, and Pantry helpers used at file-creation time
- `stitch_oversized_pages.py` updated so render writes provenance immediately, face refresh runs before CTM application, and CTM is no longer applied during render
- Existing `ai_view_regions.py` and `detect-view-regions` CLI step gain explicit empty-result tracking and result validation before XMP write
- Crop generation stays limited to page views in `_Photos/`; derived `_D##-##_V.jpg` outputs are not region-detected or cropped
- XMP sidecar writers update the canonical sidecar in place and preserve unrelated/manual fields on reruns
