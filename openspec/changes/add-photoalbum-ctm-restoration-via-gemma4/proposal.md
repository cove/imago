## Why

Some historical photoalbum scans exhibit red-shift degradation caused by cyan pigment failure. Imago needs a repeatable, metadata-first restoration workflow that can estimate and persist a reusable 3×3 Color Transformation Matrix (CTM) from stitched images so restorations stay non-destructive, portable, and consistent across rendering tools.

## What Changes

- Add a CTM-generation workflow for stitched photoalbum images that uses a local Gemma 4 model in LM Studio to estimate a 3×3 color restoration matrix.
- Persist the resulting matrix in XMP using Adobe Camera Raw-compatible `crs:ColorMatrix1` metadata rather than destructively rewriting archival master scans.
- Support manifest-style XMP metadata that can also carry stitch provenance data such as source ingredients and homography matrices.
- Add a job-capable CLI and MCP workflow to generate CTMs for a page, album, or album set and review stored CTM results.
- Update the photoalbum render pipeline so stored CTMs can be applied as a deterministic linear transform before final stitched/rendered output is written.

## Capabilities

### New Capabilities
- `photoalbum-ctm-restoration`: Generate, persist, inspect, and apply CTM-based chromatic restoration metadata for stitched photoalbum images using a Gemma 4 model workflow.

### Modified Capabilities
- `view-xmp-regions`: Ensure existing view-image XMP metadata can coexist with Camera Raw CTM metadata and archive-manifest fields in the same XMP documents.

## Impact

- Affected photoalbum AI model configuration and LM Studio integration.
- New photoalbum library code for CTM estimation, validation, storage, and application.
- XMP helper changes for `crs:ColorMatrix1`, manifest fields, and coexistence with existing metadata.
- New or updated CLI and MCP job surfaces for CTM generation and review.
- Render pipeline changes for deterministic CTM application during stitched image output.
