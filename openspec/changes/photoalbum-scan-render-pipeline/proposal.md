## Why

The scan-to-viewable-JPEG workflow spans several ad-hoc steps — CTM colour correction, photo-region detection, render, face-region refresh, and provenance metadata — that currently have no single entry point, no defined order, and no shared provenance chain linking rendered outputs back to their archive sources. This change unifies those steps into one ordered CLI pipeline, adds the missing XMP provenance fields (`xmpMM:DocumentID`, `xmpMM:DerivedFrom`, `xmpMM:Pantry`) across all rendered outputs, and ensures each step runs exactly once in the right sequence.

## What Changes

- New `render-pipeline` CLI command that runs the full scan-to-view pipeline in order for a page, album, or album set
- CTM colour-restoration workflow: estimate a 3×3 matrix from the stitched view image via Gemma 4, persist in `_Archive/` XMP (`crs:ColorMatrix1`), apply deterministically during render
- Photo-region detection (Gemma 4 vision, already implemented as `view-region-detection`) wired into the pipeline order after CTM estimation and before derived-image render
- Render-time face-region refresh: replace inherited person-identifying XMP regions on rendered outputs with a fresh `buffalo_l` + Cast pass against the rendered pixels
- `xmpMM:DocumentID` written to every rendered output (view JPEG and derived JPEGs) on first creation
- `xmpMM:DerivedFrom` written on `_D##` derived images referencing their source archive scan(s), and on any region-split images referencing their parent view image
- `xmpMM:Pantry` populated with the referenced-document entries needed for `DerivedFrom` links

## Capabilities

### New Capabilities

- `scan-render-pipeline`: Top-level CLI pipeline that runs stitch/render → region detection → face refresh → provenance metadata in the correct order for a page, album, or album set
- `page-stitch-render`: Stitch one or more archive TIF scans into a single view JPEG (or convert a single scan / derived media file directly); apply stored CTM; validate output; skip existing outputs
- `ctm-color-restoration`: Estimate a CTM from a stitched view image via Gemma 4, persist `crs:ColorMatrix1` in `_Archive/` XMP; apply the matrix to already-rendered JPEGs via a separate `ctm-apply` step so colour correction can be applied retroactively without re-stitching
- `render-face-region-refresh`: After each rendered JPEG is written, rerun Cast-backed `buffalo_l` face identification against the rendered pixels and replace any inherited person-identifying `ImageRegion` entries with the fresh result
- `xmpmm-provenance`: Write `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` on rendered view and derived JPEG outputs to record the archival lineage chain

### Modified Capabilities

- `view-region-detection`: Wire the existing detect-regions step into the `scan-render-pipeline` at the correct position (after CTM estimation, before derived render); no requirement changes to the detection or XMP-write behavior itself

## Impact

- New `render-pipeline` subcommand added to `photoalbums.py` CLI
- New `photoalbums/lib/ai_ctm_restoration.py` for CTM estimation, validation, and application
- New `photoalbums/lib/xmpmm_provenance.py` for DocumentID, DerivedFrom, and Pantry write helpers
- `xmp_sidecar.py` extended with Camera Raw and `xmpMM` namespace support
- `stitch_oversized_pages.py` updated for CTM application post-stitch and face-region refresh post-render
- Existing `ai_view_regions.py` and `detect-view-regions` CLI step unchanged; pipeline command calls them in sequence
- New dependency: none (Pillow, `insightface`/buffalo_l, LM Studio already present)
- XMP sidecars for `_View/*.jpg` and derived `_D##*.jpg` outputs gain provenance fields on next render
