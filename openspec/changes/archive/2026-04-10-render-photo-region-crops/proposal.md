## Why

Region detection identifies individual photo boundaries within a stitched view JPEG, but those boundaries only exist as XMP coordinate metadata — there is no way to view, index, or distribute the individual photos as standalone files. Cropping each detected region out of the CTM-corrected view image into its own JPEG closes this gap and makes individual historical photos first-class outputs of the pipeline.

## What Changes

- New `crop-regions` pipeline step that reads `mwg-rs:RegionList` from a view JPEG's XMP sidecar, crops each region from the CTM-corrected pixel data, and writes one JPEG per region to a `_Photos/` directory alongside `_Archive/` and `_View/`
- Output files follow the existing `_D##-##_V.jpg` naming convention: region index as the first number, `00` as the second, `_V` suffix to mark them as JPEG view outputs (e.g. `Egypt_1975_B00_P26_D01-00_V.jpg`)
- Each crop sidecar receives `xmpMM:DocumentID`, `xmpMM:DerivedFrom` (pointing to its source `_V.jpg`), `xmpMM:Pantry`, and the region's associated caption as `dc:description`
- Step runs inline per-page within the existing pipeline loop — not a second album pass
- A `--skip-crops` flag suppresses the step without breaking the rest of the pipeline
- Pipeline state tracked in `imago:Detections` so crops are not regenerated on subsequent runs unless `--force` is passed
- Standalone CLI command `photoalbums crop-regions` for running the step independently
- Crop generation only accepts page `_V.jpg` inputs; derived `_D##-##_V.jpg` render outputs are skipped and never treated as crop sources
- Regions whose clamped bounds collapse to zero area are skipped with a warning instead of failing the page

## Capabilities

### New Capabilities

- `photo-region-crops`: Crop detected MWG-RS photo regions from a CTM-corrected view JPEG, write each as `_D##-00_V.jpg` in `_Photos/`, assign DocumentID and DerivedFrom provenance, record pipeline state

### Modified Capabilities

- `scan-render-pipeline`: Add `crop-regions` as a pipeline step between `detect-regions` and `face-refresh`; add `--skip-crops` flag to suppress it

## Impact

- New `_Photos/` sibling directory created per album (alongside `_Archive/` and `_View/`)
- New `photoalbums/lib/ai_photo_crops.py` for region-to-crop extraction logic
- `stitch_oversized_pages.py` or `commands.py` updated to call crop step in the per-page pipeline loop
- `xmpmm_provenance.py` called for each crop to write DocumentID and DerivedFrom
- No changes to archive TIFs, view JPEGs, or region detection behavior
- New dependency: Pillow (already present) for pixel crop and JPEG write
