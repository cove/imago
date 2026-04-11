## Context

The pipeline (`photoalbum-scan-render-pipeline`) produces a page `_V.jpg` per page and, after `detect-regions`, stores per-photo bounding boxes in MWG-RS `mwg-rs:RegionList` XMP. Region coordinates are normalised centre-point (`stArea:`) coordinates. `xmpmm_provenance.py` (from the parent pipeline change) provides `assign_document_id`, `write_derived_from`, and `write_pantry_entry`.

The `_View/` and `_Archive/` directory naming pattern is established by `get_view_dirname` in `stitch_oversized_pages.py`. A `_Photos/` directory follows the same sibling pattern. Derived file naming uses `_D##-##_V.jpg` per AGENTS.md.

## Goals / Non-Goals

**Goals:**
- Read MWG-RS region coordinates from the page view XMP, crop the corresponding pixel rectangle from the page `_V.jpg`, and write one `_D##-00_V.jpg` per region into `_Photos/`
- Assign `xmpMM:DocumentID` and write `xmpMM:DerivedFrom` + `xmpMM:Pantry` pointing to the source page `_V.jpg` at crop-creation time
- Copy the region's `dc:description` caption (if any) to the crop sidecar
- Track step completion in `pipeline.crop_regions` in the page view's `imago:Detections`; skip on subsequent runs unless `--force`
- Integrate as an inline per-page step in `render-pipeline` between `detect-regions` and `face-refresh`, with a `--skip-crops` bypass flag
- Preserve unrelated existing crop-sidecar fields when re-cropping a crop that keeps the same path on rerun

**Non-Goals:**
- Archival TIF crops (crops are JPEG view outputs only)
- Re-running region detection if no regions exist (crop step skips silently if `mwg-rs:RegionList` is absent or empty)
- Scaling or padding crops (write exact pixel rectangle)
- Cropping derived `_D##-##_V.jpg` outputs

## Decisions

### Output directory is `_Photos/`, same naming root as `_View/` and `_Archive/`

`get_photos_dirname(archive_path)` mirrors `get_view_dirname`: replace `_Archive` suffix with `_Photos`. This keeps all three sibling directories discoverable with the same naming logic.

### File naming: `_D{region_index:02d}-00_V.jpg`

Region index is 1-based (matching MWG-RS `mwg-rs:Name` values `photo_1`, `photo_2`, ...). The `-00` second slot is reserved for future sub-crops or variants; `_V` marks it as a JPEG view output per AGENTS.md. Example: `Egypt_1975_B00_P26_D01-00_V.jpg`.

### Crop step runs per page inline, not as a second album pass

`crop_page_regions(view_path, photos_dir, *, force=False)` is called inside the same per-page loop that calls render, detect-regions, face-refresh, and ctm-apply. This avoids a second album scan and keeps all page work grouped together.

### Pipeline state is tracked on the page view sidecar, not on each crop

`pipeline.crop_regions` is written to the page `_V.jpg` sidecar when all crops for that page complete. If any crop fails mid-page, the state is not written and the step re-runs on the next pipeline invocation. Individual crop sidecars do not carry a crop-step `pipeline` key.

### MWG-RS normalised coords convert to pixel rectangles for crop

`mwg-rs:stArea:x/y/w/h` are centre-point normalised (0-1). Convert to top-left pixel rect: `left = (cx - w/2) * img_w`, `top = (cy - h/2) * img_h`, etc. Clamp to image bounds before cropping.

### Crops are cut from raw pixels; face refresh runs before CTM apply

`crop-regions` reads and writes raw page-view pixel data. `face-refresh` then runs on the raw crop JPEGs. `ctm-apply` runs later and can apply a per-photo `crs:ColorMatrix1` stored in each crop's own XMP sidecar without requiring the crop to be regenerated or its face metadata to be recomputed.

### Crop sidecars receive creation-time provenance and are updated in place on rerun

The crop sidecar is created if missing and otherwise updated in place. The crop writer owns the fields it writes but preserves unrelated existing XMP fields when a crop with the same path is regenerated. It writes:
- `xmpMM:DocumentID`
- `xmpMM:DerivedFrom` + `xmpMM:Pantry` pointing to the source page `_V.jpg`
- `dc:description` from the region's best available caption (if non-empty)
- `dc:source` = relative path to the source `_V.jpg`
- inherited page-level location/date/subject metadata

`--force` removes orphaned crop files and sidecars whose paths are no longer produced by the current region set, but matching crop paths are regenerated in place so unrelated/manual sidecar fields survive.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| No regions detected for a page | Crop step skips silently; no crops written; pipeline continues |
| Region bounds extend beyond image edge | Clamp rect to `[0, img_w] x [0, img_h]` before crop; log a warning if clamp was significant (>5% of dimension) |
| Existing crops present but regions have changed | `--force` removes orphaned crop files for outputs that no longer correspond to a region while preserving matching sidecars in place |
| `_Photos/` directory missing on first run | `mkdir(parents=True, exist_ok=True)` on first write |
| Face-refresh skipped for crops | Crop JPEGs remain valid outputs; CTM can still be applied later and the page failure summary will show the face-refresh failure |
