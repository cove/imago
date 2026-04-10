## Context

The pipeline (`photoalbum-scan-render-pipeline`) produces a `_V.jpg` per page and, after `detect-regions`, stores per-photo bounding boxes in MWG-RS `mwg-rs:RegionList` XMP. The `ctm-apply` step corrects colours in the `_V.jpg` before detection. Region coordinates are normalised centre-point (MWG-RS `stArea:` format). `xmpmm_provenance.py` (from the parent pipeline change) provides `assign_document_id`, `write_derived_from`, and `write_pantry_entry`.

The `_View/` and `_Archive/` directory naming pattern is established by `get_view_dirname` in `stitch_oversized_pages.py`. A `_Photos/` directory follows the same sibling pattern. Derived file naming uses `_D##-##_V.jpg` per AGENTS.md.

## Goals / Non-Goals

**Goals:**
- Read MWG-RS region coordinates from the view XMP, crop the corresponding pixel rectangle from the `_V.jpg`, and write one `_D##-00_V.jpg` per region into `_Photos/`
- Assign `xmpMM:DocumentID` and write `xmpMM:DerivedFrom` + `xmpMM:Pantry` pointing to the source `_V.jpg`
- Copy the region's `dc:description` caption (if any) to the crop sidecar
- Track step completion in `pipeline.crop_regions` in the view JPEG's `imago:Detections`; skip on subsequent runs unless `--force`
- Integrate as an inline per-page step in `render-pipeline` between `detect-regions` and `face-refresh`, with a `--skip-crops` bypass flag
- Provide a standalone `crop-regions` CLI command

**Non-Goals:**
- Archival TIF crops (crops are JPEG view outputs only)
- Re-running region detection if no regions exist (crop step skips silently if `mwg-rs:RegionList` is absent)
- Scaling or padding crops (write exact pixel rectangle)
- Copying face or non-caption XMP fields to crop sidecars (face-refresh runs after and handles that)

## Decisions

### Output directory is `_Photos/`, same naming root as `_View/` and `_Archive/`

`get_photos_dirname(archive_path)` mirrors `get_view_dirname`: replace `_Archive` suffix with `_Photos`. This keeps all three sibling directories discoverable with the same naming logic.

Alternative: sub-directory inside `_View/`. Rejected — crops are a distinct output class, not a variant of the view.

### File naming: `_D{region_index:02d}-00_V.jpg`

Region index is 1-based (matching MWG-RS `mwg-rs:Name` values `photo_1`, `photo_2`, …). The `-00` second slot is reserved for future sub-crops or variants; `_V` marks it as a JPEG view output per AGENTS.md. Example: `Egypt_1975_B00_P26_D01-00_V.jpg`.

Alternative: flat numbering without `-00` slot. Rejected — inconsistent with existing `_D##-##` convention.

### Crop step runs per-page inline, not as a second album pass

`crop_page_regions(view_path, photos_dir, *, force=False)` is called inside the same per-page loop that calls render, ctm-apply, detect-regions, face-refresh, and provenance. This avoids a second album scan and keeps all page work grouped together.

### Pipeline state is tracked on the view JPEG's sidecar, not on each crop

`pipeline.crop_regions` is written to the `_V.jpg` sidecar when all crops for that page complete. If any crop fails mid-page, the state is not written and the step re-runs on the next pipeline invocation. Individual crop sidecars do not carry a `pipeline` key.

Alternative: track per-crop state on each crop sidecar. Rejected — a per-page state on the view sidecar is sufficient and avoids scattering pipeline tracking across N sidecars.

### MWG-RS normalised coords converted to pixel rectangle for crop

`mwg-rs:stArea:x/y/w/h` are centre-point normalised (0–1). Convert to top-left pixel rect: `left = (cx - w/2) * img_w`, `top = (cy - h/2) * img_h`, etc. Clamp to image bounds before cropping.

### Crops are cut from raw pixels; CTM is applied by the subsequent ctm-apply step

`crop-regions` reads and writes raw (pre-CTM) pixel data. The `ctm-apply` step that follows can then apply a per-photo `crs:ColorMatrix1` stored in each crop's own XMP sidecar. This means each photo gets individually calibrated colour correction: the operator (or the AI) can run `ctm generate --per-photo` on the crop, store a CTM in the crop's sidecar, and `ctm-apply` will correct that crop independently of any page-level correction.

### Crop sidecar gets DocumentID, DerivedFrom, Pantry, caption, and inherited page metadata

The crop sidecar is written fresh (not copied from the view sidecar). It receives:
- `xmpMM:DocumentID` via `assign_document_id`
- `xmpMM:DerivedFrom` + `xmpMM:Pantry` via `write_derived_from` / `write_pantry_entry` pointing to the `_V.jpg`
- `dc:description` from the region's caption (if non-empty)
- `dc:source` = relative path to the source `_V.jpg`
- **Inherited from view sidecar** (read via `read_ai_sidecar_state`):
  - `exif:GPSLatitude` / `exif:GPSLongitude`
  - `photoshop:City` / `photoshop:State` / `photoshop:Country`
  - `Iptc4xmpExt:Sublocation`
  - `Iptc4xmpExt:LocationCreated`
  - `Iptc4xmpExt:LocationShown` (bag — same shown-location entries as the page)
  - `xmp:CreateDate` and `dc:date` (the page date; each crop depicts a photo from the same time)
  - `dc:subject` keywords (genre/topic tags apply equally to all photos on the page)

Location, date, and subject fields are page-level metadata that apply equally to every photo on the page. Propagating them at crop time avoids a separate copy step and makes each crop sidecar a self-contained record.

No `crs:ColorMatrix1` is written at crop time — that comes from `ctm generate --per-photo` run separately. Face regions are not copied — `face-refresh` runs after `ctm-apply` and writes them fresh from the (now colour-corrected) crop pixels.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| No regions detected for a page | Crop step skips silently; no crops written; pipeline continues |
| Region bounds extend beyond image edge | Clamp rect to `[0, img_w] × [0, img_h]` before crop; log a warning if clamp was significant (>5% of dimension) |
| Existing crops present but regions have changed | `--force` deletes all `_D##-00_V.jpg` files and sidecars for the page before re-cropping, preventing orphaned crops from previous runs with different region counts |
| `_Photos/` directory missing on first run | `mkdir(parents=True, exist_ok=True)` on first write |
| Face-refresh skipped for crops | Face-refresh already iterates all rendered JPEGs including those in `_Photos/`; no special handling needed |
