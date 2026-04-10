## Context

Imago renders archival TIFF scans into viewable JPEGs via `stitch_oversized_pages.py`. Several post-render enrichment steps exist in separate specs and partial implementations:

- **CTM colour restoration** (`ai_ctm_restoration.py`, `commands.py run_ctm`) — fully implemented; estimates a 3×3 colour matrix from a stitched view image and applies it during the next render pass. Steps: (1) run `photoalbums ctm generate` to store CTM in `_Archive/` XMP; (2) re-run render which picks up and applies the CTM.
- **Photo region detection** (`ai_view_regions.py`, `commands.py run_detect_view_regions`) — fully implemented; detects per-photo bounding boxes in `_V.jpg` via Gemma 4 vision and writes MWG-RS `RegionList` XMP. Invoked via `photoalbums detect-view-regions`.
- **Render-time face refresh** — not yet implemented; needs to re-run Cast-backed `buffalo_l` against rendered pixels and replace person-identifying `ImageRegion` entries.
- **xmpMM provenance** (`xmp_sidecar.py` has `xmpMM` namespace registered, `History` helpers exist) — `DocumentID`, `DerivedFrom`, `Pantry` fields not yet written to rendered outputs. No dedicated provenance module exists.

There is no single CLI command that ties these into a correct, reproducible sequence. Each step has to be invoked separately, in the wrong order, or is simply skipped.

## Goals / Non-Goals

**Goals:**
- Add a `render-pipeline` CLI subcommand to `photoalbums.py` that runs: render → detect-regions → face-refresh → provenance metadata for a page, album, or album set
- Implement render-time face-region refresh using the existing Cast-backed `buffalo_l` path from `ai_index_runner.py`
- Implement `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` in a new dedicated `xmpmm_provenance.py` module
- CTM generation (`photoalbums ctm generate`) remains a separate, explicit pre-step; the pipeline only applies a stored CTM during render

**Non-Goals:**
- MCP endpoints for the pipeline (CLI-only for now)
- Retroactively migrating already-rendered files outside a normal render run
- Auto-generating CTM as part of the pipeline run (CTM generation requires manual review before re-render)
- Training or fine-tuning any model
- Region-split photo creation (future step after regions are validated)

## Decisions

### CTM generation is a manual pre-step; the pipeline applies whatever is stored

CTM generation requires operator review before applying — baking it into the pipeline would silently produce unchecked colour corrections. The boundary is: `photoalbums ctm generate` (human-reviewed, run explicitly) → `photoalbums render-pipeline` (applies whatever is stored in any sidecar it encounters).

CTM operates at two independent levels:
- **Page level**: generated from the stitched `_V.jpg`; stored in `_Archive/` XMP as `crs:ColorMatrix1`; applied to the `_V.jpg` in-place
- **Per-photo level**: generated from an individual crop JPEG in `_Photos/`; stored in that crop's own XMP sidecar as `crs:ColorMatrix1`; applied to that crop JPEG in-place

Each photo on a page may have been shot with different film stock, lighting, or exposure, so a single page-level CTM may overcorrect some and undercorrect others. Per-photo CTM lets the AI calibrate each image independently.

`ctm generate` accepts a `--per-photo` flag to run on crop JPEGs rather than the page view. Both levels are fully independent: you can have a page CTM, per-photo CTMs, both, or neither.

### CTM application is a separate pipeline step, placed after crop-regions

The render step writes raw (uncorrected) pixels. Crops are then made from those raw pixels. `ctm-apply` runs last among the output-creation steps — after crops exist — so it can apply corrections at both page and per-photo level in a single pass.

Pipeline position: render → detect-regions → crop-regions → **ctm-apply** → face-refresh → provenance

This ordering means:
1. Crops are always cut from the same raw stitched pixels, regardless of what CTM exists
2. CTM can be generated from the raw crop (accurate, uncontaminated by prior correction)
3. Re-running `ctm-apply` after regenerating a CTM corrects the JPEG without re-stitching or re-cropping

`ctm-apply` in a single invocation:
- Reads `crs:ColorMatrix1` from the archive XMP → applies to `_V.jpg` if present
- Reads `crs:ColorMatrix1` from each crop's own XMP sidecar → applies to that crop if present
- Tracks completion per-file in `pipeline.ctm_applied` on each JPEG's sidecar

The existing `_apply_archive_ctm_if_present` calls inside `stitch()`, `tif_to_jpg()`, and `derived_to_jpg()` are removed.

Alternative: apply CTM between render and detect-regions so detection sees corrected pixels. Rejected because (a) region boundaries are geometry-based and not significantly affected by colour cast, and (b) placing CTM after crops allows per-photo calibration which is the primary motivation.

### Region detection runs on raw stitched pixels

Region boundaries are determined by image geometry (content discontinuity, perspective changes) rather than colour accuracy. Running detection on raw pixels before CTM means coordinates are computed once and remain stable regardless of how many times CTM is regenerated and re-applied.

Alternative: detect after CTM. Rejected — it would couple region coordinates to CTM application and require re-detection whenever CTM changes.

### Face refresh runs after each rendered output exists, replacing only person-identifying regions

Reuse `_process_people_update` from `ai_index_runner.py` via a narrow render-time entrypoint rather than adding a second face-matching implementation. The refresh removes only `ImageRegion` entries whose `Iptc4xmpExt:RCtype` is `face-*` and replaces them with the fresh Cast result. Non-face regions (photo regions from MWG-RS, etc.) are untouched.

Alternative: rewrite entire `ImageRegion` bag. Rejected — it would delete photo-region metadata from previous steps.

### Pipeline step completion is tracked in the imago:Detections JSON blob under a `pipeline` key

Each AI-backed step records its completion in the `imago:Detections` JSON blob on the relevant XMP sidecar. The archive sidecar holds `pipeline.ctm`; the rendered view sidecar holds `pipeline.view_regions`, `pipeline.face_refresh`, and `pipeline.provenance`. Each entry stores at minimum `completed` (ISO timestamp) and `model` (where applicable).

This is the sole gate for skipping a step — the pipeline reads the `pipeline` subkey, checks if the step's entry is present, and skips if so (unless `--force`). This is preferred over checking for output data existence (e.g. presence of `mwg-rs:RegionList`) because it is explicit: a step that ran but produced zero regions still records its entry, and a step whose output was hand-edited does not trigger an unwanted re-run.

`--force` clears the `pipeline` entries for all steps before re-running, so the next run behaves as if the steps have never executed.

Pipeline state is written/read via helpers in a new `pipeline_state.py` module (or within `xmpmm_provenance.py` — see module decision below).

Alternative: use output-data presence (e.g. `mwg-rs:RegionList` present → skip). Rejected because it cannot distinguish "ran and found nothing" from "never ran", and makes `--force` semantics harder to define cleanly.

### Each AI step is also a standalone CLI command

Every AI-backed step (`ctm generate`, `detect-view-regions`, `face-refresh`, `write-provenance`) runs the same pipeline-state check/write logic whether invoked directly or via `render-pipeline`. This means the pipeline command is purely orchestration — it calls the same underlying function that the standalone command calls.

### xmpMM provenance lives in a dedicated `xmpmm_provenance.py` module

All DocumentID, DerivedFrom, and Pantry logic is isolated in `photoalbums/lib/xmpmm_provenance.py`. `xmp_sidecar.py` provides the low-level XML read/write primitives; `xmpmm_provenance.py` owns the provenance-specific logic and is the only caller of those primitives for provenance fields.

Alternative: add helpers directly to `xmp_sidecar.py`. Rejected — provenance logic is a distinct concern and `xmp_sidecar.py` is already large.

### `xmpMM:DocumentID` is assigned at the point of creation, not at the end of the pipeline

Archive scan sidecars receive their `DocumentID` inside `_ensure_archive_page_sidecar` (the first thing the render step touches). Rendered JPEG sidecars receive their `DocumentID` immediately after the JPEG is written. This means every file has a stable ID from the moment it exists, and the final provenance step only needs to write `DerivedFrom` + `Pantry` — it never needs to back-fill missing IDs.

The ID is a UUID (`xmp:uuid:{uuid4}`), written once and never regenerated, since it identifies the conceptual document across re-renders.

Alternative: write all IDs in the provenance step at the end. Rejected — creates a window where files exist without IDs and requires ordering guarantees inside the provenance step.

### `xmpMM:DerivedFrom` uses the primary archive scan as the source reference for view JPGs

For `_V.jpg` outputs: DerivedFrom references the `xmpMM:DocumentID` of the primary archive scan (`_S01.tif`), which already exists by the time DerivedFrom is written. For `_D##-##_V.jpg` derived outputs: references the source derived TIF or media file.

The `xmpMM:Pantry` bag stores one entry per unique DerivedFrom source, letting any XMP-aware tool resolve the reference chain without accessing the archive directory.

### Pipeline command order

```
photoalbums render-pipeline <album_id> [--page N] [--photos-root PATH] [--force]
```

Runs for every matching page in sequence:
1. **render** — `_ensure_archive_page_sidecar` (assigns DocumentID to archive scan) → stitch (no CTM) → write `_V.jpg` + `_D##-##_V.jpg` (assigns DocumentID to each rendered output immediately after write)
2. **detect-regions** — detect photo regions on raw `_V.jpg`, write MWG-RS XMP; skip if `pipeline.view_regions` recorded
3. **crop-regions** *(from render-photo-region-crops change)* — crop each detected region from raw `_V.jpg` pixels → `_Photos/`; assigns DocumentID to each crop immediately after write; skip if `pipeline.crop_regions` recorded
4. **ctm-apply** — for `_V.jpg`: read `crs:ColorMatrix1` from archive XMP, apply in-place, record `pipeline.ctm_applied`; for each crop in `_Photos/`: read `crs:ColorMatrix1` from crop XMP sidecar, apply in-place, record `pipeline.ctm_applied`; skip per-file if already recorded; skip silently if no CTM stored
5. **face-refresh** — run buffalo_l + Cast against `_V.jpg` and each crop in `_Photos/`, replace person face regions; skip per-file if `pipeline.face_refresh` recorded
6. **provenance** — write `xmpMM:DerivedFrom` + `xmpMM:Pantry` on `_V.jpg` and crop sidecars; skip per-file if `pipeline.provenance` recorded

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Face refresh drops historically correct inherited names | Intended; rendered sidecar reflects the rendered image, not inherited archive guesses |
| Pipeline re-renders pages unnecessarily on second run | Render step already has skip-if-exists logic; `--force` needed to re-render |
| DerivedFrom DocumentID references an archive scan that has no DocumentID yet | DocumentID is written to archive scan sidecars inside `_ensure_archive_page_sidecar` at render time, before any derived output is written |
| xmpMM:Pantry grows unbounded for albums with many re-renders | Pantry stores one entry per unique source DocumentID, deduplicated on write |
| Buffalo_l Cast store not available at render time | Face refresh logs a warning and skips; render output is still written; non-face regions preserved |
