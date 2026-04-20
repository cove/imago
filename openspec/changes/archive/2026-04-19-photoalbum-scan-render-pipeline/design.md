## Context

Imago renders archival TIFF scans into viewable JPEGs via `stitch_oversized_pages.py`. Several post-render enrichment steps exist in separate specs and partial implementations:

- **CTM colour restoration** (`ai_ctm_restoration.py`, `commands.py run_ctm`) - fully implemented for generation; estimates a 3x3 colour matrix from a stitched view image and stores it in `_Archive/` XMP
- **Photo region detection** (`ai_view_regions.py`, `commands.py run_detect_view_regions`) - fully implemented; detects per-photo bounding boxes in page `_V.jpg` images via Gemma 4 vision and writes MWG-RS `RegionList` XMP
- **Render-time face refresh** - not yet implemented; needs to re-run Cast-backed `buffalo_l` against rendered page and crop pixels and replace person-identifying `ImageRegion` entries
- **xmpMM provenance** (`xmp_sidecar.py` has `xmpMM` namespace registered, `History` helpers exist) - `DocumentID`, `DerivedFrom`, and `Pantry` fields are not yet written when outputs are created

There is no single CLI command that ties these into a correct, reproducible sequence. Each step has to be invoked separately, in the wrong order, or is simply skipped. The current artifacts also leave several operational questions underspecified: what counts as a successful empty AI result, how partial page failures are reported, and how reruns preserve unrelated sidecar edits.

## Goals / Non-Goals

**Goals:**
- Add a `render-pipeline` CLI subcommand to `photoalbums.py` that runs: render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply for a page, album, or album set
- Write provenance at the earliest possible creation point for archive sidecars, rendered JPEGs, and crop sidecars
- Preserve unrelated XMP sidecar fields, including manual edits, when a pipeline step reruns and updates only its owned fields
- Treat "AI found no regions" as an explicit successful outcome rather than an error or an implicit absence of work
- Report per-page failures immediately and again in a summary once the job completes

**Non-Goals:**
- MCP endpoints for the pipeline (CLI-only for now)
- Retroactively migrating already-rendered files outside a normal render run
- Auto-generating CTM as part of the pipeline run (CTM generation remains an explicit reviewed pre-step)
- Region detection or crop generation for derived `_D##-##_V.jpg` outputs
- Training or fine-tuning any model

## Decisions

### CTM generation is a manual pre-step; CTM application runs last in the pipeline

CTM generation requires operator review before applying - baking it into the pipeline would silently produce unchecked colour corrections. The boundary is: `photoalbums ctm generate` (human-reviewed, run explicitly) -> `photoalbums render-pipeline` (applies whatever is stored in any sidecar it encounters).

Pipeline position: render -> detect-regions -> crop-regions -> face-refresh -> **ctm-apply**

This ordering means:
1. Region detection and face refresh both work from the same raw rendered pixels
2. Re-running `ctm-apply` after regenerating a CTM does not require re-detecting regions, re-cropping, or re-running face refresh
3. Per-photo CTM remains possible because crops already exist by the time `ctm-apply` runs

`ctm-apply` in a single invocation:
- Reads `crs:ColorMatrix1` from the archive XMP -> applies to `_V.jpg` if present
- Reads `crs:ColorMatrix1` from each crop's own XMP sidecar -> applies to that crop if present
- Tracks completion per file in `pipeline.ctm_applied` on each JPEG sidecar

The existing `_apply_archive_ctm_if_present` calls inside `stitch()`, `tif_to_jpg()`, and `derived_to_jpg()` are removed.

Alternative: apply CTM before detection or face refresh so those steps see corrected pixels. Rejected because geometry and identity refresh should remain stable when CTMs are regenerated later, and `ctm-apply` should be safely repeatable on already-created outputs.

### Region detection and crop generation are limited to page `_V.jpg` images

`detect-view-regions` runs only against page view JPEGs. It does not run against `_D##-##_V.jpg` derived outputs, and crop generation reads only the page `_V.jpg` plus its `mwg-rs:RegionList`.

Alternative: detect and crop on derived outputs as well. Rejected because derived outputs are already individual images, not page layouts that need region splitting.

### Zero detected regions is a valid successful result with an explicit signal

Some pages, especially title pages like `P01`, may legitimately contain no crop-worthy photo regions. Region detection therefore treats an empty result as success, not failure. The pipeline state for `view_regions` records that the step completed and explicitly notes that no regions were found.

This keeps three cases distinct:
- detection never ran
- detection ran and found no regions
- detection failed

### Non-empty region results are validated before XMP write

When the AI returns one or more candidate regions, the pipeline validates them before writing `mwg-rs:RegionList`:
- zero-area or negative-area boxes are rejected
- clearly degenerate tiny boxes are rejected
- heavily overlapping boxes are resolved deterministically rather than written blindly

The empty-result case skips these validations because there are no boxes to validate.

### Face refresh runs before CTM application and replaces only person-identifying regions

Reuse `_process_people_update` from `ai_index_runner.py` via a narrow render-time entrypoint rather than adding a second face-matching implementation. The refresh removes only `ImageRegion` entries whose `Iptc4xmpExt:RCtype` is `face-*` and replaces them with the fresh Cast result. Non-face regions remain untouched.

Face refresh runs on:
- page `_V.jpg` outputs
- crop JPEGs in `_Photos/`
- derived `_D##-##_V.jpg` outputs created by render

It runs before `ctm-apply` so face metadata stays tied to raw rendered pixels and does not need to be recomputed when colour correction changes later.

### Provenance is written at file-creation time, not as a late pipeline step

All DocumentID, DerivedFrom, and Pantry logic is isolated in `photoalbums/lib/xmpmm_provenance.py`. `xmp_sidecar.py` provides the low-level XML read/write primitives; `xmpmm_provenance.py` owns provenance-specific logic and is the only caller of those primitives for provenance fields.

Archive scan sidecars receive `xmpMM:DocumentID` inside `_ensure_archive_page_sidecar` before any render work begins. Rendered JPEG sidecars receive `xmpMM:DocumentID` immediately after the JPEG is written. `xmpMM:DerivedFrom` and initial `xmpMM:Pantry` entries are written as soon as the source set for that new file is known:
- page `_V.jpg`: primary archive scan in `DerivedFrom`; every contributing scan recorded in `Pantry`
- derived `_D##-##_V.jpg`: source derived media or scan recorded immediately after render
- crop sidecars: source page `_V.jpg` recorded when the crop sidecar is created

This keeps provenance attached to outputs even if a later pipeline step fails.

### XMP sidecars are updated in place under a per-page lock

The pipeline acquires a per-page lock file before doing any work for that page. There is only one active job in this project, so the lock exists to make page ownership explicit and prevent accidental overlapping work if that assumption changes.

XMP updates are read-modify-write operations against the canonical sidecar on disk. Writers may use temporary in-memory or temporary-file staging internally, but the final operation updates the real sidecar while preserving unrelated/manual fields. No pipeline step is allowed to replace the entire XMP document with only the fields it owns.

Owned-field boundaries:
- region detection owns `mwg-rs:RegionList` and its attached region metadata
- crop generation owns crop-file creation plus the crop-sidecar fields it writes
- face refresh owns face-type `ImageRegion` entries and `PersonInImage`
- CTM generation/application owns CTM-related fields and CTM pipeline state
- provenance owns `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry`

### Pipeline step completion is tracked in the `imago:Detections` JSON blob under a `pipeline` key

Each AI-backed step records its completion in `imago:Detections` on the relevant XMP sidecar. At minimum each entry stores `completed` and `model` where applicable; steps may also add step-specific metadata such as `result = "no_regions"`.

This is the sole gate for skipping a step - the pipeline reads the `pipeline` subkey, checks if the step's entry is present, and skips if so (unless `--force`). This is preferred over checking only for output data existence because it cleanly distinguishes "ran and found nothing" from "never ran".

### Each step uses the same underlying function whether run standalone or via the pipeline

`detect-view-regions`, `crop-regions`, `face-refresh`, `ctm generate`, and `ctm-apply` all use the same underlying helpers whether invoked directly or by `render-pipeline`. The pipeline command is orchestration, not a second implementation.

### Page failures are isolated and summarized at the end

If one page fails, the pipeline prints the page id, failing step, and underlying error immediately, does not write pipeline state for that failed step, releases the page lock, and continues with the next page. After the full run completes, it prints a summary of every failed page/step pair and exits non-zero if any page failed.

## Pipeline Command Order

```
photoalbums render-pipeline <album_id> [--page N] [--photos-root PATH] [--force]
```

Runs for every matching page in sequence:
1. **render** - acquire per-page lock -> `_ensure_archive_page_sidecar` (assigns DocumentID to archive scan) -> stitch (no CTM) -> write `_V.jpg` + `_D##-##_V.jpg` -> assign DocumentID and write initial DerivedFrom/Pantry immediately after each file is created
2. **detect-regions** - detect photo regions on raw page `_V.jpg`, validate non-empty results, write MWG-RS XMP, record `pipeline.view_regions` with either regions found or `result = "no_regions"`
3. **crop-regions** *(from `render-photo-region-crops` change)* - crop each detected region from raw `_V.jpg` pixels into `_Photos/`; assign DocumentID and write initial DerivedFrom/Pantry immediately after each crop sidecar is created; skip if `pipeline.crop_regions` is recorded
4. **face-refresh** - run buffalo_l + Cast against page `_V.jpg`, each crop in `_Photos/`, and render-produced `_D##-##_V.jpg` outputs; replace only person-identifying face metadata; skip per file if `pipeline.face_refresh` is recorded
5. **ctm-apply** - read stored CTMs and apply them in place to `_V.jpg` and crop JPEGs; record `pipeline.ctm_applied`; skip per file if already recorded; skip silently if no CTM is stored for that file
6. **finish page** - release the page lock and move to the next page; after all pages finish, print a failure summary if any page failed

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Face refresh on raw pixels may differ slightly from a colour-corrected run | Intended; re-running CTM should not force face refresh to run again |
| Zero-region pages could be mistaken for skipped work | `pipeline.view_regions` records an explicit no-regions result |
| Manual sidecar edits could be lost on rerun | Writers update only owned fields in place and preserve unrelated XMP |
| A page fails after provenance is written but before later steps finish | Intended; provenance should survive partial pipeline failure |
| Buffalo_l Cast store not available at render time | Face refresh logs the underlying error, records no state for that file, and the page failure appears in the end-of-run summary |
