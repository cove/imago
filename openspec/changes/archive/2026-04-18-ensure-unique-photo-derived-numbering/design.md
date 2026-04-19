## Context

Crop outputs are currently named from the region index alone, so page `P40` always produces `_D01-00_V.jpg`, `_D02-00_V.jpg`, and so on. That works only inside `_Photos/`, but the album naming convention treats `D##-##` as one shared derived-image namespace per page across `_Archive/`, `_Pages/`, and `_Photos/`.

The live album tree already shows the failure mode on `Family_1907-1946_B01_P40`: `_Archive/` contains `D01` through `D03`, while `_Photos/` also contains `D01` through `D05`. The repair path therefore has to do more than rename direct collisions. It also needs to canonicalize crop numbering into one contiguous range after the archive-derived maximum so future crop reruns resolve to the same paths.

## Goals / Non-Goals

**Goals:**
- Make new crop outputs allocate `D##-00` numbers after the highest derived number already present for the same page in `_Archive/`
- Keep crop numbering deterministic for a page so reruns target the same filenames
- Add a repair command and library function that rename existing crop JPEG/XMP pairs into the canonical non-colliding sequence
- Preserve crop sidecar contents and page provenance while renaming existing files

**Non-Goals:**
- Renumber archive-derived files or page-derived `_D##-##_V` outputs in `_Pages/`
- Recompute region detection, crop geometry, captions, or restoration as part of repair
- Introduce a global database of assigned crop numbers; numbering stays reconstructable from on-disk filenames

## Decisions

### 1. Use `_Archive/` as the sole numbering baseline

For a given page, the allocator will scan matching archive-derived filenames, parse their `D##` values, and compute `archive_max`. Crop region 1 then maps to `D{archive_max + 1}-00`, region 2 to `D{archive_max + 2}-00`, and so on.

This matches the user requirement that all files could be moved into one directory without name collisions, because `_Archive/` is the authoritative source of already-assigned derived numbers for the page.

Alternative considered:
- Base crop numbering on the highest number found in `_Photos/` as well. Rejected because crop names would drift over time and would depend on previously broken outputs instead of the canonical archive lineage.

### 2. Canonicalize repairs to a contiguous crop sequence after `archive_max`

The repair logic will group crop outputs by page, sort the existing `_Photos/` crop pairs by current derived number, and assign the canonical target range `archive_max + 1 .. archive_max + crop_count`. This fixes both direct collisions and non-canonical gaps in one pass.

Example: if `_Archive/` ends at `D03` and `_Photos/` currently contains `D01` through `D05`, the repair result becomes `D04` through `D08`.

Alternative considered:
- Rename only colliding crop numbers and leave non-colliding higher values untouched. Rejected because the page would still have non-sequential crop numbering and future reruns would not have a single deterministic filename mapping.

### 3. Rename crop JPEG/XMP pairs through temporary paths

Repair cannot rename directly to final targets because a canonical destination may already exist on disk during the same page migration. The implementation should first move every affected crop JPEG/XMP pair to a unique temporary name in the same directory, then move them to final canonical names.

This keeps the rename operation atomic at the pair level and avoids overwriting surviving files such as the existing `D04-00`/`D05-00` crops on `P40`.

Alternative considered:
- Delete and recreate crops instead of renaming. Rejected because repair should preserve existing JPEG pixels, XMP metadata, and pipeline state exactly as they are.

### 4. Expose repair as a targeted command, backed by a library function

The repo already has command patterns like `repair-crop-source` for XMP-only migrations. This change should follow the same shape: a reusable library function that can repair a whole root, an album, or a single page, and a CLI entry point that reports what changed.

That keeps the migration logic scriptable and testable without embedding one-off repair code in the crop pipeline itself.

## Risks / Trade-offs

- [Manual workflows may already refer to old crop filenames] -> Mitigation: keep repair scoped and explicit, document that crop filenames become canonical after repair, and preserve crop contents and sidecar metadata exactly
- [Archive-derived numbering above 99 may exceed current filename expectations] -> Mitigation: reuse the existing derived-name parser behavior and add tests around larger derived numbers if needed rather than inventing a new format
- [Broken or partial crop pairs may make repair ambiguous] -> Mitigation: treat JPEG/XMP pairs as the unit of repair and fail loudly when a selected crop is missing its companion sidecar instead of silently discarding state

## Migration Plan

1. Add a helper that computes the per-page archive-derived offset and update crop output path generation to use it.
2. Add tests covering new crop allocation for pages with and without existing archive-derived images.
3. Add a crop-number repair library and CLI command that canonicalize existing `_Photos/` crops by page using temporary rename staging.
4. Run the repair command across the photo album tree or a targeted album/page, then verify no page has overlapping `D##` values between `_Archive/` and `_Photos/`.

Rollback is straightforward before repair runs because the runtime change only affects future filenames. After repair runs, rollback means renaming files back from a recorded mapping, so the command output should include old-to-new paths.

## Open Questions

- None at proposal time.
