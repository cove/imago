## Context

`photoalbums/stitch_oversized_pages.py` currently produces rendered view outputs in two different ways:

- Base page views copy the archive page sidecar with `_copy_base_view_sidecar`.
- Derived views run `_index_rendered_view_image` against the rendered image.

The indexer already has a Cast-backed people-refresh path in `photoalbums/lib/ai_index_runner.py` that re-runs face matching when the Cast store changes, but that path is aimed at sidecar refresh for indexing work and currently preserves inherited `PersonInImage` names. The user request is narrower and stricter: during render, any person-identifying regions on the rendered output need to be replaced by a fresh `buffalo_l` run using Cast for the names.

The XMP helpers also need care here. Face regions live in IPTC `ImageRegion`, which can also hold non-face photo regions. The current helper shape makes it easy to remove the entire `ImageRegion` bag, which would be broader than requested.

## Goals / Non-Goals

**Goals:**
- Refresh person-identifying metadata for rendered outputs as part of the render workflow.
- Run face identification against the rendered image pixels, not against inherited archive metadata.
- Use Cast-backed `buffalo_l` matches as the source of truth for rendered-output face names.
- Replace only person-identifying image-region entries while preserving unrelated render metadata.

**Non-Goals:**
- Rebuild the whole AI indexing pipeline for render outputs.
- Retroactively migrate already-rendered files outside the normal render workflow.
- Change OCR, caption, location, or non-person region behavior beyond what is required to preserve existing data.

## Decisions

### Reuse the existing people-refresh indexing path instead of adding a second face-matching implementation

Render should call a focused helper that reuses the existing Cast matcher and people-refresh logic from `ai_index_runner`, but in a render-specific mode. This keeps `buffalo_l` detection thresholds, Cast-store access, and processing signatures aligned with normal indexing.

Alternative considered: invoking a separate render-only face-matching routine from `stitch_oversized_pages.py`.
Rejected because it would duplicate Cast loading, matching logic, and processing-state updates.

### Treat the rendered image as the only source of truth for rendered-output person regions

The refresh must analyze the rendered JPEG itself and write the refreshed results back to that rendered file’s sidecar. Existing person-identifying regions and inherited `PersonInImage` names on the rendered sidecar should be replaced by the new refresh result, not unioned with copied archive names.

Alternative considered: keep the current people-update union behavior and only replace face boxes.
Rejected because it can preserve stale names that no longer correspond to any rendered-image match.

### Replace only face regions, preserve non-face image regions and other copied metadata

The XMP update should remove prior person-identifying `ImageRegion` entries such as `face-*`, then write the newly detected face regions, while preserving any non-face image regions and all unrelated XMP metadata already present on the rendered sidecar.

Alternative considered: rewrite the entire `ImageRegion` field during the refresh.
Rejected because it risks deleting photo-region metadata that is not part of this request.

### Use one render-time refresh step for every rendered JPEG output

After a render output exists and has a sidecar, the render pipeline should run the same person-region refresh for:

- single-scan page renders,
- stitched page renders,
- derived JPEG renders.

This gives one consistent rule for rendered outputs instead of keeping separate copy-vs-index behavior for person-identifying metadata.

Alternative considered: refresh only copied page sidecars and leave derived-output behavior alone.
Rejected because the request applies to the render process broadly and users should get the same rendered-output behavior regardless of render path.

## Risks / Trade-offs

- Fresh rendered-image matching may drop inherited names that were previously present but no longer meet Cast thresholds on the rendered JPEG. → This is intentional; the rendered file should reflect the fresh render-time pass, not inherited archive guesses.
- Reusing the existing people-refresh path may require a small refactor to expose it cleanly outside the normal index run loop. → Keep the refactor narrow and centered on one helper entrypoint.
- Preserving non-face image regions requires more careful XMP merge logic than today. → Add targeted tests for mixed face and non-face `ImageRegion` content before wiring the render path to it.

## Migration Plan

No standalone migration is required. The new behavior applies the next time render writes or refreshes a rendered output sidecar.

## Open Questions

- Whether the render command should refresh people metadata for skipped existing outputs when the JPEG is unchanged but the rendered sidecar still carries inherited face regions.
- Whether render should surface a clearer log label for the person-region refresh step so troubleshooting shows that the rendered sidecar was refreshed from Cast.
