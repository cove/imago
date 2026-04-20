## Context

Scanned photo album pages are indexed by `ai-index`, which runs OCR, extracts captions, and geocodes locations at page granularity. The render pipeline then crops individual photos from detected regions. Today, every crop on a page inherits the same merged caption blob and the same page-level location — correct only when all photos share the same caption and place.

Two structural gaps exist:
1. Multiple captions on a page are concatenated into one string with no boundaries, so there is no way to assign the right caption to the right crop without re-running AI.
2. Location is a page-level scalar; when a page spans multiple places, every crop gets the wrong or blended location.

The caption matching step (`_caption_matching.py`) already asks an LM Studio model to assign a caption per photo. It is the natural extension point for location assignment.

## Goals / Non-Goals

**Goals:**
- Number distinct captions in page `dc:description` so they are individually referenceable.
- Extend caption matching to return a location name per photo when the page has multiple `LocationShown` entries.
- Resolve matched location names to GPS via Nominatim and write to each crop's XMP.
- Support manual location override at page level (archive XMP, `gps_source: "manual"`) and at region level (`imago:LocationOverride` in page XMP).
- Protect manual overrides from being clobbered by re-runs.

**Non-Goals:**
- Tooling UI for setting manual overrides (separate spec/change).
- Changing how single-location pages work (no regression for the common case).
- Re-running existing crops automatically; migration is opt-in re-render.

## Decisions

### D1: Extend caption matching prompt rather than add a separate location-matching step
**Decision:** Extend the existing LM Studio call to return `{"caption": ..., "location": ...}` per photo when multiple locations exist.  
**Rationale:** The model already sees the whole page image in context. A second call would double latency and the model would need to re-reason about the same image. One call is cheaper and the model can correlate caption text with location in the same pass.  
**Alternative considered:** A separate post-caption location assignment step using only region crops. Rejected — individual crops often lack enough context to determine location without seeing the full page and caption layout.

### D2: Location name resolved to GPS via Nominatim at write time
**Decision:** Caption matching returns a plain-text location name (e.g., `"Luxor, Egypt"`). GPS resolution via Nominatim happens at XMP write time, same as the existing `_resolve_location_payload` path.  
**Rationale:** Keeps the AI call simple; reuses existing geocoding infrastructure and caching.

### D3: `gps_source: "manual"` exempted from stale detection
**Decision:** `_has_legacy_ai_locations_shown_gps()` skips entries where `gps_source == "manual"`. The `location_shown_ran: true` flag in archive XMP Detections prevents re-triggering the location step.  
**Rationale:** Without this, any manually-added GPS entry with a non-Nominatim source triggers `location_shown_ai_gps_stale` and causes the pipeline to overwrite the override.  
**Alternative considered:** A separate `imago:LocationPinned` top-level flag. Rejected — per-entry `gps_source` is already the pattern used to distinguish Nominatim vs. other sources; extending it to `"manual"` is the minimal change.

### D4: Per-region override stored as `imago:LocationOverride` in page XMP
**Decision:** Page XMP region entries gain an optional `imago:LocationOverride` struct (same fields as a `LocationShown` entry). Render reads this before falling back to the AI-assigned or page-level location for that crop.  
**Rationale:** Archive XMP has no concept of regions; region-level data can only live in page XMP. The render pipeline already reads region entries; adding one more field is the smallest change.  
**Re-render protection:** Render checks presence of `imago:LocationOverride` and skips writing location for that crop if set, treating it as locked.

### D5: Numbered captions built from caption matching output
**Decision:** After the caption matching step runs, the distinct caption texts assigned to regions are collected, deduplicated, and written to page `dc:description` as a numbered list. If only one unique caption exists, the current single-string format is preserved.  
**Rationale:** Building numbering from the caption matching output (post-hoc) avoids requiring a separate pre-detection page analysis step. The AI has already done the work of identifying distinct captions; we surface it in the description.

## Risks / Trade-offs

- **Caption matching model quality on location** — the model must correctly infer location from the caption text and photo context. If the caption doesn't mention the location and the photo is ambiguous, the model may return an empty or wrong location string. Mitigation: treat empty location response as "no assignment"; fall back to page-level location for that crop.
- **Nominatim rate limiting** — per-crop geocoding adds more Nominatim calls per page. Mitigation: existing geocode cache in `data/geocode_cache.json` absorbs repeated identical queries.
- **Re-render required for existing crops** — existing crops will not get updated locations/captions without a re-render. Mitigation: scoped re-render by page or album; document as expected behavior.
- **Prompt regression** — extending the caption matching prompt changes its output format. Mitigation: keep backward compatibility by detecting whether the model returned the old string format or the new object format, and handle both.

## Migration Plan

1. Deploy pipeline changes (no data migration needed for new crops).
2. For existing crops needing updated location/caption: re-render affected pages via `imago render --rerender`.
3. For manual overrides: edit archive XMP directly or use `photoalbums-map` tooling (separate spec). No automated migration.

## Open Questions

- Should the numbered caption format on the page description be opt-in (only when multiple captions detected) or always emit numbers even for single captions? Current decision: opt-in (preserve single-caption format for backward compat).
- Should `imago:LocationOverride` accept a raw address string and geocode at read time, or require pre-resolved GPS? Current decision: accept address string + optional GPS; geocode lazily at crop write time.
