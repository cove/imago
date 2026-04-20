## Why

When cropping individual photos from a scanned album page, every crop currently receives the same merged caption blob and the same page-level location, even when the page contains multiple distinct captions and photos from different places. The goal is to associate each cropped photo with the right caption, location, and date.

## What Changes

- Page `dc:description` emits captions as a numbered list (`1. Caption A. 2. Caption B.`) when the page contains multiple distinct captions, so individual captions are identifiable.
- The LM Studio caption matching step is extended: when a page has 2+ `LocationShown` entries, the prompt also returns a location name per photo alongside the caption; when only 1 entry exists, page-level location is inherited as before.
- Per-photo location names returned by caption matching are resolved to GPS coordinates via Nominatim and written to each crop's XMP.
- Manual location override is supported at two granularities:
  - **Page-level**: archive XMP `LocationShown` entries with `gps_source: "manual"` are treated as authoritative and exempt from stale detection and re-geocoding.
  - **Per-region**: `imago:LocationOverride` field on a page XMP region entry; render reads this and does not overwrite it, allowing per-photo manual location pinning.
- `_has_legacy_ai_locations_shown_gps()` is updated to exempt `gps_source: "manual"` from triggering `location_shown_ai_gps_stale`.

## Capabilities

### New Capabilities
- `caption-location-per-photo`: Per-crop caption selection and location assignment, including numbered page captions, extended caption matching with per-photo location output, Nominatim resolution of matched locations, and `imago:LocationOverride` support in region XMP.

### Modified Capabilities
- `gemma4-caption-matching`: Caption matching prompt and response schema extended to optionally return per-photo location alongside caption text.
- `view-xmp-regions`: Region XMP contract extended with `imago:LocationOverride` field; render respects this field and does not overwrite it.
- `docling-region-detection`: No requirement change; location assignment now flows through the caption matching step rather than being inherited uniformly.

## Impact

- `photoalbums/lib/_caption_matching.py`: extend prompt and response parsing to handle `{"caption": ..., "location": ...}` per photo.
- `photoalbums/lib/ai_location.py`: exempt `gps_source: "manual"` in `_has_legacy_ai_locations_shown_gps()`.
- `photoalbums/lib/ai_index_runner.py`: apply per-photo location from caption matching result; pass to crop XMP writer.
- `photoalbums/lib/ai_photo_crops.py`: accept and write per-crop location; read `imago:LocationOverride` from region entry before falling back to page location.
- `photoalbums/lib/xmp_sidecar.py`: write `imago:LocationOverride` field; read it back during crop sidecar creation.
- Page `dc:description` formatting logic (wherever page description is assembled).
