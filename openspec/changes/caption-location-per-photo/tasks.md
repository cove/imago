## 1. Pipeline: gps_source "manual" exemption

- [x] 1.1 Update `_has_legacy_ai_locations_shown_gps()` in `ai_location.py` to return False for entries where `gps_source == "manual"`
- [x] 1.2 Add tests: manual entry does not trigger `location_shown_ai_gps_stale`; mixed manual+nominatim entries do not trigger stale; legacy entry with no gps_source still triggers stale

## 2. Caption matching: multi-location prompt and response

- [x] 2.1 Update `call_lmstudio_caption_matching()` in `_caption_matching.py` to accept a `locations_shown` parameter; when 2+ entries, build the extended object-value prompt with known locations list
- [x] 2.2 Update `assign_captions_from_lmstudio()` to handle both string-value format (single-location) and object-value format (multi-location); return location name alongside caption per region
- [x] 2.3 Update `_apply_lmstudio_captions()` in `ai_photo_crops.py` to pass `locations_shown` from the page archive XMP into the caption matching call
- [x] 2.4 Add tests: single-location page uses string-value prompt and returns no location; multi-location page uses object-value prompt; empty location string falls back gracefully; malformed JSON handled

## 3. Per-photo location resolution and crop XMP write

- [x] 3.1 After caption matching, resolve non-empty location name strings via `_resolve_location_payload()` (Nominatim + geocode cache) for each region that received a location assignment
- [x] 3.2 Update crop XMP writer in `ai_photo_crops.py` to accept per-region location payload and use it in preference to page-level location when present
- [x] 3.3 Add tests: crop with assigned location receives resolved GPS; crop with empty location inherits page-level location; Nominatim result cached

## 4. imago:LocationOverride in page XMP regions

- [x] 4.1 Add `imago:LocationOverride` read/write support to `xmp_sidecar.py`: serialize struct fields to region entry, deserialize on read
- [x] 4.2 Update region read path in `ai_photo_crops.py` to check `imago:LocationOverride` first; if present and non-empty, use it as the crop location and skip AI-assigned and page-level location
- [x] 4.3 Update render pipeline to preserve `imago:LocationOverride` on re-render (do not clear or overwrite region entries that have it set)
- [x] 4.4 If `imago:LocationOverride` has `address` but no GPS, geocode via Nominatim at crop write time and write resolved GPS to crop XMP
- [x] 4.5 Add tests: override present → crop uses override location; re-render preserves override; address-only override geocoded at write time; no override → unchanged behavior

## 5. Numbered captions in page dc:description

- [x] 5.1 After caption matching, collect distinct non-empty caption strings in reading order; if 2+ unique strings exist, format page `dc:description` as numbered list (`1. Caption A. 2. Caption B.`)
- [x] 5.2 If only one unique caption string, preserve current single-string format; if no captions, fall back to OCR/scene text summary
- [x] 5.3 Add tests: two distinct captions → numbered list; single caption → no numbering; no captions → OCR fallback

## 6. Validation

- [x] 6.1 Run `uv run python -m py_compile` on all changed modules
- [x] 6.2 Run `just test`
- [x] 6.3 Run `just dupes`
- [x] 6.4 Run `just deadcode`
- [x] 6.5 Run `just complexity`
- [ ] 6.6 Re-render a mixed-location page from the live album root and verify crops receive distinct locations and correct captions
