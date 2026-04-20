## 1. Location string filter utility

- [ ] 1.1 Add `_build_location_filter_set(locations_shown, city, state, country, location_created)` in `ai_photo_crops.py` (or a shared util): returns a set of normalized (uppercase, punctuation-stripped) location strings built from all provided inputs
- [ ] 1.2 Add `_filter_location_names_from_people(person_names, location_filter_set)`: returns person_names with any entry whose normalized form is contained-in or contains a filter set entry removed
- [ ] 1.3 Add unit tests for both functions: empty inputs, full match, partial containment match, no match, case/punctuation variants

## 2. Apply person name filter at crop write time

- [ ] 2.1 In `ai_photo_crops.py`, before passing `person_names` to the crop XMP writer, call `_build_location_filter_set` from the page `locations_shown` and primary location fields, then call `_filter_location_names_from_people`
- [ ] 2.2 Add tests: "KARNTEN, AUSTRIA" removed when LocationShown contains "Karnten, Austria"; genuine person name preserved; empty locations_shown skips filtering

## 3. Apply person name filter at page write time

- [ ] 3.1 Identify the page sidecar write path(s) in `ai_index_runner.py` or `xmp_sidecar.py` where `PersonInImage` is assembled
- [ ] 3.2 Apply the same filter before writing `PersonInImage` on page sidecars, using the page's own location data
- [ ] 3.3 Add tests: page sidecar with location-matching person candidate is filtered; page with no location data is unchanged

## 4. Caption-to-LocationShown matching for crop location

- [ ] 4.1 Add `_match_caption_to_location_shown(caption, locations_shown)` in `ai_photo_crops.py`: returns the best-matching `LocationShown` entry dict (preferring entries with GPS) or None
- [ ] 4.2 In `resolve_region_caption` / crop location resolution, call this match before falling back to page primary location; if a match is returned use its GPS, city, state, country
- [ ] 4.3 Add tests: caption "KARNTEN, AUSTRIA" matches entry "Karnten, Austria" with GPS → crop gets Karnten GPS; no match → page primary used; empty caption → page primary used; multiple matches, one with GPS → GPS entry wins

## 5. Validation

- [ ] 5.1 Run `uv run python -m py_compile` on changed modules
- [ ] 5.2 Run `just test`
- [ ] 5.3 Run `just dupes`
- [ ] 5.4 Run `just deadcode`
- [ ] 5.5 Re-render page P03 of EasternEuropeSpainAndMorocco_1988 and verify crop D05 gets Karnten GPS and "KARNTEN, AUSTRIA" is absent from PersonInImage
