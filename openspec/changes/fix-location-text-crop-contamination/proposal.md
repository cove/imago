## Why

When a page has multiple locations (e.g., "ZAGREB, YUGOSLAVIA" and "KARNTEN, AUSTRIA"), the crop writer assigns the page primary location to every crop regardless of which location the region caption names. Separately, location text from OCR (e.g., "KARNTEN, AUSTRIA") gets passed into `PersonInImage` because the AI mistakes geographic proper nouns for person names. Both failures stem from the same gap: the crop writer never cross-checks person name candidates or per-crop location against the page's known `LocationShown` entries.

## What Changes

- When writing a crop, if the region's caption (`mwg-rs:Name`) matches or is contained within a `LocationShown` name from the page, use that entry's GPS and location fields for the crop instead of the page primary location.
- Before writing `PersonInImage` on a crop, filter out any candidate name that matches (case-insensitively) a `LocationShown` name, the page primary city, country, or state. Geographic strings are not person names.
- Apply the same `PersonInImage` filter when writing page-level sidecars, not just crop sidecars.

## Capabilities

### New Capabilities
- `location-text-person-filter`: At XMP write time, remove entries from `PersonInImage` that match any known location string (from `LocationShown` names, city, state, or country fields on the same sidecar).

### Modified Capabilities
- `view-xmp-regions`: Region caption-to-location matching: when writing a crop, check the region caption against `LocationShown` names before falling back to page primary location.

## Impact

- `photoalbums/lib/ai_photo_crops.py`: add caption-to-LocationShown matching before page-level location fallback; add person name filter before `PersonInImage` write.
- `photoalbums/lib/xmp_sidecar.py`: expose or apply location-string filter when assembling `PersonInImage` bag.
- Tests for both the location match and the person filter.
