## Context

Page P03 of EasternEuropeSpainAndMorocco_1988 illustrates both bugs cleanly. The page OCR reads "ZAGREB,YUGOSLAVIA - AUGUST 1988 KARNTEN,AUSTRIA". The AI correctly extracts two `LocationShown` entries (Zagreb and Karnten with correct GPS). The caption matching step correctly assigns `mwg-rs:Name = "KARNTEN, AUSTRIA"` to region D05. But the crop writer:

1. Ignores the caption-to-location relationship and stamps Zagreb (the page primary GPS) on every crop.
2. Passes "KARNTEN, AUSTRIA" through to `PersonInImage` because it was in the AI's people/name extraction output.

Both are post-processing failures at crop write time. All the correct data is already present on the page — `locations_shown`, region captions — it just isn't being used to guard these two fields.

## Goals / Non-Goals

**Goals:**
- Use a region's own caption to select the matching `LocationShown` entry when one exists.
- Strip location strings from `PersonInImage` before writing, on both page and crop sidecars.

**Non-Goals:**
- Full per-photo location assignment from AI (that's `caption-location-per-photo`).
- Geocoding new locations not already in `LocationShown`.
- Fixing existing already-written crop XMP files automatically (re-render is the path for that).

## Decisions

### D1: Caption-to-LocationShown match is a simple substring check
**Decision:** Normalize both the region caption and each `LocationShown` name to uppercase, strip punctuation, and check if either contains the other. If a match is found, use that entry's GPS/city/country for the crop.  
**Rationale:** "KARNTEN, AUSTRIA" and "Karnten, Austria" are the same string modulo case/punctuation. A full fuzzy match adds complexity with no benefit for this use case. If no match, fall back to page primary location unchanged.  
**Edge case:** If multiple `LocationShown` entries match the caption (unlikely but possible), prefer the one with GPS resolved.

### D2: PersonInImage filter runs at the point names are assembled, not in the AI step
**Decision:** Filter in `ai_photo_crops.py` (and the equivalent page sidecar write path) just before passing `person_names` to the XMP writer, using the already-available `locations_shown` list plus page city/state/country fields.  
**Rationale:** The AI extraction step shouldn't be modified — it may legitimately surface location strings for other purposes. The filter is a write-time guard, not an AI correction. This also means the filter applies consistently regardless of which AI engine produced the names.  
**Filter set:** Collect all of: `LocationShown` names, page `photoshop:City`, `photoshop:State`, `photoshop:Country`, `Iptc4xmpExt:LocationCreated`. Normalize to uppercase, strip punctuation. Remove any `PersonInImage` candidate whose normalized form is contained in or contains any entry in the filter set.

### D3: Apply filter to page sidecars too, not only crops
**Decision:** The same contamination can happen on page-level `PersonInImage`. Apply the filter when writing page sidecars.  
**Rationale:** Symmetric fix; same root cause, same data available.

## Risks / Trade-offs

- **False positives:** A person genuinely named "Austria" would be filtered. Acceptable — family album names are not country names; the risk is negligible.
- **Caption matching miss:** If the region caption is only partial (e.g., "KARNTEN" without "AUSTRIA") and the `LocationShown` name is "Karnten, Austria", the substring check still passes because "KARNTEN" is contained in "KARNTEN, AUSTRIA". Direction: caption contained-in LocationShown OR LocationShown contained-in caption.
- **No retroactive fix:** Existing wrong crops need a re-render. Document this expectation.

## Migration Plan

No data migration. Re-render affected pages to apply corrected location and PersonInImage values to existing crops. The fix is purely in the write path.
