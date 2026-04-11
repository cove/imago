## MODIFIED Requirements

### Requirement: caption_hint is stored in each region's XMP alongside dc:description
The system SHALL store the model's `caption_hint` value for each region in the `mwg-rs:RegionList` XMP so that downstream steps (such as crop sidecar writing) can use it as a caption source when no externally-assigned `dc:description` is present.

#### Scenario: caption_hint written to region XMP
- **WHEN** region detection completes and `write_region_list` writes the MWG-RS XMP
- **THEN** each region entry that has a non-empty `caption_hint` includes it as a retrievable field in the region's XMP block (stored as `imago:CaptionHint`)

#### Scenario: caption_hint readable from region XMP on subsequent steps
- **WHEN** `crop_page_regions` reads the `mwg-rs:RegionList` from the view sidecar
- **THEN** the `caption_hint` field is available per region so it can be used in the caption priority chain

### Requirement: A per-collection people roster maps shorthand names to full names
The system SHALL read a `[sets.<collection>.people]` table from `album_sets.toml` for the active album set. Each key is a lowercase shorthand (as it appears in hyphenated captions, e.g. `audrey`) and each value is the corresponding full name (e.g. `"Audrey Cordell"`). Entries with an empty value are ignored. This roster is passed verbatim to the vision model as context; the model uses it to expand hyphenated name sequences into full names. If no `people` table is present for the collection, the roster is empty and name expansion is best-effort.

#### Scenario: Roster loaded from album_sets.toml
- **WHEN** `run_detect_view_regions` is called for the `cordell` album set and `album_sets.toml` contains `[sets.cordell.people]` with `audrey = "Audrey Cordell"`
- **THEN** the roster `{"audrey": "Audrey Cordell", ...}` is passed to the model prompt as name context

#### Scenario: Empty roster value skipped
- **WHEN** the roster contains `karl = ""`
- **THEN** `karl` is excluded from the name context passed to the model

### Requirement: The vision model returns structured person names per region; hyphenated name shorthand is expanded by the model
The system SHALL include a `person_names` array in the JSON response schema returned by the vision model for each region. When the model is given album context (family surname) and an existing page caption, it SHALL expand any hyphenated name shorthand (e.g. `audrey-leslie-karl` with family name `Cordell` → `["Audrey Cordell", "Leslie Cordell", "Karl Cordell"]`) into individual full names in that array. No Python string parsing of hyphenated names is performed — the AI is solely responsible for name expansion.

The detection call SHALL accept two optional context parameters passed into the model prompt:
- `album_context` — the album family/collection name (e.g. `"Cordell"`) derived from `parse_album_filename`
- `page_caption` — the existing `dc:description` from the view sidecar, if present

These are provided as context only; the model uses them to inform `person_names` but they do not change region geometry output.

#### Scenario: Hyphenated name shorthand expanded into person_names
- **WHEN** the page caption contains `"audrey-leslie-karl"` and album context is `"Cordell"`
- **THEN** the model returns `person_names: ["Audrey Cordell", "Leslie Cordell", "Karl Cordell"]` for the relevant region(s)

#### Scenario: person_names empty when no names identifiable
- **WHEN** the model sees no people or cannot identify names from image or caption context
- **THEN** `person_names` is an empty array for that region

#### Scenario: person_names stored in region XMP as imago:PersonNames bag
- **WHEN** `write_region_list` writes the MWG-RS XMP and a region has non-empty `person_names`
- **THEN** those names are stored as an `imago:PersonNames` bag alongside `imago:CaptionHint` in the region's XMP block

#### Scenario: person_names used to populate PersonInImage on crop sidecar
- **WHEN** `_write_crop_sidecar` runs for a region that has `person_names: ["Audrey Cordell"]`
- **THEN** the crop sidecar's `Iptc4xmpExt:PersonInImage` bag contains `"Audrey Cordell"` (face-refresh may add further names or refine this later)
