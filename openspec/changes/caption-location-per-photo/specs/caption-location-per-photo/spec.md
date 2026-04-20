## ADDED Requirements

### Requirement: Page dc:description emits numbered captions when multiple distinct captions exist
When the caption matching step produces two or more distinct caption strings across the regions on a page, the system SHALL write page `dc:description` as a numbered list (`1. Caption A. 2. Caption B.`). When only one unique caption string exists, the system SHALL preserve the current single-string format. The numbering follows reading order (left-to-right, top-to-bottom).

#### Scenario: Page has two distinct captions
- **WHEN** caption matching returns two or more unique non-empty caption strings across regions
- **THEN** page `dc:description` is written as `1. <first caption>. 2. <second caption>.` in reading order

#### Scenario: Page has only one distinct caption
- **WHEN** all regions share the same caption string or only one unique caption exists
- **THEN** page `dc:description` is written as the single caption string without numbering

#### Scenario: No captions assigned
- **WHEN** caption matching returns empty strings for all regions
- **THEN** page `dc:description` falls back to the existing OCR/scene text summary format

### Requirement: Per-photo location is assigned when a page has multiple LocationShown entries
When a page's archive XMP contains two or more `Iptc4xmpExt:LocationShown` entries, the system SHALL extend the caption matching LM Studio call to also return a location name per photo. The returned location name SHALL be resolved to GPS coordinates via Nominatim (using the existing geocode cache) and written to the corresponding crop XMP. When a page has zero or one `LocationShown` entries, the system SHALL use the existing page-level location inheritance behavior unchanged.

#### Scenario: Page has multiple locations, model assigns locations
- **WHEN** the page archive XMP has two or more `LocationShown` entries and the caption matching model returns a non-empty location string for a photo
- **THEN** that location string is resolved via Nominatim and written to the crop XMP as the crop's primary location, overriding page-level location inheritance for that crop

#### Scenario: Page has multiple locations, model returns empty location for a photo
- **WHEN** the page archive XMP has two or more `LocationShown` entries but the model returns an empty location string for a specific photo
- **THEN** that crop falls back to page-level location inheritance for that photo

#### Scenario: Page has one or zero LocationShown entries
- **WHEN** the page archive XMP has zero or one `LocationShown` entries
- **THEN** the caption matching prompt is sent without location assignment instructions and all crops inherit page-level location as before

### Requirement: Manual page-level location override via gps_source "manual" is exempt from stale detection
The system SHALL treat `Iptc4xmpExt:LocationShown` entries in archive XMP where `gps_source` equals `"manual"` as authoritative. These entries SHALL NOT trigger the `location_shown_ai_gps_stale` reprocess condition. When `location_shown_ran: true` is set in `imago:Detections` and all GPS entries are either from Nominatim or marked `"manual"`, the system SHALL skip the location_shown step on re-index.

#### Scenario: Manual LocationShown entry present with gps_source "manual"
- **WHEN** an archive XMP `LocationShown` entry has `gps_source: "manual"` and `location_shown_ran: true` is set in Detections
- **THEN** `_has_legacy_ai_locations_shown_gps()` returns False for that page and `location_shown_ai_gps_stale` is NOT added to reprocess reasons

#### Scenario: Mixed manual and Nominatim entries
- **WHEN** a page has both `gps_source: "nominatim"` and `gps_source: "manual"` entries and `location_shown_ran: true`
- **THEN** neither entry triggers stale detection and the location step is skipped

#### Scenario: Legacy entry with GPS but no gps_source still triggers stale detection
- **WHEN** a `LocationShown` entry has GPS coordinates but no `gps_source` field
- **THEN** stale detection behavior is unchanged (legacy entries still trigger re-geocoding)

### Requirement: Per-region manual location override via imago:LocationOverride
The system SHALL support an optional `imago:LocationOverride` field on each region entry in the page XMP `mwg-rs:RegionList`. When present and non-empty, the crop step SHALL use `imago:LocationOverride` as the location for that crop instead of the AI-assigned or page-level location. The render pipeline SHALL NOT overwrite `imago:LocationOverride` on re-render; its presence is treated as a locked manual override.

`imago:LocationOverride` SHALL accept a struct with the same fields as a `LocationShown` entry, plus an optional `address` string field (raw address for lazy geocoding at write time). If GPS is not pre-populated, the crop writer SHALL attempt to geocode `address` via Nominatim and write the result to the crop XMP.

#### Scenario: Region has imago:LocationOverride set
- **WHEN** a region entry in page XMP has a non-empty `imago:LocationOverride`
- **THEN** the crop for that region uses the override location; AI-assigned and page-level locations are ignored for that crop

#### Scenario: Re-render does not clear imago:LocationOverride
- **WHEN** a page is re-rendered and a region entry has `imago:LocationOverride` set
- **THEN** the render pipeline preserves `imago:LocationOverride` on the region entry and does not overwrite it

#### Scenario: imago:LocationOverride with address but no GPS
- **WHEN** `imago:LocationOverride` contains an `address` string but no GPS coordinates
- **THEN** the crop writer geocodes the address via Nominatim, writes GPS to the crop XMP, and caches the result

#### Scenario: Region has no imago:LocationOverride
- **WHEN** a region entry has no `imago:LocationOverride` field
- **THEN** the crop uses AI-assigned location (if available) or page-level location as before; no behavior change
