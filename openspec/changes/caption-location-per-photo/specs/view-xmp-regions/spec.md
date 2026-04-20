## ADDED Requirements

### Requirement: Region entries support imago:LocationOverride for manual location pinning
The system SHALL support an optional `imago:LocationOverride` field on each region entry within the page XMP `mwg-rs:RegionList`. This field MAY contain a struct with location fields (`city`, `state`, `country`, `sublocation`, `gps_latitude`, `gps_longitude`, `address`). When present and non-empty on a region, the crop step SHALL use this value as the crop's location and SHALL NOT fall back to AI-assigned or page-level location for that crop. The render pipeline SHALL preserve `imago:LocationOverride` on re-render and MUST NOT overwrite or clear it.

#### Scenario: Region has imago:LocationOverride, crop uses it
- **WHEN** a region entry in the page XMP has a non-empty `imago:LocationOverride` struct
- **THEN** the crop sidecar for that region uses the override location for all location fields (city, state, country, GPS)

#### Scenario: Re-render preserves imago:LocationOverride
- **WHEN** the page is re-rendered and a region has `imago:LocationOverride` set
- **THEN** the region entry retains `imago:LocationOverride` unchanged after re-render

#### Scenario: Region has no imago:LocationOverride
- **WHEN** a region entry has no `imago:LocationOverride` field
- **THEN** the crop step uses AI-assigned or page-level location (unchanged behavior)
