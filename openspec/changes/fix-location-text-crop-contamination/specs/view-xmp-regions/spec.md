## ADDED Requirements

### Requirement: Crop location is resolved from matching LocationShown entry when region caption names a location
Before falling back to page primary GPS, the system SHALL check whether the region caption (`mwg-rs:Name`) matches any `LocationShown` entry name on the page. Matching is case-insensitive with punctuation stripped; a match occurs when the normalized caption is contained in the normalized `LocationShown` name or vice versa. When a match is found, the system SHALL use that entry's GPS coordinates, city, state, and country as the crop location. When no match is found, the system SHALL use page primary location as before.

When multiple `LocationShown` entries match the caption, the system SHALL prefer the entry with GPS coordinates resolved over one without.

#### Scenario: Region caption matches a LocationShown entry name
- **WHEN** a region's `mwg-rs:Name` is "KARNTEN, AUSTRIA" and the page has a `LocationShown` entry named "Karnten, Austria" with GPS 46.75N 13.83E
- **THEN** the crop sidecar uses that entry's GPS and location fields instead of the page primary GPS

#### Scenario: Region caption matches no LocationShown entry
- **WHEN** a region's `mwg-rs:Name` is "Uncle Bob at the lake" and no `LocationShown` entry contains that text
- **THEN** the crop sidecar inherits the page primary location unchanged

#### Scenario: Region has no caption
- **WHEN** a region's `mwg-rs:Name` is empty
- **THEN** the crop sidecar inherits the page primary location unchanged

#### Scenario: Multiple LocationShown entries match, one has GPS
- **WHEN** two `LocationShown` entries both match the caption but only one has GPS coordinates
- **THEN** the crop uses the entry with GPS coordinates
