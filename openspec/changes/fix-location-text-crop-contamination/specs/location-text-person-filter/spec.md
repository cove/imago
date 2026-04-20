## ADDED Requirements

### Requirement: PersonInImage entries that match known location strings are filtered before writing
The system SHALL remove any candidate person name from `Iptc4xmpExt:PersonInImage` whose normalized form (uppercase, punctuation stripped) is contained in or contains the normalized form of any known location string on the same sidecar. The location filter set SHALL be built from: all `LocationShown` name values, `photoshop:City`, `photoshop:State`, `photoshop:Country`, and `Iptc4xmpExt:LocationCreated`. This filter SHALL apply when writing both crop sidecars and page sidecars.

#### Scenario: Location text present in person name candidates
- **WHEN** the person name candidate list includes "KARNTEN, AUSTRIA" and the sidecar has a `LocationShown` entry named "Karnten, Austria"
- **THEN** "KARNTEN, AUSTRIA" is removed from `PersonInImage` before writing

#### Scenario: City name present in person name candidates
- **WHEN** a candidate name matches the sidecar's `photoshop:City` value (case-insensitive)
- **THEN** that candidate is removed from `PersonInImage` before writing

#### Scenario: Genuine person name not in location set is preserved
- **WHEN** a candidate person name does not match any location string in the filter set
- **THEN** it is written to `PersonInImage` unchanged

#### Scenario: Filter set is empty
- **WHEN** the sidecar has no `LocationShown` entries and no city/state/country fields
- **THEN** all person name candidates are written unchanged (no filtering occurs)

#### Scenario: Page sidecar receives the same filter
- **WHEN** a page sidecar is written with person name candidates that include a location string matching the page's own `LocationShown` or primary location fields
- **THEN** the location string is removed from `PersonInImage` on the page sidecar as well
