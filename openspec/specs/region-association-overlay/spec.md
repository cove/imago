## Purpose
Define the numbered region-association overlay artifact used to hand off accepted Docling region identity to Gemma semantic association.

## Requirements

### Requirement: A prompt-safe numbered region overlay is generated from accepted regions
The system SHALL generate a region-association overlay image from the accepted Docling regions. The overlay SHALL render the original page image plus only:
- precise region outlines derived from the accepted regions' normalized geometry
- visible region numbers

The overlay SHALL NOT contain candidate captions, person names, validation notes, or other debug annotation text.

#### Scenario: Accepted regions produce a numbered overlay
- **WHEN** region detection completes with three accepted regions
- **THEN** the system generates an overlay image showing those three accepted region outlines and visible numbers over the page image

#### Scenario: Prompt overlay excludes debug annotation text
- **WHEN** accepted regions already have caption hints or other debug labels available in memory
- **THEN** the region-association overlay omits those labels
- **AND** renders only outlines and visible numbers

### Requirement: Visible overlay numbers are the authoritative region identifiers for semantic association
The system SHALL treat the visible numbers rendered on the region-association overlay as the authoritative identifiers for Gemma semantic association. The model response SHALL key region associations to those visible numbers directly. The system SHALL NOT require the model to invent a separate reading-order numbering scheme.

#### Scenario: Model response keys map directly to overlay numbers
- **WHEN** the overlay shows visible region numbers 1, 2, and 3
- **THEN** the Gemma response keys refer directly to those visible region identifiers
- **AND** the merge path maps each response entry back to the corresponding accepted region without scanline sorting

#### Scenario: Unknown response key is ignored
- **WHEN** Gemma returns an association for a region number not present on the overlay
- **THEN** that association is ignored
- **AND** no other accepted region is remapped to compensate

### Requirement: Semantic association uses the numbered overlay as prompt input
The system SHALL feed the region-association overlay image into the Gemma semantic-association step so the model can associate captions and location/date semantics to the exact accepted regions Docling identified.

#### Scenario: Overlay image drives caption association
- **WHEN** Gemma performs semantic association for a page with accepted regions
- **THEN** the prompt input includes the numbered region-association overlay
- **AND** Gemma uses the visible numbered regions to decide which caption belongs to which accepted region
