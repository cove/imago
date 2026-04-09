## MODIFIED Requirements

### Requirement: View-image XMP metadata can store photo regions
Imago MUST support storing detected photo regions for stitched view images in XMP sidecars and MUST preserve coexistence with other supported metadata domains written to the same XMP documents.

#### Scenario: Store regions alongside CTM metadata
- **WHEN** a stitched view image has photo-region metadata and CTM restoration metadata
- **THEN** Imago preserves the existing region metadata structure
- **AND** writes or updates Camera Raw and archive-manifest metadata without removing valid region data
- **AND** allows both metadata domains to be read back from the same XMP document
