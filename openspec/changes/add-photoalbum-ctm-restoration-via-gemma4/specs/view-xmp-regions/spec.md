## MODIFIED Requirements

### Requirement: View-image XMP metadata can store photo regions
Imago MUST support storing detected photo regions for stitched view images in XMP sidecars and MUST preserve clean separation from CTM restoration metadata stored in the `_Archive/` XMP.

#### Scenario: Preserve region metadata while CTM lives in `_Archive/` XMP
- **WHEN** a stitched view image has photo-region metadata and a corresponding stitched image has CTM restoration metadata
- **THEN** Imago preserves the existing region metadata structure for the view-image XMP
- **AND** stores CTM metadata only in the `_Archive/` XMP
- **AND** does not require archive-manifest ingredient or homography fields for this CTM workflow
