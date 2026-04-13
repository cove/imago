## MODIFIED Requirements

### Requirement: View-image XMP metadata can store photo regions
Imago MUST support storing detected photo regions for stitched view images in XMP sidecars and MUST preserve clean separation from CTM restoration metadata stored in the `_Archive/` XMP. Only region sets that pass current validation rules may be written or retained as the accepted view-image region list, and previously stored invalid region lists MUST be replaced through reprocessing rather than preserved as trusted ground truth.

#### Scenario: Preserve region metadata while CTM lives in `_Archive/` XMP
- **WHEN** a stitched view image has photo-region metadata and a corresponding stitched image has CTM restoration metadata
- **THEN** Imago preserves the existing region metadata structure for the view-image XMP
- **AND** stores CTM metadata only in the `_Archive/` XMP
- **AND** does not require archive-manifest ingredient or homography fields for this CTM workflow

#### Scenario: Invalid stored region list is replaced on reprocessing
- **WHEN** a view-image XMP already contains a region list that fails current validation rules
- **THEN** Imago does not keep that invalid region list as the accepted page state
- **AND** replaces it only with a newly accepted validated region list after reprocessing
