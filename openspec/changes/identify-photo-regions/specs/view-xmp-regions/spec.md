## ADDED Requirements

### Requirement: Write detected regions as MWG-RS RegionList XMP metadata
The system SHALL write detected photo regions to the view image's XMP sidecar (or embedded XMP) using the MWG-RS `RegionList` schema (`http://www.metadataworkinggroup.com/schemas/regions/`).

#### Scenario: Writing regions for a view with 3 detected photos
- **WHEN** 3 regions are detected in a view image
- **THEN** the XMP contains `mwg-rs:RegionInfo` with `mwg-rs:AppliedToDimensions` (full image width/height) and a `mwg-rs:RegionList/rdf:Bag` with 3 `rdf:li` entries, each having `mwg-rs:Type = "Photo"`, `mwg-rs:Name`, and `stArea:*` normalised coordinates (centre-point, 0–1 range)

#### Scenario: XMP sidecar already contains region data
- **WHEN** the XMP sidecar already has a `mwg-rs:RegionList` for this image
- **THEN** the existing region list is replaced with the new detection results, and an `xmp:ModifyDate` is updated

### Requirement: Region coordinates are converted from pixel space to MWG-RS normalised space
The system SHALL convert pixel `x/y/width/height` (top-left origin) from detection output into MWG-RS normalised coordinates (centre-point `stArea:x/y`, fractional `stArea:w/h` relative to image dimensions).

#### Scenario: Conversion correctness
- **WHEN** a region is at pixel x=100, y=200, width=400, height=300 in a 1000×1000 image
- **THEN** `stArea:x = 0.3` (centre), `stArea:y = 0.35`, `stArea:w = 0.4`, `stArea:h = 0.3`

### Requirement: Caption association per region
The system SHALL associate a caption string with each region by spatial nearest-centre matching; when ambiguous (closest caption equidistant within 10% of image width, or no positional data available), the page caption SHALL be applied to all regions with `captionAmbiguous = true` in the sidecar JSON.

#### Scenario: Unambiguous caption assignment
- **WHEN** a caption text is spatially closest to region 2 and not equidistant to any other region
- **THEN** only region 2 receives the caption in its `dc:description` XMP field

#### Scenario: Ambiguous captions broadcast
- **WHEN** no caption positional data is available, or the caption is equidistant to two regions
- **THEN** all regions receive the page caption, and `captionAmbiguous = true` is recorded in the sidecar JSON
