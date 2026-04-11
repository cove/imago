## ADDED Requirements

### Requirement: Each detected region is cropped to its own JPEG in the _Photos directory
The system SHALL read the `mwg-rs:RegionList` from a page view JPEG's XMP sidecar, convert each normalised centre-point region to a pixel rectangle, crop the corresponding area from the raw (pre-face-refresh, pre-CTM) page `_V.jpg`, and write the result as `_D{index:02d}-00_V.jpg` in the album's `_Photos/` sibling directory. CTM colour correction is applied to each crop later by the subsequent `ctm-apply` pipeline step.

#### Scenario: Crops written for a page with multiple regions
- **WHEN** `crop-regions` runs for a page whose view sidecar contains a `mwg-rs:RegionList` with 3 regions
- **THEN** three files are written: `_D01-00_V.jpg`, `_D02-00_V.jpg`, `_D03-00_V.jpg` in `<Album>_Photos/`

#### Scenario: No regions - step skips silently
- **WHEN** the page view sidecar has no `mwg-rs:RegionList` or the list is empty
- **THEN** no files are written, no error is raised, and the pipeline continues

#### Scenario: Region bounds clamped to image edge
- **WHEN** a region's normalised coordinates extend beyond [0, 1]
- **THEN** the crop rectangle is clamped to the image bounds before cropping, and a warning is printed if any dimension was clamped by more than 5% of the image size

### Requirement: crop-regions operates only on page view JPEGs
The system SHALL crop only from page `_V.jpg` images. It SHALL NOT attempt to crop render-produced `_D##-##_V.jpg` derived outputs.

#### Scenario: Derived render output is ignored by crop-regions
- **WHEN** `_D01-02_V.jpg` exists for a page
- **THEN** `crop-regions` does not treat that derived JPEG as a crop source image

### Requirement: _Photos directory mirrors _Archive and _View naming
The system SHALL derive the `_Photos/` directory path by replacing the `_Archive` suffix of the archive directory name with `_Photos`, creating it if absent.

#### Scenario: _Photos directory created on first crop
- **WHEN** `crop-regions` runs for an album with no existing `_Photos/` directory
- **THEN** the directory is created before the first crop JPEG is written

#### Scenario: _Photos path derived from archive path
- **WHEN** the archive directory is `Egypt_1975_Archive`
- **THEN** crops are written to `Egypt_1975_Photos/`

### Requirement: Each crop sidecar receives DocumentID, DerivedFrom, Pantry, and the best available caption
The system SHALL write or update an XMP sidecar for each crop containing `xmpMM:DocumentID`, `xmpMM:DerivedFrom` and `xmpMM:Pantry` (referencing the source page `_V.jpg`), `dc:source` (relative path to the `_V.jpg`), and `dc:description` resolved from the caption priority chain.

The caption priority chain for each crop is:
1. The region's `dc:description` from the `mwg-rs:RegionList`
2. The region's `caption_hint` stored in the `mwg-rs:RegionList`
3. The page view sidecar's `dc:description`
4. Empty - no `dc:description` written on the crop sidecar

#### Scenario: Crop sidecar provenance links to page view JPEG
- **WHEN** a crop is written for region 2 of page P26
- **THEN** the crop sidecar's `xmpMM:DerivedFrom` references the `xmpMM:DocumentID` of `Egypt_1975_B00_P26_V.jpg`

#### Scenario: Region-specific caption used when present
- **WHEN** a region's `dc:description` in the `mwg-rs:RegionList` is non-empty
- **THEN** the crop sidecar's `dc:description` is set to that caption text

#### Scenario: caption_hint used when no dc:description is set on the region
- **WHEN** a region has no `dc:description` but has a non-empty `caption_hint` stored in the region's XMP
- **THEN** the crop sidecar's `dc:description` is set to the `caption_hint` text

#### Scenario: Page caption falls back to all crops when no per-region captions exist
- **WHEN** no region in the `mwg-rs:RegionList` has a `dc:description` or `caption_hint`, but the view sidecar has a non-empty `dc:description`
- **THEN** every crop sidecar for that page receives the view sidecar's `dc:description` as its `dc:description`

#### Scenario: No caption written when all sources are empty
- **WHEN** a region has no `dc:description`, no `caption_hint`, and the view sidecar has no `dc:description`
- **THEN** the crop sidecar has no `dc:description` field

### Requirement: Crop-sidecar updates preserve unrelated existing fields
The system SHALL update matching crop sidecars in place on rerun and SHALL preserve unrelated existing XMP fields when the crop path remains the same.

#### Scenario: Rerun preserves manual sidecar fields on matching crop path
- **WHEN** `_D01-00_V.xmp` already contains manual `dc:subject` entries and `crop-regions --force` regenerates `_D01-00_V.jpg`
- **THEN** the crop writer updates its owned fields and leaves the unrelated manual fields unchanged

### Requirement: Each crop sidecar inherits location, date, and subject metadata from the page view sidecar
The system SHALL copy the following page-level metadata from the page view sidecar to each crop sidecar at crop time: `exif:GPSLatitude`, `exif:GPSLongitude`, `photoshop:City`, `photoshop:State`, `photoshop:Country`, `Iptc4xmpExt:Sublocation`, `Iptc4xmpExt:LocationCreated`, `Iptc4xmpExt:LocationShown`, `xmp:CreateDate`, `dc:date`, and `dc:subject`. Fields that are absent or empty on the page view sidecar SHALL be omitted from the crop sidecar.

#### Scenario: GPS coordinates propagated to crop
- **WHEN** the page view sidecar has `exif:GPSLatitude` and `exif:GPSLongitude`
- **THEN** each crop sidecar written for that page contains the same GPS coordinates

#### Scenario: LocationShown bag propagated to all crops
- **WHEN** the page view sidecar's `Iptc4xmpExt:LocationShown` contains two location entries
- **THEN** every crop sidecar for that page contains the same two `LocationShown` entries

#### Scenario: Empty location fields not written to crop sidecar
- **WHEN** the page view sidecar has no GPS or city data
- **THEN** the crop sidecar contains no GPS or city fields

#### Scenario: Page date inherited by crops
- **WHEN** the page view sidecar has `xmp:CreateDate` set to a year value
- **THEN** each crop sidecar contains the same `xmp:CreateDate`

### Requirement: Existing crops are skipped without --force
The system SHALL skip writing a crop JPEG if the target file already exists, unless `--force` is passed.

#### Scenario: Existing crop skipped
- **WHEN** `_D01-00_V.jpg` already exists in `_Photos/` and `--force` is not set
- **THEN** that crop is not overwritten; other crops for the same page that are missing are still written

#### Scenario: --force overwrites existing crop JPEGs
- **WHEN** `crop-regions --force` is run for a page with existing crops
- **THEN** the matching crop JPEGs are regenerated

### Requirement: --force removes orphaned crop files before re-cropping
The system SHALL, when `--force` is set, delete orphaned `_D##-00_V.jpg` files (and their sidecars) in `_Photos/` for the page whose paths are no longer produced by the current region set. Matching output paths SHALL be updated in place instead of being deleted wholesale.

#### Scenario: Orphaned crop removed when region count decreases under --force
- **WHEN** a previous run wrote 3 crops and a new `crop-regions --force` run detects only 2 regions
- **THEN** `_D03-00_V.jpg` and its sidecar are deleted before completion
- **AND** `_D01-00_V.jpg` and `_D02-00_V.jpg` are regenerated in place

#### Scenario: No orphan deletion without --force
- **WHEN** `crop-regions` is run without `--force`
- **THEN** no existing crop files are deleted; only missing crops are written

### Requirement: Crop step completion is tracked in the page view sidecar's pipeline state
The system SHALL write `pipeline.crop_regions` to the page `_V.jpg` sidecar's `imago:Detections` when all crops for a page complete successfully. If any crop fails, the state SHALL NOT be written. The step SHALL be skipped on re-run if the state record is present and `--force` is not set.

#### Scenario: Successful crop records pipeline state on page view sidecar
- **WHEN** all crops for a page are written without error
- **THEN** the page view sidecar's `imago:Detections` contains `{"pipeline": {"crop_regions": {"completed": "<iso-timestamp>"}}, ...}`

#### Scenario: Pipeline state skips crop step on re-run
- **WHEN** `pipeline.crop_regions.completed` is present in the page view sidecar and `--force` is not set
- **THEN** the crop step is skipped and a skip message is printed

#### Scenario: Partial failure prevents pipeline state from being written
- **WHEN** one region fails to crop
- **THEN** `pipeline.crop_regions` is not written; the next pipeline run retries the crop step for that page
