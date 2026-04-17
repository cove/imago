## Requirements

### Requirement: Photo restoration runs automatically after each crop is written
The crop pipeline SHALL call `restore_photo()` on each crop JPEG immediately after the file is written to disk, before moving on to the next region. Restoration is applied by default and is best-effort: a failure does not abort the crop step.

#### Scenario: Restoration succeeds for a crop
- **WHEN** `crop_page_regions()` has cropped a region into an in-memory PIL Image
- **THEN** `restore_photo(crop_img)` is called before saving
- **AND** the returned restored image is saved as the crop JPEG to `_Photos/`
- **AND** the JPEG is written exactly once (no intermediate save/reload cycle)
- **AND** the crop sidecar is written after the restored image is saved

#### Scenario: Restoration is skipped via flag
- **WHEN** `crop_page_regions()` is called with `skip_restoration=True`
- **THEN** `restore_photo()` is not called for any crop
- **AND** crop JPEGs are written as-is

#### Scenario: Restoration fails for one crop
- **WHEN** `restore_photo()` raises or logs a warning for one crop
- **THEN** `crop_page_regions()` continues writing remaining crops
- **AND** the failed crop retains its unrestored JPEG
- **AND** the crop count returned is unchanged (the crop was written, just not restored)
