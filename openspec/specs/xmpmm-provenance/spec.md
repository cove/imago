## ADDED Requirements

### Requirement: All provenance logic lives in a dedicated xmpmm_provenance module
The system SHALL implement `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` write logic in `photoalbums/lib/xmpmm_provenance.py`. `xmp_sidecar.py` SHALL provide low-level XML primitives; `xmpmm_provenance.py` SHALL be the sole caller of those primitives for provenance fields.

#### Scenario: Provenance helpers are importable from xmpmm_provenance
- **WHEN** code imports `from photoalbums.lib.xmpmm_provenance import assign_document_id, write_derived_from, write_pantry_entry`
- **THEN** all three are available and `xmp_sidecar.py` contains no DocumentID, DerivedFrom, or Pantry write logic of its own

### Requirement: xmpMM:DocumentID is assigned at the point of file creation
The system SHALL write `xmpMM:DocumentID` (`xmp:uuid:{uuid4}`) to an archive scan sidecar inside `_ensure_archive_page_sidecar` before render begins, and to each rendered or cropped JPEG sidecar immediately after that output file is created.

#### Scenario: Archive scan receives DocumentID before render work
- **WHEN** the render step calls `_ensure_archive_page_sidecar` for a scan whose sidecar has no `xmpMM:DocumentID`
- **THEN** a UUID is written to the archive sidecar before stitching begins

#### Scenario: Rendered JPEG sidecar receives DocumentID immediately after write
- **WHEN** the render step writes `_V.jpg` and its sidecar does not contain `xmpMM:DocumentID`
- **THEN** a UUID is written to the rendered sidecar before the render step returns

#### Scenario: Existing DocumentID is not overwritten
- **WHEN** a sidecar already contains `xmpMM:DocumentID`
- **THEN** the existing value is preserved and no new UUID is generated

### Requirement: DerivedFrom and Pantry are written as soon as a new output's source set is known
The system SHALL write `xmpMM:DerivedFrom` and `xmpMM:Pantry` when each new output file is created, rather than waiting for a late pipeline step.

#### Scenario: View JPEG DerivedFrom references the primary archive scan
- **WHEN** `Egypt_1975_B00_P26_V.jpg` is created
- **THEN** its sidecar contains `xmpMM:DerivedFrom` with `stRef:documentID` equal to the `xmpMM:DocumentID` of `Egypt_1975_Archive/Egypt_1975_B00_P26_S01.tif`

#### Scenario: Stitched view Pantry records every contributing scan
- **WHEN** a stitched page view is created from `_S01.tif`, `_S02.tif`, and `_S03.tif`
- **THEN** the view sidecar's `xmpMM:Pantry` contains one entry for each contributing scan documentID
- **AND** duplicate entries are deduplicated on write

#### Scenario: Derived JPEG provenance is written at render time
- **WHEN** `Egypt_1975_B00_P26_D01-02_V.jpg` is created
- **THEN** its sidecar receives `xmpMM:DerivedFrom` and `xmpMM:Pantry` immediately using the source derived media or scan

#### Scenario: Crop sidecar provenance links to the page view when the crop is created
- **WHEN** crop `_D01-00_V.jpg` is created from `Egypt_1975_B00_P26_V.jpg`
- **THEN** the crop sidecar receives `xmpMM:DocumentID`
- **AND** `xmpMM:DerivedFrom` references the `xmpMM:DocumentID` of the page view
- **AND** `xmpMM:Pantry` includes the page view as a pantry source

### Requirement: Provenance writes preserve unrelated XMP fields
The system SHALL update provenance fields in the canonical sidecar in place and SHALL preserve unrelated XMP fields already present in that sidecar.

#### Scenario: Adding provenance does not remove unrelated metadata
- **WHEN** a rendered sidecar already contains location metadata and manual `dc:subject` fields
- **THEN** writing `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` leaves those unrelated fields unchanged
