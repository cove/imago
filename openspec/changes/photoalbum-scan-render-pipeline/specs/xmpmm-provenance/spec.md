## ADDED Requirements

### Requirement: All provenance logic lives in a dedicated xmpmm_provenance module
The system SHALL implement `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and `xmpMM:Pantry` write logic in `photoalbums/lib/xmpmm_provenance.py`. `xmp_sidecar.py` SHALL provide low-level XML primitives; `xmpmm_provenance.py` SHALL be the sole caller of those primitives for provenance fields.

#### Scenario: Provenance helpers are importable from xmpmm_provenance
- **WHEN** code imports `from photoalbums.lib.xmpmm_provenance import assign_document_id, write_derived_from, write_pantry_entry`
- **THEN** all three are available and `xmp_sidecar.py` contains no DocumentID, DerivedFrom, or Pantry write logic of its own

### Requirement: xmpMM:DocumentID is assigned at the point of file creation, not at the end of the pipeline
The system SHALL write `xmpMM:DocumentID` (`xmp:uuid:{uuid4}`) to an archive scan sidecar inside `_ensure_archive_page_sidecar` (the first operation of the render step), and to each rendered JPEG sidecar immediately after that JPEG is written. The final provenance step SHALL only write `DerivedFrom` and `Pantry`; it SHALL NOT assign new DocumentIDs.

#### Scenario: Archive scan receives DocumentID at render time
- **WHEN** the render step calls `_ensure_archive_page_sidecar` for a scan whose sidecar has no `xmpMM:DocumentID`
- **THEN** a UUID is written to the archive sidecar before stitching begins

#### Scenario: Rendered JPEG sidecar receives DocumentID immediately after write
- **WHEN** the render step writes `_V.jpg` and its sidecar does not contain `xmpMM:DocumentID`
- **THEN** a UUID is written to the rendered sidecar before the render step returns

#### Scenario: Existing DocumentID is not overwritten
- **WHEN** a sidecar already contains `xmpMM:DocumentID`
- **THEN** the existing value is preserved and no new UUID is generated

### Requirement: xmpMM:DerivedFrom links rendered outputs to their archive sources
The system SHALL write `xmpMM:DerivedFrom` on each rendered JPEG sidecar as an `stRef:ResourceRef` pointing to the primary archive source from which the rendered file was derived.

#### Scenario: View JPEG DerivedFrom references primary archive scan
- **WHEN** the provenance step runs for `Egypt_1975_B00_P26_V.jpg`
- **THEN** its sidecar contains `xmpMM:DerivedFrom` with `stRef:documentID` equal to the `xmpMM:DocumentID` of `Egypt_1975_Archive/Egypt_1975_B00_P26_S01.tif`

#### Scenario: Derived JPEG DerivedFrom references its source derived TIF or scan
- **WHEN** the provenance step runs for `Egypt_1975_B00_P26_D01-02_V.jpg`
- **THEN** its sidecar contains `xmpMM:DerivedFrom` referencing the `documentID` of the source `_D01-02` media file or primary archive scan

#### Scenario: DerivedFrom is updated on re-render
- **WHEN** a view JPEG is re-rendered (e.g. with `--force`) and the archive scan's DocumentID has changed
- **THEN** the `xmpMM:DerivedFrom` on the rendered sidecar is updated to reflect the current archive source DocumentID

### Requirement: xmpMM:Pantry holds one entry per unique DerivedFrom source
The system SHALL maintain a `xmpMM:Pantry` bag on each rendered sidecar containing one `rdf:Description` entry per unique DerivedFrom source document, storing the `documentID` and file path for offline reference. Duplicate entries for the same `documentID` SHALL be deduplicated on write.

#### Scenario: Pantry entry written alongside DerivedFrom
- **WHEN** `xmpMM:DerivedFrom` is written for a rendered JPEG
- **THEN** the sidecar's `xmpMM:Pantry` contains an entry with the same `stRef:documentID` and the relative path to the archive source

#### Scenario: Pantry entries are deduplicated
- **WHEN** the provenance step runs twice for the same rendered JPEG with the same archive source
- **THEN** the Pantry bag contains only one entry for that source documentID

### Requirement: Provenance step records completion in imago:Detections pipeline state
The system SHALL write a `pipeline.provenance` record to the rendered sidecar's `imago:Detections` JSON when DerivedFrom and Pantry are written, and SHALL skip the provenance step when that record is already present and `--force` is not set.

#### Scenario: Successful provenance write records pipeline state
- **WHEN** DerivedFrom and Pantry are written to a rendered sidecar
- **THEN** the sidecar's `imago:Detections` contains `{"pipeline": {"provenance": {"completed": "<iso-timestamp>"}}, ...}`

#### Scenario: Pipeline state skips provenance on re-run
- **WHEN** `write-provenance` is run and `pipeline.provenance.completed` is already present in the rendered sidecar's `imago:Detections` and `--force` is not set
- **THEN** the system skips writing DerivedFrom and Pantry and prints a skip message
