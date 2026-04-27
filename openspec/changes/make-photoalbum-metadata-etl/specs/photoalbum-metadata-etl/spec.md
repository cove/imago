## ADDED Requirements

### Requirement: Metadata pipeline has explicit ETL boundaries
The system SHALL process normal photo album metadata writes through explicit extract, transform, and load phases.

Extraction SHALL read exact source facts from XMP sidecars, region lists, model outputs, and file naming without applying fallback or coalescing. Transformation SHALL apply ownership, precedence, inheritance, filtering, and preservation policy. Loading SHALL serialize resolved records to XMP without inferring missing canonical values from unrelated fields.

#### Scenario: Raw extraction does not fill missing captions
- **WHEN** a crop sidecar has no `dc:description`
- **THEN** raw extraction reports the crop description as empty
- **AND** raw extraction does not fill it from page `dc:description`, region `CaptionHint`, OCR text, or existing effective display helpers

#### Scenario: Loading does not infer unrelated values
- **WHEN** the loader receives a resolved crop metadata record with an empty description
- **THEN** the written crop XMP has no canonical crop `dc:description`
- **AND** the loader does not infer a description from page text, OCR text, or location text

### Requirement: Normal pipeline writes use resolved metadata records
The system SHALL write page and crop canonical XMP fields from resolved metadata records produced by the metadata resolver. Normal pipeline write paths SHALL NOT call effective/coalescing sidecar readers as the source of canonical field values.

#### Scenario: Crop description comes from resolved region caption
- **WHEN** a page region has `mwg-rs:Name` set to `OXFORD STREET, LONDON, ENGLAND - AUG. 1988`
- **AND** the corresponding crop has no existing description
- **THEN** the resolver emits a crop description with that value
- **AND** the crop writer serializes that value to crop `dc:description`

#### Scenario: Empty resolved value remains empty
- **WHEN** a region has no `mwg-rs:Name`
- **AND** the resolver policy does not declare any inheritance or preservation source
- **THEN** the resolved crop description is empty
- **AND** the crop writer does not substitute `CaptionHint` or page `dc:description`

### Requirement: Resolved fields include source provenance
The system SHALL attach source provenance to canonical fields resolved by the ETL transform layer. The provenance SHALL identify the source field or policy that produced the value.

#### Scenario: Region caption provenance is recorded
- **WHEN** a crop description is resolved from the first page region's `mwg-rs:Name`
- **THEN** the resolved field source identifies `page.region[1].mwg-rs:Name`
- **AND** the source is available for diagnostics after the crop XMP is written

#### Scenario: Preserved existing value provenance is explicit
- **WHEN** a crop already has a manually preserved description
- **AND** the preservation policy keeps that value
- **THEN** the resolved field source identifies the existing crop field and preservation policy

### Requirement: Fallback is limited to named non-normal modes
The system SHALL allow fallback-derived metadata only in explicitly named migration, repair, or display-only paths. Normal pipeline writes SHALL NOT use fallback unless the resolver policy names that fallback and records its source.

#### Scenario: Repair fallback records its source
- **WHEN** a repair command fills a missing crop description from a legacy page caption
- **THEN** the repair output records that the value came from a repair fallback
- **AND** the normal pipeline resolver does not treat that fallback as an implicit rule

#### Scenario: Display fallback does not write canonical fields
- **WHEN** a UI or diagnostic view displays an effective caption using fallback logic
- **THEN** that displayed value is not written back to canonical XMP fields unless an explicit repair or resolver policy is invoked
