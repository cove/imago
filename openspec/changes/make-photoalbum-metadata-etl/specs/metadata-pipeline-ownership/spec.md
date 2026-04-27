## MODIFIED Requirements

### Requirement: Metadata production ownership is explicit by subsystem
The system SHALL treat metadata production as a set of subsystem-owned stages with explicit ownership boundaries:

- Docling owns page layout and accepted photo-region geometry.
- Gemma owns caption and location/date semantic association for accepted regions.
- `buffalo_l` owns people identity.
- YOLO owns object detections.
- Nominatim owns resolved location payloads derived from location queries.
- The metadata resolver owns precedence, inheritance, filtering, preservation, fallback policy, and effective page/crop metadata.
- XMP writers own field serialization, provenance fields, and preservation of unrelated XMP content.

No subsystem SHALL emit or overwrite fields outside its owned stage except through the metadata resolver and XMP writers. Normal pipeline readers and writers SHALL NOT apply hidden fallback or coalescing when producing canonical metadata fields.

#### Scenario: People identity remains owned by buffalo_l
- **WHEN** a crop contains an identifiable face and Gemma also sees person-like text on the page
- **THEN** the final person identity used for `PersonInImage` comes from `buffalo_l`
- **AND** Gemma does not become the authority for people identity on that crop

#### Scenario: Geocoding remains owned by Nominatim
- **WHEN** Gemma returns a location query such as `"Karnten, Austria"`
- **THEN** Nominatim resolves the GPS and normalized city/state/country payload
- **AND** Gemma does not become the authority for final resolved GPS coordinates

#### Scenario: Writer does not become a fallback owner
- **WHEN** the resolver emits an empty crop description
- **THEN** the XMP writer serializes the empty canonical value
- **AND** the writer does not fill it from page description, OCR text, caption hint, or any existing effective reader output

### Requirement: Effective page and crop metadata is computed by a deterministic resolver
The system SHALL compute effective page metadata and effective crop metadata in a deterministic resolver layer before XMP write. Final top-level XMP fields SHALL be treated as resolver outputs rather than raw producer-owned facts.

The resolver SHALL be responsible for:
- inheritance from page to crop where appropriate
- precedence between manual overrides, region assignments, and page fallbacks
- filtering invalid values such as location strings leaking into people-name outputs
- mapping producer outputs into final page/crop field values
- declaring any fallback or preservation policy used for a resolved field
- attaching source provenance to resolved canonical fields

#### Scenario: Crop location follows resolver precedence
- **WHEN** a crop has both a region location override and a page-level location fallback
- **THEN** the resolver chooses the region override as the effective crop location
- **AND** the page-level location is used only if no higher-precedence crop-specific value applies

#### Scenario: Page facts are inherited without re-deriving intent
- **WHEN** page-level OCR, date, and album-title facts already exist upstream
- **THEN** the resolver carries those values forward into effective page/crop metadata according to inheritance rules
- **AND** later write paths do not re-infer those values from unrelated fields

#### Scenario: Crop caption follows region name authority
- **WHEN** a page region has a non-empty `mwg-rs:Name`
- **AND** the corresponding crop description is empty
- **THEN** the resolver uses `mwg-rs:Name` as the crop description source
- **AND** the resolved field provenance identifies the page region name field

### Requirement: Final PersonInImage is a resolved output, not a raw producer field
The system SHALL treat `Iptc4xmpExt:PersonInImage` as a resolved sidecar output rather than as a direct mirror of any one upstream producer. The resolver SHALL assemble it from the authoritative people-identity source for that image scope and SHALL filter out values that conflict with known location strings before write.

#### Scenario: Location text is removed from final PersonInImage
- **WHEN** the people-identity input contains a candidate such as `"KARNTEN, AUSTRIA"` and the same sidecar also contains that value as a known location string
- **THEN** the resolver removes the location string before writing final `PersonInImage`

#### Scenario: Valid person identity survives filtering
- **WHEN** the authoritative people-identity input contains `"Audrey Cordell"` and no known location string matches it
- **THEN** the resolver preserves `"Audrey Cordell"` in final `PersonInImage`
