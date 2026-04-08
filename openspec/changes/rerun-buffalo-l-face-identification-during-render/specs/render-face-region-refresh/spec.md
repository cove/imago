## ADDED Requirements

### Requirement: Rendered outputs refresh person-identifying metadata from rendered pixels
The `photoalbums` render process SHALL run a fresh Cast-backed `buffalo_l` face-identification pass for every  rendered JPEG output sidecar that it writes or refreshes when the image was stitched. The refresh SHALL analyze the rendered output image itself and SHALL write the refreshed person-identifying metadata to that rendered file's sidecar.

#### Scenario: Single-scan page render refreshes copied sidecar people metadata
- **WHEN** render creates a `_V.jpg` page output from a single archive scan and copies the archive sidecar to the rendered output
- **THEN** render SHALL run a fresh Cast-backed `buffalo_l` face-identification pass against the rendered `_V.jpg`
- **THEN** the rendered output sidecar SHALL contain the refreshed rendered-image people result instead of leaving the copied person-identifying metadata untouched

#### Scenario: Stitched page render refreshes people metadata
- **WHEN** render creates a stitched `_V.jpg` page output from multiple archive scans
- **THEN** render SHALL run the same Cast-backed `buffalo_l` face-identification refresh against the stitched rendered `_V.jpg`
- **THEN** the rendered output sidecar SHALL reflect the stitched image's refreshed people result

#### Scenario: Derived render refreshes people metadata
- **WHEN** render creates a derived `_D##-##_V.jpg` output
- **THEN** render SHALL run the same Cast-backed `buffalo_l` face-identification refresh against the derived rendered JPEG
- **THEN** the rendered output sidecar SHALL reflect the derived image's refreshed people result

### Requirement: Render-time people refresh replaces inherited person-identifying regions and names
When render refreshes person-identifying metadata for a rendered output, the refreshed Cast-backed match result SHALL be the source of truth for rendered-output face regions and person names. Render SHALL replace inherited person-identifying regions and names instead of unioning them with stale copied values.

#### Scenario: Existing face regions are replaced by refreshed matches
- **WHEN** a rendered output sidecar already contains face-identifying `ImageRegion` entries and `PersonInImage` names from an inherited sidecar copy
- **THEN** render SHALL remove those prior person-identifying regions before writing the refreshed rendered-image face matches
- **THEN** the rendered output `PersonInImage` names SHALL come from the refreshed Cast-backed match result

#### Scenario: No rendered-image matches remove stale inherited person metadata
- **WHEN** a copied rendered-output sidecar contains inherited person-identifying regions or names but the fresh rendered-image Cast-backed pass finds no identified people
- **THEN** render SHALL remove the stale inherited person-identifying regions from the rendered output sidecar
- **THEN** render SHALL not preserve inherited `PersonInImage` names that are unsupported by the fresh rendered-image pass

### Requirement: Render-time people refresh preserves unrelated render metadata
Render-time people refresh SHALL preserve non-person metadata already present on the rendered output sidecar, including OCR, source references, processing history, location metadata, and any non-face image regions.

#### Scenario: Non-face image regions remain after face refresh
- **WHEN** a rendered output sidecar contains both face-identifying `ImageRegion` entries and non-face image-region entries
- **THEN** render SHALL replace only the person-identifying region entries
- **THEN** render SHALL preserve the non-face image-region entries

#### Scenario: Non-people XMP fields remain after face refresh
- **WHEN** render refreshes people metadata for a rendered output sidecar that already contains non-people XMP fields
- **THEN** render SHALL preserve those non-people XMP fields while updating the person-identifying metadata
