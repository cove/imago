## ADDED Requirements

### Requirement: Render-time face refresh runs Cast-backed buffalo_l against the rendered JPEG
The system SHALL, after a rendered JPEG is written, run Cast-backed `buffalo_l` face identification against the rendered image pixels and use the match results as the sole source for person-identifying `ImageRegion` metadata on that rendered sidecar.

#### Scenario: Fresh face regions written to rendered view JPEG sidecar
- **WHEN** the render pipeline produces `Egypt_1975_B00_P26_V.jpg` and face refresh runs
- **THEN** the `_V.jpg` sidecar contains `Iptc4xmpExt:ImageRegion` entries sourced from the rendered-image Cast match, not from the archive sidecar copy

#### Scenario: Cast store unavailable at render time
- **WHEN** the Cast store cannot be loaded during face refresh
- **THEN** face refresh is skipped for that image, a warning is printed including the underlying error, and the rendered sidecar retains whatever person regions it already has; the pipeline continues

#### Scenario: Face refresh runs for derived JPEG outputs
- **WHEN** the pipeline produces a `_D##-##_V.jpg` derived output
- **THEN** face refresh also runs on that derived JPEG using the rendered derived image pixels

### Requirement: Only person-identifying ImageRegion entries are replaced; non-person regions are preserved
The system SHALL remove only `Iptc4xmpExt:ImageRegion` entries whose `Iptc4xmpExt:RCtype` identifies them as face or person regions (`face-*` type values), write the refreshed face regions, and leave all other `ImageRegion` entries untouched.

#### Scenario: Non-face photo regions preserved after face refresh
- **WHEN** the rendered sidecar contains both `face-*` type image regions and non-face photo regions
- **THEN** after face refresh the sidecar retains the non-face regions unchanged and only the face-type regions are replaced

#### Scenario: Stale inherited face names removed
- **WHEN** the rendered sidecar contains inherited face names that no longer match in the Cast store for the rendered image
- **THEN** those person-identifying entries are removed and not carried forward

### Requirement: Identified person names are written to Iptc4xmpExt:PersonInImage
The system SHALL, after running buffalo_l + Cast matching, collect the names of all identified people and write them as the `Iptc4xmpExt:PersonInImage` bag on the sidecar, replacing any previously stored values. This is separate from the `ImageRegion` bounding-box entries and allows photo management tools to search/filter by person name without parsing region geometry.

#### Scenario: Identified face names written to PersonInImage
- **WHEN** face refresh runs and Cast matches two faces to "Alice Smith" and "Bob Jones"
- **THEN** the sidecar's `Iptc4xmpExt:PersonInImage` bag contains exactly `["Alice Smith", "Bob Jones"]`

#### Scenario: PersonInImage cleared when no faces are identified
- **WHEN** face refresh runs and buffalo_l finds faces but Cast matches none above threshold
- **THEN** the sidecar's `Iptc4xmpExt:PersonInImage` bag is empty (any stale names from a prior run are removed)

#### Scenario: PersonInImage on crop sidecar reflects crop-level identification
- **WHEN** face refresh runs on a `_D##-00_V.jpg` crop that contains one identifiable face
- **THEN** the crop sidecar's `Iptc4xmpExt:PersonInImage` contains that person's name

### Requirement: Face refresh records completion in imago:Detections pipeline state
The system SHALL write a `pipeline.face_refresh` record to the rendered sidecar's `imago:Detections` JSON when face refresh succeeds, and SHALL skip refresh when that record is already present and `--force` is not set.

#### Scenario: Successful refresh writes pipeline state
- **WHEN** face refresh runs and writes new face regions to the rendered sidecar
- **THEN** the sidecar's `imago:Detections` contains `{"pipeline": {"face_refresh": {"completed": "<iso-timestamp>", "model": "buffalo_l"}}, ...}`

#### Scenario: Pipeline state skips face refresh on re-run
- **WHEN** `face-refresh` is run and `pipeline.face_refresh.completed` is already present in the rendered sidecar's `imago:Detections` and `--force` is not set
- **THEN** the system skips the buffalo_l pass and prints a skip message

#### Scenario: Failed refresh does not write pipeline state
- **WHEN** face refresh fails (e.g. Cast store unavailable)
- **THEN** no `pipeline.face_refresh` record is written and the sidecar is unchanged

### Requirement: Face refresh reuses the existing ai_index_runner people-refresh logic
The system SHALL expose a narrow render-time entrypoint that calls the same Cast loading, buffalo_l matching, and processing-state update path used by `ai_index_runner._process_people_update`, rather than adding a second face-matching implementation.

#### Scenario: Render-time entrypoint delegates to existing people-refresh path
- **WHEN** face refresh runs during the pipeline
- **THEN** it uses the same Cast matcher configuration (threshold, min face size) as the normal index run, loaded from the same `ai_models.toml` settings
