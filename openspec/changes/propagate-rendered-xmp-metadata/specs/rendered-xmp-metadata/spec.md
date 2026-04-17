## ADDED Requirements

### Requirement: Rendered page view sidecars inherit copy-safe top-level metadata from the source page sidecar
The system SHALL propagate copy-safe top-level metadata from the source page sidecar to the rendered page `_V.jpg` sidecar when that metadata is present and non-empty upstream. At minimum the inherited field set SHALL include `dc:title`, `dc:description`, `dc:subject`, `dc:date`, `xmp:CreateDate`, `exif:DateTimeOriginal`, `imago:AlbumTitle`, `imago:OCRText`, `imago:AuthorText`, `imago:SceneText`, `photoshop:City`, `photoshop:State`, `photoshop:Country`, `Iptc4xmpExt:Sublocation`, `Iptc4xmpExt:LocationCreated`, and `Iptc4xmpExt:LocationShown`.

#### Scenario: Source page location bag reaches the rendered page sidecar
- **WHEN** the source page sidecar contains two `Iptc4xmpExt:LocationShown` entries and a non-empty `Iptc4xmpExt:LocationCreated`
- **THEN** the rendered page `_V.jpg` sidecar contains the same two `LocationShown` entries
- **AND** preserves the same `LocationCreated` value

#### Scenario: Source page caption and dates reach the rendered page sidecar
- **WHEN** the source page sidecar contains `dc:description`, `dc:date`, and `xmp:CreateDate`
- **THEN** the rendered page `_V.jpg` sidecar contains the same caption and date values unless a later page-level step intentionally replaces one of them

### Requirement: Crop sidecars inherit accepted page-view metadata
The system SHALL propagate the rendered page view sidecar's copy-safe top-level metadata to each crop `_D##-00_V.jpg` sidecar at crop time. When both the parent page view sidecar and the archive page sidecar are available, the parent page view sidecar SHALL be the authoritative source for crop inheritance because it represents the accepted rendered-page metadata state.

#### Scenario: Crop inherits structured location metadata from the page view
- **WHEN** the page view sidecar contains `Iptc4xmpExt:LocationShown`, `Iptc4xmpExt:LocationCreated`, and top-level city/country fields
- **THEN** each crop sidecar for that page contains the same `LocationShown` entries
- **AND** the same `LocationCreated`, city, and country values

#### Scenario: Crop inherits caption, subjects, and dates from the page view
- **WHEN** the page view sidecar contains a page-level `dc:description`, `dc:subject`, `dc:date`, and `xmp:CreateDate`
- **THEN** each crop sidecar receives those same values unless a crop-specific caption or other crop-specific override is intentionally chosen for that field

### Requirement: Rendered JPEG provenance separates archive lineage from immediate rendered parentage
For rendered JPEG sidecars, `dc:source` SHALL identify the archive scan lineage, while immediate rendered-to-rendered derivation SHALL be recorded using `xmpMM:DerivedFrom` and `xmpMM:Pantry`.

#### Scenario: Crop keeps archive lineage and page-view parentage
- **WHEN** a crop JPEG is generated from a rendered page `_V.jpg`
- **THEN** the crop sidecar's `dc:source` references the archive scan source chain for that page
- **AND** the crop sidecar's `xmpMM:DerivedFrom` references the parent page `_V.jpg`

### Requirement: Later sidecar rewrites preserve inherited top-level metadata they do not own
Any later step that rewrites an existing rendered page or crop sidecar, including people refresh, pipeline-state updates, and similar metadata updates, SHALL preserve inherited top-level metadata fields that the step is not intentionally recomputing or replacing.

#### Scenario: Face refresh preserves existing location bag and caption
- **WHEN** a rendered page or crop sidecar already contains inherited `Iptc4xmpExt:LocationShown`, `Iptc4xmpExt:LocationCreated`, and `dc:description`
- **AND** a later face-refresh-style step updates people metadata only
- **THEN** the rewritten sidecar still contains the same `LocationShown`, `LocationCreated`, and caption values

#### Scenario: Missing fresh location payload does not erase inherited location metadata
- **WHEN** a later rewrite step has no new `locations_shown` payload for a rendered page or crop sidecar that already contains inherited `Iptc4xmpExt:LocationShown`
- **THEN** the existing `LocationShown` bag remains in the sidecar rather than being cleared

### Requirement: Crop refresh paths can resolve parent page context from rendered-parent linkage
The system SHALL be able to recover parent page OCR, caption, and location context for crop sidecars from their rendered-parent linkage or equivalent explicit parent-view metadata path. It SHALL NOT assume that archive-scan filenames in `dc:source` are sufficient to recover all crop parent context.

#### Scenario: Crop refresh resolves page context even when dc:source names only archive scans
- **WHEN** a crop sidecar's `dc:source` names archive scan files and its immediate rendered parent is the page `_V.jpg`
- **THEN** a later crop metadata refresh can still resolve the parent page's accepted OCR and location context from the rendered-parent linkage
- **AND** does not drop inherited page metadata merely because the crop `dc:source` does not name the page `_V.jpg`
