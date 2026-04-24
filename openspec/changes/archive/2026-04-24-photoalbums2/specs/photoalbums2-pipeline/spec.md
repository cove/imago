## ADDED Requirements

### Requirement: The new pipeline lives in a separate top-level project
The system SHALL provide a `photoalbums2/` project at the monorepo root with its own `pyproject.toml`, independent of `photoalbums/`. The new project MUST NOT modify, rename, or move any file under `photoalbums/`.

#### Scenario: New project directory exists with its own package metadata
- **WHEN** the repository is inspected
- **THEN** `photoalbums2/pyproject.toml` exists
- **AND** `photoalbums2/__init__.py` exists
- **AND** no file under `photoalbums/` has been edited, renamed, or deleted as part of this change

#### Scenario: New project imports or copies legacy helpers deliberately
- **WHEN** `photoalbums2` needs existing XMP, filename, image geometry, geocoding, or Cast storage behavior
- **THEN** it MAY import stable library-like helpers from `photoalbums/` read-only
- **OR** copy small helper code into `photoalbums2` when the useful code is trapped inside legacy orchestration
- **AND** it MUST NOT copy large legacy orchestration modules wholesale

### Requirement: The pipeline is a Dagster asset graph over page and photo work units
The system SHALL define page-level assets for `stitch`, `regions`, and `crops`, and photo-level assets for `people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review`. A page work unit SHALL identify one album page. A photo work unit SHALL identify one cropped photo derived from a page.

#### Scenario: Page and photo assets exist
- **WHEN** the Dagster definitions are loaded
- **THEN** assets named `stitch`, `regions`, `crops`, `people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` are present
- **AND** `stitch`, `regions`, and `crops` operate on page work units
- **AND** `people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` operate on photo work units

#### Scenario: Work-unit keys preserve page and crop identity
- **WHEN** a page work unit is created
- **THEN** its key includes album, book, and page identity
- **WHEN** a photo work unit is created
- **THEN** its key includes album, book, page, and crop identity

#### Scenario: Render settings invalidate page outputs
- **WHEN** `render_settings` changes for an album page
- **THEN** `stitch` for that page becomes stale
- **AND** downstream `regions`, `crops`, and derived photo work units become stale as needed

### Requirement: Recognizing a new face triggers selective face refresh only
The system SHALL treat a newly recognized Cast face as a targeted invalidation event. Recognizing one new face SHALL only make `people` stale for photo work units whose stored embeddings are plausible matches for that face.

#### Scenario: New face recognition targets plausible matching photos
- **WHEN** a new face identity is confirmed in Cast
- **THEN** the system identifies photo work units with embeddings within the configured plausible-match threshold
- **AND** only those `people[photo]` work units become stale
- **AND** `stitch`, `regions`, `crops`, and `ocr` remain fresh
- **AND** unrelated `people[photo]` work units remain fresh

#### Scenario: Downstream work only reruns when face output changes
- **WHEN** a targeted `people[photo]` work unit is re-materialized
- **AND** its identified people output changes
- **THEN** downstream `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` for that same photo MAY become stale
- **AND** unrelated photo work units remain fresh

### Requirement: GPS location and IPTC LocationShown are explicit DAG assets
The system SHALL expose primary GPS/location resolution and IPTC `LocationShown` resolution as distinct photo-level assets so both appear in the Dagster DAG.

#### Scenario: Location query asset produces primary and shown-location queries
- **WHEN** `location_queries[photo]` runs
- **THEN** it produces one primary location query for the photo's EXIF GPS/scalar location metadata
- **AND** it produces zero or more named shown-location queries for IPTC `LocationShown`
- **AND** the prompt, response, and query outputs are recorded in Dagster metadata

#### Scenario: GPS location asset resolves primary query
- **WHEN** `gps_location[photo]` runs with a non-empty primary query
- **THEN** it resolves the query through the configured geocoder
- **AND** it outputs GPS latitude, GPS longitude, map datum, source, and available scalar location fields such as city, state, country, and sublocation
- **AND** the geocoder query and result are recorded in Dagster metadata

#### Scenario: Locations shown asset resolves named shown-location queries
- **WHEN** `locations_shown[photo]` runs with named shown-location queries
- **THEN** it resolves each query through the configured geocoder when possible
- **AND** it outputs IPTC `LocationShown` rows with name, world region, country, province/state, city, sublocation, and optional GPS/source fields
- **AND** the geocoder queries and results are recorded in Dagster metadata

#### Scenario: Candidate XMP includes GPS and LocationShown outputs
- **WHEN** candidate XMP is written for a photo
- **THEN** `gps_location[photo]` output is written to EXIF GPS fields and scalar location fields
- **AND** `locations_shown[photo]` output is written to the IPTC `LocationShown` bag

### Requirement: Technical retries are separate from AI rewrite retries
The system SHALL use Dagster retry policy for technical execution failures and a step-owned AI retry ladder for invalid or low-quality model output.

#### Scenario: Connection failure uses Dagster retry
- **WHEN** an LLM call fails due to connection error, timeout, HTTP 5xx, invalid response envelope, file lock, or process crash
- **THEN** the Dagster asset run fails with the underlying error surfaced
- **AND** Dagster retries the asset according to its configured retry policy
- **AND** the retry uses the same logical prompt/input unless the user explicitly changes configuration

#### Scenario: Invalid model output uses the AI retry ladder
- **WHEN** an LLM call returns an empty response, incomplete JSON, invalid JSON, schema mismatch, unexpected double quotes inside caption text, prompt echo, obvious truncation, or known duplicate-character spam
- **THEN** the asset does not write candidate XMP
- **AND** the step retries inside the same asset run using the next retry-ladder rung or a rewrite instruction
- **AND** each attempt records prompt, response, sampler settings, and failure reason in Dagster metadata

#### Scenario: Exhausting AI retries requires review
- **WHEN** all retry-ladder rungs for a step are exhausted
- **THEN** the photo work unit is marked `needs_review`
- **AND** no candidate XMP is written for the failed step
- **AND** diagnostic artifacts are written only to debug output

### Requirement: Each prompt is version-controlled and visible in run metadata
The system SHALL define each step's system prompt, Jinja user-prompt template, Pydantic output schema, and retry ladder in a single Python file under `photoalbums2/prompts/`. The new project MUST NOT load prompt sections from `SKILL.md` at runtime.

#### Scenario: Prompt file is the source of truth
- **WHEN** a reader opens a step file under `photoalbums2/prompts/`
- **THEN** the system prompt, user-prompt template, output schema, and retry ladder for that step are present
- **AND** the final prompt text delivered to the model can be reconstructed from that file plus runtime variables

#### Scenario: Prompt run metadata is inspectable
- **WHEN** an LLM-backed asset runs
- **THEN** Dagster metadata includes the prompt file path, prompt version or hash, rendered prompt text, sampler settings, raw model response, and retry history

#### Scenario: SKILL.md is not loaded at runtime
- **WHEN** the new pipeline runs an inference step
- **THEN** no code path in `photoalbums2/` reads `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` at runtime

### Requirement: Static validation gates writes and semantic review handles visual mistakes
The system SHALL run conservative static validators before candidate XMP is written. The system SHALL also provide an AI-backed `semantic_review` asset for mistakes requiring visual judgment.

#### Scenario: Static validation prevents malformed candidate XMP
- **WHEN** a step's output fails static validation after all retry-ladder attempts
- **THEN** no candidate XMP is written for that step
- **AND** the photo work unit is marked `needs_review`

#### Scenario: Semantic review can request AI rewrite
- **WHEN** `semantic_review[photo]` determines that a caption, GPS location, LocationShown row, date, or people-derived caption context appears wrong for the crop/page evidence
- **THEN** the system requests a rewrite from the responsible step instead of applying brittle string replacement
- **AND** the review result and rewrite attempts are recorded in Dagster metadata

#### Scenario: Good semantic review passes without rewrite
- **WHEN** `semantic_review[photo]` finds the candidate XMP consistent with the crop image, page image, and available metadata context
- **THEN** no rewrite is requested
- **AND** the photo work unit is marked review-passed

### Requirement: Legacy XMP is importable and outputs use dry-run, staging, or promote modes
The system SHALL provide a bootstrap/import or observe path that reads existing `.xmp` sidecars written by `photoalbums/` as starting materializations. New outputs from `photoalbums2` SHALL use an explicit output mode: dry-run, staging, or promote.

#### Scenario: Existing sidecar is recognized as materialized
- **WHEN** the import/observe job sees a valid legacy `.xmp` sidecar
- **THEN** it records a materialization for the relevant work unit keyed by deterministic content and input hashes
- **AND** no LLM call is made merely to import existing work

#### Scenario: Dry-run writes only under debug output
- **WHEN** a step produces output in dry-run mode
- **THEN** candidate XMP and diagnostics are written under `_debug/photoalbums2/`
- **AND** the canonical `<stem>.xmp` is left unchanged
- **AND** no adjacent `<stem>.xmp.new` is written

#### Scenario: Staging writes adjacent candidate sidecar
- **WHEN** a step produces output in staging mode
- **THEN** the output is written to `<stem>.xmp.new` adjacent to the image
- **AND** the existing `<stem>.xmp` is left unchanged

#### Scenario: Promotion is explicit
- **WHEN** a run completes in dry-run or staging mode
- **THEN** no candidate XMP is automatically promoted to canonical `.xmp`
- **AND** canonical `.xmp` is changed only by an explicit promote command or action

### Requirement: Human review UI provides a per-photo inspection view
The system SHALL provide a Streamlit review UI that lets a user select one photo at a time and see: the page image, crop image, each step's prompt sent, each step's response received, retry/validation history, and a diff between candidate XMP and canonical XMP.

#### Scenario: Review UI reads Dagster metadata for a selected photo
- **WHEN** the reviewer selects a photo in the Streamlit UI
- **THEN** the app displays the page image and crop image for that photo
- **AND** for each step asset with a materialization for that photo, the app shows prompt, response, sampler settings, and retry/validation history
- **AND** the app shows a side-by-side diff of candidate XMP against canonical XMP

#### Scenario: Review UI exposes a single-work-unit rerun trigger
- **WHEN** the reviewer clicks a rerun action for a photo's step asset
- **THEN** Dagster is invoked to re-run that single work unit

### Requirement: Dagster UI is reachable remotely
The system SHALL document and configure local Dagster development so the UI binds to `0.0.0.0`, allowing remote access on the local network.

#### Scenario: Dagster dev binds to all interfaces
- **WHEN** the user starts the `photoalbums2` Dagster UI using the documented command
- **THEN** Dagster listens on host `0.0.0.0`
- **AND** the documented URL includes the configured port

### Requirement: Inference provider is abstracted but defaults to LM Studio
The system SHALL expose a provider interface that supports OpenAI-compatible chat-completion endpoints, with LM Studio as the default implementation. Provider selection SHALL be a configuration value, not a code change. This change MUST NOT replace LM Studio with a different provider as its default.

#### Scenario: LM Studio is the default provider
- **WHEN** `photoalbums2` runs with default configuration
- **THEN** chat-completion requests go to the LM Studio base URL

#### Scenario: Provider is swappable by config
- **WHEN** configuration selects a different OpenAI-compatible provider
- **THEN** chat-completion requests go to that provider without changes to step agents or prompt files
