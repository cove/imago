## ADDED Requirements

### Requirement: Detect photo regions in a view image via vision model
The system SHALL call the LM Studio vision API (`google/gemma-4-26b-a4b`) with a view JPG and a structured prompt, and return a list of detected photo regions with normalised bounding boxes and per-region confidence scores.

#### Scenario: Successful detection of multiple adjacent photos
- **WHEN** a view image containing 2 or more photos packed edge-to-edge is submitted
- **THEN** the system returns a list of regions, each with `index`, pixel `x/y/width/height`, a `confidence` float (0–1), and an optional `caption_hint` string

#### Scenario: Model returns malformed JSON
- **WHEN** the vision model response cannot be parsed as a valid region list
- **THEN** the system retries the request up to 3 times with a stricter JSON-only prompt; if all retries fail, it returns an empty region list and logs an error

#### Scenario: Single photo fills the whole view image
- **WHEN** the model determines there is only one photo
- **THEN** the system returns a single region covering the content bounds of the image

### Requirement: Region detection model is configurable via ai_models.toml
The system SHALL read the view-region detection model identifier from `ai_models.toml` under the key `view-region`, defaulting to `google/gemma-4-26b-a4b`.

#### Scenario: Model override in config
- **WHEN** `ai_models.toml` contains `view-region = "some-other-model"`
- **THEN** the detection call uses `some-other-model` as the LM Studio model identifier

### Requirement: XMP sidecar is the single source of truth for detected regions
The system SHALL write detection results directly into the `_V.jpg` XMP sidecar as a `mwg-rs:RegionList` block. No separate JSON cache file is created.

#### Scenario: Cache hit on subsequent call
- **WHEN** `detect_view_regions` is called for a view image whose XMP sidecar already contains a `mwg-rs:RegionList` block and `force=False`
- **THEN** the system reads and returns the existing regions from the XMP without calling the vision model

#### Scenario: Force refresh
- **WHEN** `detect_view_regions` is called with `force=True`
- **THEN** the system calls the vision model and replaces the `mwg-rs:RegionList` block in the XMP sidecar with the new results
