## ADDED Requirements

### Requirement: LM Studio model assigns captions to photos using numbered-photo prompt
The system SHALL send the album page image to the configured LM Studio model using a prompt that instructs it to number photos left-to-right/top-to-bottom and return a JSON object mapping photo numbers to captions:

```
`{"photo-1": "", "photo-2": "", "photo-3": ""}`
- Number as many `photo-N` keys as there are distinct photos on the page.
- Each value is the caption text that belongs to that photo; empty string if there is no caption.
- If a caption refers to subjects shown in an adjacent photo, prepend the missing subject so the caption reads standalone. Do not rewrite or summarise; only prepend the minimum context needed.
- Just return the JSON without any extra text or explanation.
```

The response SHALL be parsed as JSON and returned as a mapping of 1-based photo index to caption string. The model used is selected by the `caption_matching_model` key in `ai_models.toml`.

#### Scenario: Model returns valid JSON caption mapping
- **WHEN** the LM Studio model responds with a JSON object keyed `photo-1`, `photo-2`, etc.
- **THEN** the system parses the JSON and returns a dict mapping integer index to caption string for each key present

#### Scenario: Model response contains extra text around JSON
- **WHEN** the model wraps the JSON in markdown code fences or leading/trailing prose
- **THEN** the system extracts the JSON object via regex and parses it, ignoring surrounding text

#### Scenario: Model returns malformed JSON
- **WHEN** the LM Studio response cannot be parsed as valid JSON after extraction
- **THEN** the system logs a WARNING with the raw response and returns an empty mapping; region detection continues without captions

#### Scenario: LM Studio is offline
- **WHEN** the LM Studio endpoint is unreachable when the caption-matching step runs
- **THEN** the system logs a WARNING and returns an empty mapping; region detection continues without captions

#### Scenario: Caption refers to subject shown in adjacent photo
- **WHEN** the model detects that a caption (e.g. "THEIR NEW CONDOMINIUM IN VICTORIA CANADA") refers to a subject established in an adjacent photo (e.g. "GILBERT & HELEN")
- **THEN** the model prepends the subject so the caption reads standalone (e.g. "GILBERT & HELEN — THEIR NEW CONDOMINIUM IN VICTORIA CANADA")

### Requirement: Caption assignment merges LM Studio output with Docling bounding boxes via coordinate-based sort
The system SHALL sort the Docling-detected `RegionResult` list using a strict coordinate-based scanline sort before assigning captions. Docling does not output boxes in reading order — boxes MUST always be sorted explicitly. The sort SHALL group regions into rows using a y-tolerance band expressed as a ratio (0.0–1.0) of image height (configurable, default 0.10), then sort rows top-to-bottom and regions within each row left-to-right by x coordinate. The 1-based position of each region in the sorted list SHALL be used as the key to look up the caption from the model's `photo-N` mapping.

#### Scenario: Three regions detected, model assigns all three captions
- **WHEN** Docling detects three regions and the model returns `{"photo-1": "A", "photo-2": "B", "photo-3": "C"}`
- **THEN** the sorted regions receive captions A, B, C respectively in left-to-right/top-to-bottom order

#### Scenario: Model returns fewer captions than detected regions
- **WHEN** the model returns captions for only a subset of the detected photo indices
- **THEN** regions without a matching entry receive an empty caption string; no error is raised

#### Scenario: Model returns more captions than detected regions
- **WHEN** the model returns more `photo-N` keys than there are detected regions
- **THEN** the extra keys are ignored and only the indices that map to a detected region are used

### Requirement: Caption-matching model is configured via ai_models.toml
The system SHALL read the caption-matching model from the `caption_matching_model` key in `ai_models.toml`, resolved against the `[models]` alias table using the same pattern as `view_region_model`. The resolved model name SHALL be passed as the `model` field in the LM Studio chat completions request. `caption_matching_model` SHALL be defined in `ai_models.toml`; if it is absent the system degrades gracefully by writing regions with empty captions and logging a DEBUG message.

#### Scenario: caption_matching_model configured
- **WHEN** `ai_models.toml` defines `caption_matching_model` pointing to a valid model alias
- **THEN** `default_caption_matching_model()` returns the first model name from that alias and caption matching runs as the integrated pipeline step

#### Scenario: caption_matching_model absent
- **WHEN** `ai_models.toml` does not define `caption_matching_model`
- **THEN** `default_caption_matching_model()` returns an empty string, a DEBUG message is logged, and regions are written with empty captions (graceful degradation only — production configurations SHALL define this key)
