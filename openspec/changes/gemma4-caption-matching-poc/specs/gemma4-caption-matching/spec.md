## ADDED Requirements

### Requirement: Gemma4 assigns captions to photos using numbered-photo prompt
The system SHALL send the album page image to Gemma4 via LM Studio using the prompt: "Number the photos from left to right and top to bottom first; then determine which caption goes with which photos and output the number and the caption you think goes with it, outputting the result in JSON: {"photo-1": "caption", "photo-2": "caption" ...}". The response SHALL be parsed as JSON and returned as a mapping of 1-based photo index to caption string.

#### Scenario: Gemma4 returns valid JSON caption mapping
- **WHEN** Gemma4 responds with a JSON object keyed `photo-1`, `photo-2`, etc.
- **THEN** the system parses the JSON and returns a dict mapping integer index to caption string for each key present

#### Scenario: Gemma4 response contains extra text around JSON
- **WHEN** Gemma4 wraps the JSON in markdown code fences or leading/trailing prose
- **THEN** the system extracts the JSON object via regex and parses it, ignoring surrounding text

#### Scenario: Gemma4 returns malformed JSON
- **WHEN** the LM Studio response cannot be parsed as valid JSON after extraction
- **THEN** the system logs a WARNING with the raw response and returns an empty mapping; region detection continues without captions

#### Scenario: LM Studio is offline
- **WHEN** the LM Studio endpoint is unreachable when the caption-matching step runs
- **THEN** the system logs a WARNING and returns an empty mapping; region detection continues without captions

### Requirement: Caption assignment merges Gemma4 output with Docling bounding boxes via coordinate-based sort
The system SHALL sort the Docling-detected `RegionResult` list using a strict coordinate-based scanline sort before assigning Gemma4 captions. Neither Docling nor Heron outputs boxes in reading order; boxes MUST always be sorted explicitly. The sort SHALL group regions into rows using a y-tolerance band expressed as a ratio (0.0–1.0) of image height (configurable, default 0.10), then sort rows top-to-bottom and regions within each row left-to-right by x coordinate. The 1-based position of each region in the sorted list SHALL be used as the key to look up the caption from Gemma4's `photo-N` mapping.

#### Scenario: Three regions detected, Gemma4 assigns all three captions
- **WHEN** Docling detects three regions and Gemma4 returns `{"photo-1": "A", "photo-2": "B", "photo-3": "C"}`
- **THEN** the sorted regions receive captions A, B, C respectively in left-to-right/top-to-bottom order

#### Scenario: Gemma4 returns fewer captions than detected regions
- **WHEN** Gemma4 returns captions for only a subset of the detected photo indices
- **THEN** regions without a matching Gemma4 entry receive an empty caption string; no error is raised

#### Scenario: Gemma4 returns more captions than detected regions
- **WHEN** Gemma4 returns more `photo-N` keys than there are detected regions
- **THEN** the extra keys are ignored and only the indices that map to a detected region are used
