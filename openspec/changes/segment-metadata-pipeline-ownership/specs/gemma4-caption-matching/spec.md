## MODIFIED Requirements

### Requirement: LM Studio model assigns captions to accepted regions using numbered overlay prompt
The system SHALL send the numbered region-association overlay image to the configured LM Studio model using a prompt that instructs it to use the visible region numbers directly and return a JSON object mapping region numbers to captions:

```
`{"region-1": "", "region-2": "", "region-3": ""}`
- Number as many `region-N` keys as there are visible numbered regions on the overlay.
- Each value is the caption text that belongs to that region; empty string if there is no caption.
- Use the visible overlay numbers as authoritative. Do not renumber the regions and do not infer a separate left-to-right/top-to-bottom ordering.
- If a caption refers to subjects shown in an adjacent photo, prepend the missing subject so the caption reads standalone. Do not rewrite or summarise; only prepend the minimum context needed.
- Just return the JSON without any extra text or explanation.
```

The response SHALL be parsed as JSON and returned as a mapping of 1-based region index to caption string. The model used is selected by the `caption_matching_model` key in `ai_models.toml`.

#### Scenario: Model returns valid JSON caption mapping
- **WHEN** the LM Studio model responds with a JSON object keyed `region-1`, `region-2`, etc.
- **THEN** the system parses the JSON and returns a dict mapping integer region index to caption string for each key present

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

### Requirement: Caption assignment merges LM Studio output with accepted regions via direct overlay-number mapping
The system SHALL map the LM Studio caption response directly to the accepted region whose visible overlay number matches the response key. The system SHALL NOT sort accepted regions into a separate coordinate-based reading order for caption assignment.

#### Scenario: Three accepted regions, model assigns all three captions
- **WHEN** the overlay shows three accepted regions and the model returns `{"region-1": "A", "region-2": "B", "region-3": "C"}`
- **THEN** the accepted regions with visible numbers 1, 2, and 3 receive captions A, B, and C respectively

#### Scenario: Model returns fewer captions than accepted regions
- **WHEN** the model returns captions for only a subset of the visible region numbers
- **THEN** accepted regions without a matching entry receive an empty caption string
- **AND** no error is raised

#### Scenario: Model returns more captions than accepted regions
- **WHEN** the model returns more `region-N` keys than there are accepted regions on the overlay
- **THEN** the extra keys are ignored
- **AND** only entries that map to an accepted region are used
