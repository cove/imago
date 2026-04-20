## MODIFIED Requirements

### Requirement: LM Studio model assigns captions to photos using numbered-photo prompt
The system SHALL send the album page image to the configured LM Studio model. When the page has fewer than two `LocationShown` entries, the prompt instructs the model to return a JSON object mapping photo numbers to caption strings (existing behavior):

```
`{"photo-1": "", "photo-2": "", "photo-3": ""}`
- Number as many `photo-N` keys as there are distinct photos on the page.
- Each value is the caption text that belongs to that photo; empty string if there is no caption.
- If a caption refers to subjects shown in an adjacent photo, prepend the missing subject so the caption reads standalone. Do not rewrite or summarise; only prepend the minimum context needed.
- Just return the JSON without any extra text or explanation.
```

When the page has two or more `LocationShown` entries, the prompt SHALL additionally instruct the model to return a location name per photo:

```
`{"photo-1": {"caption": "", "location": ""}, "photo-2": {"caption": "", "location": ""}}`
- Number as many `photo-N` keys as there are distinct photos on the page.
- "caption": the caption text for that photo; empty string if there is no caption.
- "location": the place name for that photo (e.g. "Cairo, Egypt"); use one of the known locations listed below if it matches, or empty string if unknown.
Known locations: <comma-separated list of LocationShown names from archive XMP>
- If a caption refers to subjects shown in an adjacent photo, prepend the missing subject so the caption reads standalone.
- Just return the JSON without any extra text or explanation.
```

The response SHALL be parsed as JSON. The system SHALL handle both the string-value format (single-location pages) and the object-value format (multi-location pages), returning a mapping of 1-based photo index to `{"caption": str, "location": str}`. The model used is selected by the `caption_matching_model` key in `ai_models.toml`.

#### Scenario: Single-location page — model returns string-value JSON
- **WHEN** the page has one or zero `LocationShown` entries and the model responds with `{"photo-1": "Caption A", "photo-2": "Caption B"}`
- **THEN** the system parses the JSON and returns captions for each photo; location is empty for all

#### Scenario: Multi-location page — model returns object-value JSON with location
- **WHEN** the page has two or more `LocationShown` entries and the model returns `{"photo-1": {"caption": "Cairo street scene", "location": "Cairo, Egypt"}, "photo-2": {"caption": "Karnak Temple", "location": "Luxor, Egypt"}}`
- **THEN** the system parses the JSON and returns caption and location for each photo

#### Scenario: Multi-location page — model returns empty location for a photo
- **WHEN** the page has multiple `LocationShown` entries and the model returns `"location": ""` for a specific photo
- **THEN** that photo's location is treated as unassigned; the crop falls back to page-level location inheritance

#### Scenario: Model response contains extra text around JSON
- **WHEN** the model wraps the JSON in markdown code fences or leading/trailing prose
- **THEN** the system extracts the JSON object via regex and parses it, ignoring surrounding text

#### Scenario: Model returns malformed JSON
- **WHEN** the LM Studio response cannot be parsed as valid JSON after extraction
- **THEN** the system logs a WARNING with the raw response and returns an empty mapping; region detection continues without captions or locations

#### Scenario: LM Studio is offline
- **WHEN** the LM Studio endpoint is unreachable when the caption-matching step runs
- **THEN** the system logs a WARNING and returns an empty mapping; region detection continues without captions or locations

#### Scenario: Caption refers to subject shown in adjacent photo
- **WHEN** the model detects that a caption refers to a subject established in an adjacent photo
- **THEN** the model prepends the subject so the caption reads standalone
