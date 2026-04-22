## MODIFIED Requirements

### Requirement: Pipeline step ordering
Steps SHALL execute in this fixed order:
1. `render` — stitch/convert archive scans to page view JPEGs
2. `propagate-metadata` — copy safe archive XMP fields to page sidecar
3. `detect-regions` — detect photo bounding boxes and write MWG-RS XMP regions
4. `crop-regions` — crop detected regions to `_Photos/` directory
5. `face-refresh` — update face region metadata on rendered outputs
6. `ai-index` — run AI pipeline (OCR, caption, shown location, GPS, date, XMP write)
7. `verify-crops` — review each page's crops against the shared page image and page/crop XMP context

#### Scenario: Verification runs after metadata assembly
- **WHEN** `photoalbums.py process --photos-root <root>` runs the full pipeline
- **THEN** `verify-crops` runs after `ai-index`
- **AND** it reviews the finalized crop and page metadata rather than partial upstream state

#### Scenario: Verification runs per page
- **WHEN** `verify-crops` processes outputs for a page
- **THEN** it evaluates all crops derived from that page in one page-context review pass
- **AND** pipeline state records that the page verification pass completed

#### Scenario: Failed concerns rerun before page completion
- **WHEN** `verify-crops` finds a concern marked `bad` or `uncertain` for a crop on the page
- **THEN** the pipeline reruns that specific concern before the page is considered complete
- **AND** the page is not marked done until the concern-specific retry finishes

#### Scenario: Third pass runs before human escalation
- **WHEN** a concern still is not `good` after pass 2
- **THEN** the pipeline runs a final third pass that starts with a fresh full-context parameter-suggestion session
- **AND** it reruns that concern using the suggested params before deciding on escalation

#### Scenario: Third pass failure escalates to human review
- **WHEN** a concern still is not `good` after the final third pass
- **THEN** the pipeline stops retrying that concern automatically
- **AND** it records that concern for human review

### Requirement: Pipeline steps listing
`photoalbums.py process --list-steps` SHALL print the ordered step registry (number, id, description) and exit 0 without processing any files. A `photoalbums-steps` justfile target SHALL invoke it.

#### Scenario: Listing steps includes verification
- **WHEN** user runs `photoalbums.py process --list-steps`
- **THEN** the ordered step list includes `verify-crops`
- **AND** the command exits 0

### Requirement: Step dependency declarations
Each `PipelineStep` in `photoalbums/lib/pipeline.py` SHALL declare `depends_on` step ids. The dependency graph is:
- `propagate-metadata` depends on `render`
- `detect-regions` depends on `render`
- `crop-regions` depends on `detect-regions`
- `face-refresh` depends on `crop-regions`
- `ai-index` depends on `crop-regions`
- `verify-crops` depends on `ai-index`

#### Scenario: Verification reruns after ai-index changes
- **WHEN** `ai-index` reruns and updates crop or page metadata within the same process pass
- **THEN** `verify-crops` is considered stale and reruns because it depends on `ai-index`

### Requirement: Pipeline logging exposes step discoveries
The process pipeline SHALL log each step start and completion in a crisp, human-readable way.
When a step discovers or derives metadata, the pipeline SHALL log the discovered value or conclusion at that step rather than only reporting that the step ran.

#### Scenario: ai-index logs discovered metadata
- **WHEN** `ai-index` derives metadata such as caption, shown location, GPS-backed place, or date
- **THEN** the pipeline log prints the discovered value or a concise summary of it for that item
- **AND** the log does not hide the result behind a generic "ai-index complete" message

#### Scenario: Retry logging exposes failure reasons
- **WHEN** verification triggers an immediate concern-specific retry
- **THEN** the pipeline log prints which concern failed and the concise failure reason
- **AND** the log states that the narrowed retry is running for that specific issue

#### Scenario: Retry logging exposes before and after values
- **WHEN** a concern-specific retry completes
- **THEN** the pipeline log prints the before value, the after value, and whether the value changed
- **AND** the log states whether follow-up verification became `good` or remained unresolved

#### Scenario: Third-pass logging exposes parameter suggestion
- **WHEN** the fresh-session parameter-suggestion step runs for pass 3
- **THEN** the pipeline log prints that a full-context parameter-suggestion session ran
- **AND** the log includes the suggested params that were selected for the final pass

#### Scenario: Verification logs page review outcomes
- **WHEN** `verify-crops` finishes reviewing a page
- **THEN** the pipeline log prints that the page verification pass ran
- **AND** the log includes a concise summary of any concerns marked `bad` or `uncertain`
