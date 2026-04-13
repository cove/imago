## ADDED Requirements

### Requirement: View-region sets are validated before acceptance
The system SHALL validate every candidate view-region set before accepting it as the page's current region list. Validation MUST evaluate the full region set, not each box in isolation, and MUST detect at least heavy overlap between regions.

#### Scenario: Overlapping regions are rejected
- **WHEN** a candidate region set contains two regions whose overlap exceeds the allowed threshold for the smaller region
- **THEN** the system marks the region set invalid
- **AND** records which regions overlap and the reason the set failed validation

#### Scenario: Non-overlapping regions pass overlap validation
- **WHEN** a candidate region set has no region pair exceeding the configured overlap threshold
- **THEN** the system may continue to later validation steps or acceptance

### Requirement: Stored XMP regions are revalidated on every relevant run
The system SHALL validate previously stored XMP view regions on every relevant detect-region or crop-region run because validation is fast enough to apply continuously.

#### Scenario: Stored region list is still valid
- **WHEN** a page already has an XMP region list and that region list passes current validation rules
- **THEN** the system keeps the existing region list without forcing a repair retry

#### Scenario: Stored region list is stale and invalid
- **WHEN** a page already has an XMP region list and that region list fails current validation rules
- **THEN** the system treats the page as needing reprocessing
- **AND** does not trust the stale region list as the accepted page state

### Requirement: Invalid region sets trigger a repair retry with explicit feedback
When a candidate region set fails validation, the system SHALL retry region detection with explicit feedback that includes the prior region set and the validation errors that must be fixed. The retry MUST request a complete revised region set for the same image. The retry response MAY include an `error_analysis` string describing what the model believes was wrong with the prior region set.

#### Scenario: Retry prompt includes previous boxes and overlap failures
- **WHEN** validation fails because regions overlap too much
- **THEN** the retry request includes the previous full region set
- **AND** identifies the invalid regions and the overlap failure that must be corrected

#### Scenario: Retry response includes model error analysis
- **WHEN** the repair retry returns both a revised region set and an `error_analysis` value
- **THEN** the system records the `error_analysis` value in debug or diagnostic output for that retry
- **AND** still decides acceptance based on local validation of the returned region set

#### Scenario: Retry still fails validation
- **WHEN** the repaired region set still fails validation after the allowed retries
- **THEN** the system leaves the page unresolved rather than accepting the invalid region set
