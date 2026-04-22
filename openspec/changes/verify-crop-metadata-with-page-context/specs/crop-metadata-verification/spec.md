## ADDED Requirements

### Requirement: Verification runs per page with the crop as the target and the page as context
The system SHALL provide a crop-metadata verification workflow that runs once per page under review and evaluates each crop from that page using the following inputs:
- the original album page image
- the page XMP metadata rendered as text
- each extracted crop image from that page
- each crop XMP metadata rendered as text

The verifier SHALL evaluate the final assembled crop result in context rather than re-running production metadata generation.
The verifier SHALL treat the crop image as the primary subject being reviewed and the page image as supporting context for interpreting captions, dates, and locations.

#### Scenario: Verification receives full page-and-crop context
- **WHEN** the verifier runs for a page
- **THEN** it receives the page image and page XMP text once for that page
- **AND** it receives each crop image and crop XMP text from that page as review items
- **AND** the prompt makes clear that the crop is the item under review while the page is supporting context

#### Scenario: Verification does not run from partial metadata alone
- **WHEN** crop inputs exist for a page but the page image or page XMP is missing
- **THEN** the verifier does not claim a full page-context review result
- **AND** the run reports the missing review context explicitly

### Requirement: Verification records that the page pass ran and preserves latest concern states
The system SHALL record that the verification pass ran for the page.
The system SHALL preserve the latest `good`, `bad`, or `uncertain` state for each concern on each reviewed crop so later targeted reruns can use the current review state without recomputing it from logs alone.
The system SHALL store the specific failure reason for each concern marked `bad` or `uncertain`.
The system SHALL record provenance for the accepted concern state, including the prompt variant, model, tuning params, and retry count that produced it.

#### Scenario: Verification records page pass completion
- **WHEN** verification finishes for a page
- **THEN** pipeline state records that the verification pass ran for that page
- **AND** the latest concern states for each reviewed crop remain available for later rerun decisions

#### Scenario: Verification stores failure reasons
- **WHEN** a reviewed concern is marked `bad` or `uncertain`
- **THEN** pipeline state stores the specific reason the verifier gave for that failed concern
- **AND** that reason remains available to drive the immediate retry and later review

#### Scenario: Accepted concern state records provenance
- **WHEN** a concern result is accepted as the current state
- **THEN** pipeline state records which prompt variant, model, tuning params, and retry count produced that accepted result
- **AND** that provenance remains available for debugging and later retries

### Requirement: Verification returns structured judgments and follow-up routing
The verifier SHALL return structured judgments for:
- `caption`
- `gps`
- `shown_location`
- `date`
- `overall`

Each judgment SHALL contain:
- a verdict: `good`, `bad`, or `uncertain`
- one sentence of reasoning

The verifier SHALL also return:
- `human_inference`, describing what a person would actually read or infer from the page whenever any judgment is `bad` or `uncertain`
- `needs_another_pass`, listing specific concerns that should be retried with another targeted AI pass
- `needs_human_review`, listing specific concerns that should be escalated for manual review
- `failure_reason`, or an equivalent concern-level reason field, for each concern marked `bad` or `uncertain`

The `failure_reason` SHALL be articulate and specific, and SHALL include a rationale explaining why the current concern result does not match what a human would infer from the page.

#### Scenario: All review dimensions return explicit verdicts
- **WHEN** verification completes successfully for a crop
- **THEN** the result includes `caption`, `gps`, `shown_location`, `date`, and `overall`
- **AND** each of those entries contains a verdict from `good`, `bad`, or `uncertain`
- **AND** each entry contains one sentence of reasoning

#### Scenario: Human inference is included when a mismatch is found
- **WHEN** any review dimension returns `bad` or `uncertain`
- **THEN** the verifier includes `human_inference` describing what a person would actually read or infer from the page instead
- **AND** the verifier includes the specific failure reason for the failed concern

#### Scenario: Follow-up routing uses readable concern names
- **WHEN** the verifier determines that additional review work is needed
- **THEN** `needs_another_pass` and `needs_human_review` use explicit concern names such as `caption`, `gps`, `shown_location`, and `date`
- **AND** the output does not rely on abbreviated flags that require interpretation

#### Scenario: Follow-up routing remains concern-specific
- **WHEN** a concern is currently `good`
- **THEN** the verifier does not request another pass for that concern
- **AND** the routing does not authorize rewriting unrelated fields that are already `good`

### Requirement: Failed concerns are retried immediately before page completion
When verification marks a concern as `bad` or `uncertain`, the system SHALL rerun that specific concern before the page is considered complete.
The immediate retry SHALL use the same concern flow, augmented with the verification failure reason as retry guidance.
The immediate retry prompt SHALL narrow its focus to the failed concern rather than reopening unrelated metadata concerns.
The immediate retry prompt SHALL be trimmed to: base concern prompt, concern-specific issue, and problem-to-fix summary.
The second pass SHALL use that narrowed retry contract.
If the concern still is not `good` after pass 2, the system SHALL perform one final third pass that first asks the verification model to suggest better tuning params in a fresh full-context session that resends the relevant images and metadata.
The third pass SHALL then rerun the failed concern using those suggested tuning params.
The system SHALL escalate that concern to human review if it still is not `good` after the third pass.

#### Scenario: Failed concern triggers immediate retry
- **WHEN** verification marks `caption`, `gps`, `shown_location`, or `date` as `bad` or `uncertain`
- **THEN** the system reruns only that concern before the page is marked complete
- **AND** the retry uses the stored failure reason as part of its prompt context

#### Scenario: Immediate retry stays scoped to the failed issue
- **WHEN** an immediate retry runs for one concern
- **THEN** the retry prompt focuses on that failed concern and the reason it failed
- **AND** the retry does not rewrite unrelated fields that were already `good`

#### Scenario: Third pass requests better tuning params in a fresh session
- **WHEN** a concern still is not `good` after pass 2
- **THEN** the system opens a fresh session for that concern
- **AND** it resends the relevant page image, crop image, page XMP text, and crop XMP text
- **AND** it asks the verification model to suggest better tuning params for the failed concern

#### Scenario: Third pass uses suggested tuning params
- **WHEN** the fresh-session parameter suggestion completes
- **THEN** the system reruns the failed concern using the suggested tuning params
- **AND** the selected tuning params are recorded in provenance for that retry

#### Scenario: Changed retry result is re-verified
- **WHEN** a retry changes the value for a concern whose last verification state was not `good`
- **THEN** the system reruns verification for that concern before accepting the changed result

#### Scenario: Third pass failure leads to human escalation
- **WHEN** a concern still is not `good` after the final third pass
- **THEN** that concern is added to `needs_human_review`
- **AND** the page does not keep retrying that concern automatically

### Requirement: Caption review uses page layout and visual plausibility
The `caption` judgment SHALL evaluate whether the crop's `dc:description` matches the caption that visually belongs to that photo on the page, using proximity, layout, whether the caption plausibly describes the crop image, and whether neighboring captions need to be combined to preserve the meaning a human would infer.

#### Scenario: Caption belongs to the cropped photo
- **WHEN** the crop description matches the caption visually adjacent to that photo on the page
- **AND** the caption plausibly describes what is shown in the crop
- **THEN** the `caption` verdict is `good`

#### Scenario: Caption belongs to a neighboring photo instead
- **WHEN** the crop description matches text that visually belongs to a neighboring photo rather than the reviewed crop
- **THEN** the `caption` verdict is `bad`
- **AND** the reasoning states that the caption appears to belong to a different photo on the page

#### Scenario: Caption meaning depends on nearby context
- **WHEN** the reviewed crop needs a nearby earlier caption to make a later caption meaningful, such as a named person followed by a later photo described as "their new condominium"
- **THEN** the `caption` review considers the combined page reading a human would infer
- **AND** the verifier may mark `caption` as `bad` or `uncertain` when the crop description drops that necessary context

### Requirement: Shown-location review uses written text, landmarks, and adjacent page context
The `shown_location` judgment SHALL evaluate whether the crop's human-readable location metadata matches what the page suggests from written text, recognizable landmarks, or context from adjacent captions.

#### Scenario: Shown location matches page context
- **WHEN** the crop shown-location fields agree with the place named or implied by the page
- **THEN** the `shown_location` verdict is `good`

#### Scenario: Shown location conflicts with page context
- **WHEN** the crop shown-location fields name a different place than the one suggested by the page text or visual context
- **THEN** the `shown_location` verdict is `bad`

### Requirement: GPS review checks whether the resolved place is supported by the page context
The `gps` judgment SHALL evaluate whether the crop's geocoded place appears to be grounded in the page context strongly enough to trust the GPS assignment.

#### Scenario: GPS-backed place is supported by the page context
- **WHEN** the page context identifies the place specifically enough that the resolved location appears consistent with the crop metadata
- **THEN** the `gps` verdict is `good`

#### Scenario: GPS-backed place appears to be resolved from underspecified context
- **WHEN** the crop GPS-backed place appears to come from a partial or ambiguous page reference that could plausibly resolve to a different place
- **THEN** the `gps` verdict is `bad` or `uncertain`
- **AND** the reasoning explains why the page context does not support trusting the resolved place

### Requirement: Date review uses written or implied page context
The `date` judgment SHALL evaluate whether the crop date metadata matches what is written or implied on the page.

#### Scenario: Date matches page context
- **WHEN** the crop date agrees with the date written or implied on the page
- **THEN** the `date` verdict is `good`

#### Scenario: Date review preserves month-level evidence when available
- **WHEN** the page text supports a month-level date such as `AUG. 1988`
- **THEN** the verifier treats the month and year as belonging together when a human would naturally read them together
- **AND** the verifier expects `1988-08` to be better-supported than `1988`
- **AND** it may mark `date` as `bad` or `uncertain` when month evidence was lost

#### Scenario: Date cannot be supported from page context
- **WHEN** the page does not provide enough evidence to confirm the crop date confidently
- **THEN** the `date` verdict is `uncertain`

### Requirement: Overall review reflects whether a human would accept the crop and metadata as belonging together
The `overall` judgment SHALL answer whether a person looking at the page would agree that the crop, caption, shown location, GPS, and date go together.
The `overall` judgment SHALL be summary-only and SHALL NOT by itself require another targeted AI pass without a specific concern being named in `needs_another_pass` or `needs_human_review`.

#### Scenario: Human would accept the crop and metadata as a matching set
- **WHEN** the crop, caption, shown location, GPS, date, and page context are mutually consistent
- **THEN** the `overall` verdict is `good`

#### Scenario: Human would not accept the crop and metadata as a matching set
- **WHEN** one or more review dimensions show that the crop and metadata do not belong together
- **THEN** the `overall` verdict is `bad` or `uncertain`
- **AND** the reasoning explains the mismatch in one sentence
