## Why

The current region detector can write overlapping or separator-crossing boxes that cut photos in half or include album paper, captions, and neighboring photos. Prompt-only tuning helped clarify the failure mode, but did not reliably prevent bad crops, so we need a validation-and-repair loop before regions are accepted as ground truth.

## What Changes

- Add a fast local validation pass for detected view-photo regions on every run, including overlap checks and suspicious-box checks before regions are accepted
- Retry region detection with explicit feedback when the returned boxes are invalid, including the previous boxes and the validation errors that must be fixed, and capture an optional model-provided `error_analysis` summary for debugging
- Reprocess previously saved page regions when stored XMP regions fail current validation rules, even if the page was already processed
- Prevent invalid regions from being written as the accepted XMP `RegionList` or used for crop generation
- Record validation and retry context in prompt debug artifacts so bad-region decisions are auditable

## Capabilities

### New Capabilities

- `view-region-validation`: Validate detected view regions, identify suspicious or overlapping boxes, and drive repair retries before regions are accepted

### Modified Capabilities

- `view-xmp-regions`: Accepted XMP region lists must reflect current validation rules, and pages with stale invalid regions must be reprocessed instead of trusted

## Impact

- Affected code: `photoalbums/lib/ai_view_regions.py`, `photoalbums/commands.py`, `photoalbums/lib/ai_photo_crops.py`, prompt-debug artifact generation, and related tests
- Affected data: `_View/*.xmp` region lists may be replaced when previously accepted boxes now fail validation
- Runtime impact: a lightweight local validation pass runs on every detection/crop pass; failed pages may trigger additional model retries
- No new external service dependency is required; validation is local and retry calls continue to use the existing LM Studio model
