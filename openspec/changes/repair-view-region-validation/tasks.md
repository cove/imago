## 1. Validation Rules

- [x] 1.1 Define the region-set validation API and failure payloads for hard-invalid versus suspicious boxes
- [x] 1.2 Add overlap validation that evaluates all regions in a set and returns concrete per-region and pairwise failure reasons
- [x] 1.3 Add validation coverage for stored XMP regions so previously accepted pages are reprocessed when current rules fail

## 2. Repair Retry Loop

- [x] 2.1 Extend the view-region retry prompt to include the previous full region set and explicit validation errors
- [x] 2.2 Update the detection retry loop to request a complete revised region set after validation failure, with optional `error_analysis` in the structured response
- [x] 2.3 Ensure only region sets that pass validation are written to XMP and used by downstream crop generation

## 3. Pipeline Integration

- [x] 3.1 Update detect-view-regions and crop-regions flows to validate all relevant region sets on every run
- [x] 3.2 Reprocess pages whose stored regions fail validation instead of trusting the existing XMP cache
- [x] 3.3 Record validation outcomes, retry context, and any returned `error_analysis` in debug artifacts and logs

## 4. Verification

- [x] 4.1 Add tests for overlap detection and reprocessing of stale invalid XMP region lists
- [x] 4.2 Add tests for retry prompts that include prior boxes plus validation errors, and for optional `error_analysis` capture
- [ ] 4.3 Run focused validation with `just test`, `just dupes`, `just deadcode`, and `just complexity`
