## 1. Verification Input Assembly

- [ ] 1.1 Add page-level verifier input loading for the page image
- [ ] 1.2 Add page-level verifier input loading for page XMP text
- [ ] 1.3 Add per-crop verifier input loading for crop images on the page
- [ ] 1.4 Add per-crop verifier input loading for crop XMP text on the page
- [ ] 1.5 Assemble the page image, page XMP text, crop image, and crop XMP text into the same verifier request payload for each reviewed crop
- [ ] 1.6 Report missing page image context explicitly when it is unavailable
- [ ] 1.7 Report missing page XMP context explicitly when it is unavailable
- [ ] 1.8 Define the base verifier response schema fields for `caption`, `gps`, `shown_location`, `date`, and `overall`
- [ ] 1.9 Define the verifier response schema fields for `human_inference`, `needs_another_pass`, and `needs_human_review`
- [ ] 1.10 Define the verifier response schema field for concern-level `failure_reason`
- [ ] 1.11 Define the pass-2 trimmed retry prompt contract as base prompt plus concern-specific issue plus problem-to-fix summary
- [ ] 1.12 Define the pass-3 parameter-suggestion request that resends full page and crop context in a fresh session

## 2. Verification Execution And Output

- [ ] 2.1 Add the page-level `verify-crops` runner entrypoint
- [ ] 2.2 Run the verifier once per page with shared page context and per-crop review items
- [ ] 2.3 Parse `good` / `bad` / `uncertain` verdicts from verifier output
- [ ] 2.4 Parse one-sentence concern reasoning from verifier output
- [ ] 2.5 Persist review artifacts for the raw verifier result
- [ ] 2.6 Mirror concern states into `imago:Detections["pipeline"]`
- [ ] 2.7 Mirror explicit concern names such as `caption`, `gps`, `shown_location`, and `date` into rerun routing
- [ ] 2.8 Record that the verification pass ran for the page
- [ ] 2.9 Preserve the latest concern state for each reviewed crop
- [ ] 2.10 Persist the specific verification failure reason for each concern marked `bad` or `uncertain`
- [ ] 2.11 Prevent rerun routing from rewriting unrelated concerns already marked `good`
- [ ] 2.12 Trigger pass 2 for a failed concern before the page completes
- [ ] 2.13 Build pass-2 retry prompts from failure reason plus trimmed retry contract
- [ ] 2.14 Run pass 2 through the existing concern flow
- [ ] 2.15 Open a fresh session for pass-3 parameter suggestion when pass 2 is still not `good`
- [ ] 2.16 Resend page image, crop image, page XMP text, and crop XMP text in the pass-3 parameter-suggestion session
- [ ] 2.17 Parse suggested tuning params from the pass-3 parameter-suggestion response
- [ ] 2.18 Rerun the failed concern with the pass-3 suggested tuning params
- [ ] 2.19 Record provenance for the accepted result, including prompt variant, model, tuning params, and retry count
- [ ] 2.20 Rerun verification when a retry changes a concern value and the last concern status was not `good`
- [ ] 2.21 Escalate the concern to human review when it still is not `good` after pass 3

## 3. Pipeline Integration

- [ ] 3.1 Add `verify-crops` to the ordered pipeline step registry
- [ ] 3.2 Add `verify-crops` to `--list-steps`
- [ ] 3.3 Add `verify-crops` to `--step`
- [ ] 3.4 Declare `verify-crops` dependency on `ai-index`
- [ ] 3.5 Mark `verify-crops` stale when `ai-index` reruns
- [ ] 3.6 Record pipeline completion state for page-level verification
- [ ] 3.7 Record targeted follow-up requests for failed concerns
- [ ] 3.8 Log step start and completion for `verify-crops`
- [ ] 3.9 Log discovered metadata summaries from `ai-index`
- [ ] 3.10 Prevent a page from being marked complete until pass 2 or pass 3 work is finished
- [ ] 3.11 Log concern failure reason before pass 2 starts
- [ ] 3.12 Log before/after values for concern retries
- [ ] 3.13 Log whether follow-up verification became `good` or remained unresolved
- [ ] 3.14 Log when the pass-3 parameter-suggestion session runs
- [ ] 3.15 Log the params selected for the final third pass

## 4. Regression Coverage

- [ ] 4.1 Add parsing tests for `good` verdicts
- [ ] 4.2 Add parsing tests for `bad` verdicts
- [ ] 4.3 Add parsing tests for `uncertain` verdicts
- [ ] 4.4 Add fixture coverage for obviously correct page/crop pairs
- [ ] 4.5 Add fixture coverage for obviously wrong page/crop pairs
- [ ] 4.6 Add fixture coverage for ambiguous page/crop pairs
- [ ] 4.7 Add integration coverage that `verify-crops` runs after `ai-index`
- [ ] 4.8 Add integration coverage that `verify-crops` executes per page
- [ ] 4.9 Add integration coverage that `verify-crops` receives finalized page/crop metadata
- [ ] 4.10 Add regression coverage for `human_inference` when review is bad or uncertain
- [ ] 4.11 Add regression coverage for caption carry-over reasoning across neighboring captions
- [ ] 4.12 Add regression coverage for month-plus-year date evidence producing `1988-08`
- [ ] 4.13 Add regression coverage proving a failed concern stores a failure reason
- [ ] 4.14 Add regression coverage proving a failed concern triggers pass 2 before page completion
- [ ] 4.15 Add regression coverage for accepted-result provenance recording
- [ ] 4.16 Add regression coverage for pass-3 parameter suggestion with fresh full context
- [ ] 4.17 Add regression coverage for third-pass escalation to human review
- [ ] 4.18 Add regression coverage for before/after retry logging
- [ ] 4.19 Add regression coverage for no-change loop visibility in retry logging
