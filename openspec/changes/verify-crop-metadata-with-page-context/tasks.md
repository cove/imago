## 1. Verification Input Assembly

- [x] 1.1 Add page-level verifier input loading for the page image
- [x] 1.2 Add page-level verifier input loading for page XMP text
- [x] 1.3 Add per-crop verifier input loading for crop images on the page
- [x] 1.4 Add per-crop verifier input loading for crop XMP text on the page
- [x] 1.5 Assemble the page image, page XMP text, crop image, and crop XMP text into the same verifier request payload for each reviewed crop
- [x] 1.6 Report missing page image context explicitly when it is unavailable
- [x] 1.7 Report missing page XMP context explicitly when it is unavailable
- [x] 1.8 Define the base verifier response schema fields for `caption`, `gps`, `shown_location`, `date`, and `overall`
- [x] 1.9 Define the verifier response schema fields for `human_inference`, `needs_another_pass`, and `needs_human_review`
- [x] 1.10 Define the verifier response schema field for concern-level `failure_reason`
- [x] 1.11 Define the pass-2 trimmed retry prompt contract as base prompt plus concern-specific issue plus problem-to-fix summary
- [x] 1.12 Define the pass-3 parameter-suggestion request that resends full page and crop context in a fresh session

## 2. Verification Execution And Output

- [x] 2.1 Add the page-level `verify-crops` runner entrypoint
- [x] 2.2 Run the verifier once per page with shared page context and per-crop review items
- [x] 2.3 Parse `good` / `bad` / `uncertain` verdicts from verifier output
- [x] 2.4 Parse one-sentence concern reasoning from verifier output
- [x] 2.5 Persist review artifacts for the raw verifier result
- [x] 2.6 Mirror concern states into `imago:Detections["pipeline"]`
- [x] 2.7 Mirror explicit concern names such as `caption`, `gps`, `shown_location`, and `date` into rerun routing
- [x] 2.8 Record that the verification pass ran for the page
- [x] 2.9 Preserve the latest concern state for each reviewed crop
- [x] 2.10 Persist the specific verification failure reason for each concern marked `bad` or `uncertain`
- [x] 2.11 Prevent rerun routing from rewriting unrelated concerns already marked `good`
- [x] 2.12 Trigger pass 2 for a failed concern before the page completes
- [x] 2.13 Build pass-2 retry prompts from failure reason plus trimmed retry contract
- [x] 2.14 Run pass 2 through the existing concern flow
- [x] 2.15 Open a fresh session for pass-3 parameter suggestion when pass 2 is still not `good`
- [x] 2.16 Resend page image, crop image, page XMP text, and crop XMP text in the pass-3 parameter-suggestion session
- [x] 2.17 Parse suggested tuning params from the pass-3 parameter-suggestion response
- [x] 2.18 Rerun the failed concern with the pass-3 suggested tuning params
- [x] 2.19 Record provenance for the accepted result, including prompt variant, model, tuning params, and retry count
- [x] 2.20 Rerun verification when a retry changes a concern value and the last concern status was not `good`
- [x] 2.21 Escalate the concern to human review when it still is not `good` after pass 3

## 3. Pipeline Integration

- [x] 3.1 Add `verify-crops` to the ordered pipeline step registry
- [x] 3.2 Add `verify-crops` to `--list-steps`
- [x] 3.3 Add `verify-crops` to `--step`
- [x] 3.4 Declare `verify-crops` dependency on `ai-index`
- [x] 3.5 Mark `verify-crops` stale when `ai-index` reruns
- [x] 3.6 Record pipeline completion state for page-level verification
- [x] 3.7 Record targeted follow-up requests for failed concerns
- [x] 3.8 Log step start and completion for `verify-crops`
- [x] 3.9 Log discovered metadata summaries from `ai-index`
- [x] 3.10 Prevent a page from being marked complete until pass 2 or pass 3 work is finished
- [x] 3.11 Log concern failure reason before pass 2 starts
- [x] 3.12 Log before/after values for concern retries
- [x] 3.13 Log whether follow-up verification became `good` or remained unresolved
- [x] 3.14 Log when the pass-3 parameter-suggestion session runs
- [x] 3.15 Log the params selected for the final third pass

## 4. Regression Coverage

- [x] 4.1 Add parsing tests for `good` verdicts
- [x] 4.2 Add parsing tests for `bad` verdicts
- [x] 4.3 Add parsing tests for `uncertain` verdicts
- [x] 4.4 Add fixture coverage for obviously correct page/crop pairs
- [x] 4.5 Add fixture coverage for obviously wrong page/crop pairs
- [x] 4.6 Add fixture coverage for ambiguous page/crop pairs
- [x] 4.7 Add integration coverage that `verify-crops` runs after `ai-index`
- [x] 4.8 Add integration coverage that `verify-crops` executes per page
- [x] 4.9 Add integration coverage that `verify-crops` receives finalized page/crop metadata
- [x] 4.10 Add regression coverage for `human_inference` when review is bad or uncertain
- [x] 4.11 Add regression coverage for caption carry-over reasoning across neighboring captions
- [x] 4.12 Add regression coverage for month-plus-year date evidence producing `1988-08`
- [x] 4.13 Add regression coverage proving a failed concern stores a failure reason
- [x] 4.14 Add regression coverage proving a failed concern triggers pass 2 before page completion
- [x] 4.15 Add regression coverage for accepted-result provenance recording
- [x] 4.16 Add regression coverage for pass-3 parameter suggestion with fresh full context
- [x] 4.17 Add regression coverage for third-pass escalation to human review
- [x] 4.18 Add regression coverage for before/after retry logging
- [x] 4.19 Add regression coverage for no-change loop visibility in retry logging
