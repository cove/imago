## Why

The production pipeline can assemble crop metadata from multiple specialized systems, but it still lacks the final kind of check a human naturally performs when flipping through a family photo album: does this crop, caption, shown location, GPS, date, and page context actually go together? We need a local verification step that runs on a per-page basis, reviews each extracted crop as the primary subject, uses the original page as supporting context, and inspects both XMP sidecars together to decide whether a targeted follow-up pass is needed.

## What Changes

- Add a new crop-metadata verification capability that runs once per page, reviews each crop on that page against the shared page context, and returns structured `good` / `bad` / `uncertain` assessments for `caption`, `gps`, `shown_location`, `date`, and `overall`.
- Define a normative review output schema that records readable follow-up routing such as `needs_another_pass` and `needs_human_review` in `imago:Detections["pipeline"]`, along with the fact that the verification pass ran, the latest concern states, specific failure reasons when a concern is bad or uncertain, and provenance for the prompt/model/tuning settings that produced the current result.
- Require the verifier to explain what a human would actually infer from the page whenever a result is bad or uncertain, especially for caption-to-photo reasoning and location disambiguation.
- Make the verifier runnable locally as a per-page pipeline step or standalone command after crop and metadata generation are complete.
- When verification finds a failed concern, rerun that specific concern before the page is considered complete. Use a trimmed retry contract of base prompt plus concern-specific issue plus problem-to-fix for the second pass, then allow a final third pass where the verification model suggests better params in a fresh full-context session that resends the images and metadata before the concern runs again.
- Escalate to human review after that final third pass if the concern still is not `good`.
- Tighten the process specification so pipeline logs clearly show what each step ran and what each metadata-producing step discovered.

## Capabilities

### New Capabilities
- `crop-metadata-verification`: Defines the human-judgment verification workflow, required inputs, review criteria, and structured output for checking page/crop consistency.

### Modified Capabilities
- `unified-process-pipeline`: Add a locally runnable verification step so crop-metadata review can be executed as part of the process pipeline or in isolation.

## Impact

- `photoalbums.py process` / pipeline step registry — add a verification step after crop and metadata assembly, with page-level execution semantics.
- Local verification command path — load the page image and page XMP once per page, then review each crop image and crop XMP from that page against the shared context.
- Prompt/schema code for the local verifier — return structured review judgments, explicit articulate failure reasons with rationale, and readable rerun/human-review routing rather than free-form notes.
- Concern-specific retry flow — rerun only the failed concern before the page completes, using the failure reason to focus a trimmed retry prompt on the second pass, and using a fresh full-context parameter-suggestion session before the final third pass.
- Detection pipeline state — persist that verification ran, the latest `good` / `bad` / `uncertain` concern states, the failure reason for bad or uncertain concerns, provenance for the active result, and which concerns need another targeted AI pass such as `caption`, `gps`, `shown_location`, or `date`.
- Pipeline logging — emit a crisp step-by-step record, including what metadata-producing steps discovered such as location/date outputs, what happened on pass 2, when the parameter-suggestion session ran, and the before/after value changes across retries.
- Review/debug artifacts — persist verification results so bad or uncertain crops can be inspected and rerun deliberately.
