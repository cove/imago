## Context

The production pipeline now has clearer ownership boundaries for producing crop metadata, but it still lacks the last kind of check a person naturally performs while reading a family album page: does this crop, caption, shown location, GPS, date, and page context actually belong together? That judgment is not identical to any existing production step. It is not caption assignment, not face identification, and not geocoding. It is an adjudication step over already-produced outputs.

The user’s target workflow is local and human-like. The verifier should review:
- the extracted crop image
- the original page image
- the page XMP text
- the crop XMP text

and then answer the same questions a person would ask:
- Is this cropped photo the thing I am evaluating?
- What does the full page tell me that the crop alone does not?
- Does the caption belong to this photo?
- Does the shown location fit the page context?
- Does the GPS-backed place look correctly grounded in the page context?
- Does the date fit the page context?
- Overall, do these things go together?

This change therefore defines a local review pipeline that runs after crop generation and metadata assembly. The unit of work is a page: the verifier sees the shared page context once and evaluates every crop on that page against it. It does not replace or directly write production metadata. It evaluates the results and reports structured judgments that can be inspected, persisted into pipeline state, and used to decide whether a targeted follow-up AI pass or a human review is needed.

## Goals / Non-Goals

**Goals:**
- Add a local verification step that runs per page, reviewing each crop image as the primary subject with the page image as context, plus page XMP and crop XMP together.
- Capture human-style album reasoning in a structured, reproducible output schema.
- Produce machine-readable judgments for `caption`, `gps`, `shown_location`, `date`, and `overall`.
- Explain what a human would actually infer whenever the verifier finds a mismatch or uncertainty.
- Make verification runnable as part of the local process pipeline and in single-step mode.
- Record readable follow-up routing in `imago:Detections["pipeline"]` so later passes can focus on the specific concern that failed.
- Record that the verification pass ran and preserve the latest concern states so the pipeline can safely decide whether another targeted pass is needed.
- Record the specific failure reason for each bad or uncertain concern.
- Rerun the failed concern before the page is considered complete, using the verification reason to focus the retry.
- Record provenance for the accepted result, including the prompt/model/tuning configuration that produced it.
- Use a fixed second pass with a trimmed retry prompt, then a final third pass where the verification model suggests better tuning params from a fresh full-context session.
- Make the process pipeline legible in logs, including what each metadata-producing step discovered.

**Non-Goals:**
- Automatically rewriting XMP metadata based on verification output.
- Replacing production ownership boundaries for Docling, Gemma, `buffalo_l`, YOLO, or Nominatim.
- Treating verification results as a new source of truth for metadata; verification is an adjudication layer, not a producer.
- Building an open-ended retry loop with unlimited automatic attempts.

## Decisions

### D1: Verification runs per page over final outputs
**Decision:** The verifier consumes the page image, page XMP text, and all crop images and crop XMP texts for that page after crop and metadata assembly are complete.

**Rationale:** The user’s desired reasoning is explicitly comparative and contextual. The crop is the review target, the page provides the context needed to interpret captions and dates, and a page-oriented run keeps that shared context stable while evaluating sibling crops that may depend on one another.

**Alternative considered:** Verify only the crop and crop XMP. Rejected because many judgments depend on page layout, neighboring captions, and context from the page XMP.

### D2: The verifier returns structured judgments and routing signals, not free-form commentary
**Decision:** The verifier returns `good`, `bad`, or `uncertain` plus one-sentence reasoning for each review dimension: `caption`, `gps`, `shown_location`, `date`, and `overall`. It also returns a human-inference field describing what a person would actually infer when anything is bad or uncertain, plus readable routing fields such as `needs_another_pass` and `needs_human_review`. The pipeline state records that the verification pass ran, preserves the latest concern states for each reviewed crop, stores the specific failure reason for any bad or uncertain concern, and records provenance for the prompt/model/tuning configuration that produced the current accepted state.

**Rationale:** This makes the verifier output testable, comparable across runs, and useful in local tooling. Free-form review text alone is too hard to consume in the pipeline, and the user wants the pipeline state to show both that verification happened and exactly what still needs attention.

**Alternative considered:** Return only a single overall pass/fail. Rejected because the user explicitly wants to know which aspect is wrong and why.

### D3: Verification remains advisory; it does not mutate production metadata directly
**Decision:** Verification output is recorded as a review artifact and pipeline state, but it does not rewrite XMP automatically.

**Rationale:** Verification should help humans or later correction tooling decide what to fix. Automatically overwriting metadata from an adjudication step would blur production and review responsibilities, especially when the user wants later passes to use different prompts or models for specific concerns.

**Alternative considered:** Have the verifier directly patch metadata when confidence is high. Rejected because this change is about review, not auto-repair.

### D4: Verification runs locally as a process step after metadata assembly
**Decision:** Add a `verify-crops` pipeline step after `ai-index` in the unified process pipeline, with the same local single-step behavior available through `--step verify-crops`.

**Rationale:** The verifier depends on the crop image and the finalized crop/page XMP, so it belongs after metadata assembly. Putting it in the process pipeline makes the user’s “run locally” goal concrete without requiring a second orchestration path.

**Alternative considered:** Make verification only a standalone ad hoc command. Rejected because the user wants a repeatable local pipeline, not just a separate one-off tool.

### D5: Review criteria explicitly model human album reasoning
**Decision:** The verifier prompt/schema explicitly encodes:
- caption proximity and plausibility
- caption carry-over reasoning when neighboring captions need to be combined to make sense of a later photo
- shown location from page text, landmarks, and adjacent caption context
- GPS plausibility based on whether the page context appears specific enough to support the resolved place
- date from written or implied page context
- overall consistency

**Rationale:** The desired behavior is not generic image QA; it is a specific family-album review task with common-sense contextual reasoning.

**Alternative considered:** Use a generic “is this metadata correct?” prompt. Rejected because it would underspecify the kind of human reasoning the user wants reproduced.

### D6: Concern-specific reruns must not rewrite unrelated metadata
**Decision:** Follow-up routing is concern-specific. A later targeted rerun for `gps`, `caption`, `shown_location`, or `date` should only update the fields for the named concern and should not rewrite unrelated fields that are already `good`.

**Rationale:** The user explicitly wants to avoid oscillation where one retry breaks a field that was already correct. Concern-specific routing only helps if later reruns are constrained to that concern.

**Alternative considered:** Let any targeted rerun rewrite all crop metadata opportunistically. Rejected because it invites infinite churn and makes the verification state hard to trust.

### D7: Failed concerns are retried before the page is complete
**Decision:** When verification marks a concern as `bad` or `uncertain`, the pipeline immediately reruns that specific concern before the page is considered complete. Pass 2 uses a trimmed prompt contract made of the base concern prompt, the specific concern issue, and the problem-to-fix summary from verification. If the concern still is not `good`, pass 3 opens a fresh full-context session, resends the images and metadata, asks the verification model to suggest better tuning params, and then reruns that concern with those suggested params.

**Rationale:** The user wants the verification step to be actionable during the same page run rather than only leaving work for a later batch. A fixed second pass keeps the retry simple and focused, while the final fresh-context third pass gives the system one stronger attempt with better settings before escalating.

**Alternative considered:** Only record the failure and leave all reruns to a later pass. Rejected because it delays obvious recoverable fixes and leaves the page in a less-finished state than necessary.

### D8: Retry sequence is pass 2, then a final third pass, then human review
**Decision:** A failed concern may receive one immediate concern-specific retry as pass 2. If the concern still is not `good`, the pipeline performs one final third pass that first asks the verification model for better tuning params in a fresh full-context session, then reruns the concern with those params. If the concern still is not `good`, the pipeline escalates that concern to human review.

**Rationale:** The user wants iteration, but not an endless automatic loop. This gives one focused retry and one stronger full-context retry before handing off to a human.

**Alternative considered:** Keep retry count open-ended or leave it unspecified. Rejected because it makes the pipeline harder to reason about and invites loops.

### D9: Third-pass parameter suggestion happens in a fresh full-context session
**Decision:** The third-pass parameter-suggestion step runs in a fresh session that resends the relevant images and metadata context for the page and concern under review.

**Rationale:** The user explicitly wants the verification model to see the full context again when proposing better params. A fresh session reduces the risk that the parameter suggestion is biased by partial conversational state from earlier retries.

**Alternative considered:** Ask for better params inline in the same retry session. Rejected because it does not match the desired clean re-evaluation workflow.

### D10: Result changes trigger re-verification when the last status was not good
**Decision:** If a concern retry changes the concern value and the last verification status for that concern was not `good`, the system reruns verification for that concern before accepting the updated state.

**Rationale:** The user wants provenance recorded and wants changed results to be checked rather than assumed correct. Re-verifying changed results prevents silent regressions and makes the retry loop observable.

**Alternative considered:** Accept retry output immediately without another verification pass. Rejected because it weakens the entire verification-driven workflow.

### D11: Region correction is out of scope for this change
**Decision:** This change does not introduce a `region` review dimension for crop geometry.

**Rationale:** The user clarified that the important failures here are caption reasoning, date understanding, shown location grounding, and GPS/location disambiguation rather than crop-boundary geometry. Keeping `region` in scope would dilute the targeted follow-up model runs this change is meant to enable.

**Alternative considered:** Keep geometry-oriented `region` review in the same verifier. Rejected because it mixes a different class of concern into a workflow now focused on metadata review and rerun routing.

### D12: Pipeline logging must expose step discoveries and deltas clearly
**Decision:** The process pipeline logs each step start/completion and, for metadata-producing steps, logs the values or conclusions discovered in that step in a human-readable way. Retry logs must include before/after values and whether a retry changed the value or repeated the same result.

**Rationale:** The user wants a crisp, inspectable pipeline and specifically called out that too much behavior is hidden inside `ai-index`. Logging what each step discovered, plus the before/after delta during retries, makes troubleshooting and trust much better without changing ownership boundaries.

**Alternative considered:** Keep only coarse step-level logging such as "ai-index ran." Rejected because it hides too much of the pipeline behavior and makes it difficult to understand what the model actually found.

## Risks / Trade-offs

- [Verifier disagrees with production pipeline in ambiguous cases] → Mitigation: support `uncertain` explicitly and require one-sentence reasoning instead of forcing false precision.
- [Review output becomes noisy if run on every crop by default] → Mitigation: keep the output structured and local, and allow users to run the verification step in isolation.
- [Verification artifacts drift from the exact crop/XMP inputs used at review time] → Mitigation: run verification after metadata assembly and record enough context to tie results to the reviewed page/crop pair.
- [Users may assume verification is authoritative enough to auto-fix metadata] → Mitigation: keep the verifier advisory in this change and do not let it rewrite production XMP.
- [Follow-up routing becomes unreadable if encoded as terse flags] → Mitigation: store explicit concern names such as `caption`, `gps`, `shown_location`, and `date` rather than short abbreviations.
- [Concern-specific retries may oscillate if they rewrite unrelated fields] → Mitigation: constrain later reruns to the named concern and preserve already-good fields.
- [Immediate retries may repeat the same mistake without learning from the failure] → Mitigation: feed the verification failure reason into the retry prompt and narrow the retry to the failed concern.
- [Final retries may still get stuck if they reuse weak inference settings] → Mitigation: use a dedicated third-pass parameter-suggestion session with full context and record the resulting provenance.
- [Verbose logging could become noisy] → Mitigation: log one crisp summary per step and discovered value rather than dumping raw model responses.

## Migration Plan

1. Define the review capability and structured output schema.
2. Add a local `verify-crops` process step that runs after crop and metadata assembly.
3. Persist verification results as review artifacts and pipeline state without mutating XMP, including page-level pass recording, latest concern states, failure reasons, readable rerun and human-review routing, and accepted-result provenance.
4. Rerun failed concerns before the page completes, using the failure reason to focus pass 2.
5. Run one final third pass that asks the verification model for better tuning params in a fresh full-context session, then rerun the concern with those params.
6. Escalate to human review if the concern still is not `good`.
7. Add clear process logging so each step shows what it discovered, including before/after retry deltas.
8. Add regression fixtures covering obviously correct, obviously wrong, and ambiguous page/crop pairs.

Rollback is straightforward: remove the `verify-crops` process step and stop producing verification artifacts. Because this change is advisory and does not rewrite XMP, rollback does not require metadata repair.
