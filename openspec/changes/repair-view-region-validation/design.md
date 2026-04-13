## Context

`ai_view_regions.py` currently accepts model-proposed boxes, applies limited geometric validation, and writes the surviving results into the page XMP sidecar as the source of truth. The observed failures are not just malformed JSON or page-sized boxes; they are semantically wrong photo boundaries that include striped album paper, caption plaques, text panels, or neighboring photos. Those failures then flow directly into crop generation.

The bad-crop examples on `Europe_1985_B02_P04` show two related problems:

- the detector can return overlapping boxes for neighboring bottom-row photos
- the detector can produce boxes that cross into separator content such as text plaques and album background
- the detector can produce boxes that are degenerate rectangles that have no area (area = 0)  

The local validation cost is low compared with a model call, so we can afford to validate region sets every run, not just when the page is first detected.

## Goals / Non-Goals

**Goals:**
- Validate every candidate region set on every relevant run, including previously stored XMP regions
- Detect obviously invalid boxes, especially heavy overlap and separator-crossing boxes
- Feed concrete validation failures plus the previous region set back into Gemma on retry
- Reprocess stale XMP region sets when they no longer satisfy current validation rules
- Keep XMP as the source of truth, but only after a region set passes validation

**Non-Goals:**
- Build a full photo-edge segmentation system
- Add a second mandatory verifier model pass for every page
- Guarantee recovery of every missing photo in the first iteration; rejecting unsafe boxes is acceptable if repair still fails
- Redesign crop numbering or metadata schema beyond what is needed for validated rewrites

## Decisions

### Validate all stored and fresh regions on every run
Every run that touches view regions should validate the current region set, even when regions already exist in XMP. If the stored regions fail current rules, the page is treated as stale and is reprocessed. Rationale: overlap and suspicious-box checks are fast, and stale bad regions are worse than re-running a small number of pages.

Alternatives considered:
- Validate only newly detected regions: rejected because existing bad XMP region lists would remain trusted forever
- Validate only when `--force` is used: rejected because normal runs would keep emitting bad crops from stale regions

### Use a two-stage validator: hard failure rules first, suspicious-box rules second
The validator should distinguish:

- hard-invalid regions: overlap too large, zero/negative size, box mostly outside image, or other clearly unacceptable geometry
- suspicious regions: likely separator/text/background intrusion detected by local image heuristics

Hard-invalid sets always require repair. Suspicious sets also require repair, but the recorded reason should stay explicit so we can tune thresholds later.

Alternatives considered:
- single boolean valid/invalid check only: rejected because we lose useful retry feedback and diagnostics

### Retry Gemma with prior boxes and explicit validation errors
When validation fails, the retry prompt should include:

- the previous full region set
- which regions are invalid or suspicious
- why they failed
- the invariants for the next answer, such as non-overlap and one physical print per box

The retry should request a complete revised region set for the same image rather than partial edits. The response may also include an optional `error_analysis` string that briefly explains what the model believes was wrong with the prior boxes. `error_analysis` is recorded for debugging and prompt review only; it is not part of acceptance logic. Rationale: full-set replacement keeps indexing and writeback simpler than patching individual boxes, while the explanation gives us a cheap visibility channel into whether the model understood the validation feedback.

Alternatives considered:
- retry with only a stricter prompt and no prior boxes: rejected because the model cannot reliably infer what was wrong
- ask the model to patch only invalid boxes: rejected because partial edits make indexing, overlaps, and writeback more brittle
- require `error_analysis` and trust it for acceptance: rejected because model self-diagnosis is useful telemetry but not reliable enough to replace local validation

### Keep local validation as the final authority
Even after a retry, the returned region set must pass local validation before it is written to XMP or used for cropping. If retries still fail, the page should remain unresolved rather than writing known-bad regions. Rationale: the model may repeat the same failure mode in different coordinates.

Alternatives considered:
- trust the second model attempt automatically: rejected because it can still overlap or cross separators

### Start with overlap-driven repair plus auditable suspicious-box hooks
The first implementation should include strong overlap validation and the retry-feedback path, while structuring suspicious-box checks so we can add heuristic signals without redesigning the loop. Candidate suspicious signals include text intrusion, striped-paper intrusion, and separator crossing.

Alternatives considered:
- implement all heuristic classifiers before wiring retry: rejected because overlap-only feedback already fixes a concrete failure mode and keeps the first increment focused

## Risks / Trade-offs

- Stricter validation may reject boxes that are usable but imperfect → Prefer missing a crop over writing a wrong crop, and keep retries before final rejection
- Heuristic suspicious-box rules may overfit current album layouts → Start with overlap and simple separator cues, keep thresholds explicit, and record validation reasons for review
- Revalidating stored regions on every run may trigger extra model calls on pages that were previously accepted → The local pass is cheap, and repair should only occur when stored regions are now known-bad
- Full-set retry may reorder or renumber regions between attempts → Treat retries as authoritative replacements and keep crop regeneration tied to the accepted XMP list
