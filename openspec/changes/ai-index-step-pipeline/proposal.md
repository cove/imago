## Why

The `ai-index` pipeline stage runs OCR, people detection, caption generation, GPS extraction, locations-shown, object detection, and date estimation as a single monolithic pass. If any one step's settings or upstream data changes (e.g. cast store updated, caption model swapped), the entire pipeline reruns from scratch — wasting significant compute. This change decomposes `ai-index` into discrete, independently addressable steps with a dependency graph, so only stale steps and their downstream dependents are re-executed.

## What Changes

- **Break `ai-index` into named steps** with explicit inputs, outputs, and dependencies
- **Record per-step completion state** in the XMP sidecar under the `imago:detectors` pipeline namespace
- **Staleness evaluation** checks each step's recorded hash against current inputs; stale steps invalidate downstream steps
- **CLI gains step-level control**: individual steps can be targeted for re-run without forcing a full pipeline reset
- Each step's output is stored in the XMP detections payload; downstream steps read from there rather than recomputing

## Capabilities

### New Capabilities

- `ai-index-steps`: Defines the named steps within `ai-index` (ocr, people, caption, gps, locations-shown, objects, date-estimate), their declared input dependencies and output fields, and the staleness/invalidation rules
- `detector-pipeline-xmp`: Schema and read/write contract for recording which `imago:detectors` pipeline steps have run, with what settings hash, in the XMP sidecar

### Modified Capabilities

- `docling-region-detection`: Pipeline step tracking now uses the shared `imago:detectors` XMP record rather than bespoke sidecar fields

## Impact

- `photoalbums/lib/ai_index_runner.py` — major refactor of processing dispatch and step execution
- `photoalbums/lib/xmp_sidecar.py` — new pipeline step read/write helpers for `imago:detectors`
- `photoalbums/lib/ai_index.py` — orchestration updated to evaluate step staleness before dispatch
- Existing XMP sidecars remain valid; missing step records are treated as "not run" and trigger re-execution
