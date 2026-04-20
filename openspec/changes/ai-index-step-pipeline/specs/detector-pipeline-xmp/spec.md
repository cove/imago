## ADDED Requirements

### Requirement: Per-step completion records are stored in the "pipeline" sub-key of imago:Detections
The system SHALL store pipeline step completion records as a JSON object under the `"pipeline"` key within the `imago:Detections` JSON blob. Each key in the object SHALL be a namespaced step identifier of the form `<stage>/<step>` (e.g. `ai-index/ocr`, `ai-index/caption`, `detect-regions/docling`). Each value SHALL be an object with the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `timestamp` | ISO-8601 string | yes | UTC time when the step completed |
| `input_hash` | hex string (≥ 16 chars) | yes | Hash of all inputs that affect this step's output |
| `result` | `"ok"` \| `"skipped"` \| `"error"` \| `"not-applicable"` | yes | Outcome of this step execution; `"not-applicable"` means the required engine or model is not configured |
| `model` | string | no | Model identifier used, where applicable |

#### Scenario: OCR step completes successfully
- **WHEN** the `ai-index/ocr` step runs and produces output
- **THEN** `imago:Detections["pipeline"]["ai-index/ocr"]` is written with `result: "ok"`, a current `input_hash`, and a `timestamp` reflecting the completion time

#### Scenario: Step is skipped due to unchanged inputs
- **WHEN** a step's recorded `input_hash` matches the current hash and no upstream step reran
- **THEN** the existing `"pipeline"` entry for that step is left unchanged (timestamp and hash are NOT updated)

#### Scenario: Step fails
- **WHEN** a step encounters a runtime error
- **THEN** the `"pipeline"` entry for that step records `result: "error"`; the entry SHALL still include `input_hash` and `timestamp` so the failure is attributable to a specific run

#### Scenario: Step cannot run because engine is not configured
- **WHEN** `date-estimate` or `objects` runs but the required engine or model is not configured
- **THEN** the step records `result: "not-applicable"` with `input_hash: ""`; the empty hash ensures the step reruns if the engine is later enabled

---

### Requirement: Legacy pipeline step records are migrated to the new schema on read and write
Existing XMP sidecars written by earlier versions of `write_pipeline_step` use the legacy format `{"completed": "<ISO-8601>"}` with no `input_hash` or `result` fields. The system SHALL handle these transparently:

- **On read**: `read_pipeline_state` SHALL normalise any entry that has `"completed"` but no `"timestamp"` by mapping `completed → timestamp` and inferring `result: "ok"` and `input_hash: ""`. An empty `input_hash` means the step will be treated as stale on the next staleness check, triggering a clean rerun.
- **On write**: all new entries SHALL use the new schema. Legacy entries for the same key are overwritten.
- **Migration pass**: a one-off `migrate-pipeline-records` sub-command SHALL rewrite all legacy entries across all XMP sidecars in a directory tree, converting them to the new schema in place.

#### Scenario: Existing detect-regions record in legacy format
- **WHEN** `read_pipeline_state` reads a sidecar whose `detect-regions/docling` entry contains only `{"completed": "2025-01-15T10:00:00Z"}`
- **THEN** the returned dict contains `{"timestamp": "2025-01-15T10:00:00Z", "result": "ok", "input_hash": ""}` and no error is raised

#### Scenario: Legacy entry triggers stale evaluation
- **WHEN** a normalised legacy entry has `input_hash: ""`
- **THEN** the step is treated as stale (empty hash never matches any computed hash) and reruns on the next invocation

#### Scenario: Migration pass updates all legacy entries
- **WHEN** `migrate-pipeline-records` runs over a directory
- **THEN** every XMP sidecar with legacy `{"completed": ...}` entries has them rewritten to the new schema, and sidecars with no `"pipeline"` key are left untouched

---

### Requirement: Missing step records are treated as not-run
The system SHALL treat an absent key in `imago:Detections["pipeline"]` for any step as equivalent to the step never having run. Absent keys SHALL NOT be treated as errors. Existing XMP sidecars written before this feature was introduced SHALL be processed normally, with all steps treated as stale on first run.

#### Scenario: Existing sidecar with no "pipeline" key in imago:Detections
- **WHEN** `ai-index` processes an image whose `imago:Detections` JSON has no `"pipeline"` key
- **THEN** all steps are treated as not-run and execute in full; after the XMP write all step records are present under `"pipeline"`

#### Scenario: Sidecar has some but not all step records
- **WHEN** `imago:Detections["pipeline"]` contains records for `ai-index/ocr` and `ai-index/people` but not `ai-index/caption`
- **THEN** `ocr` and `people` are evaluated for staleness normally, and `caption` is treated as not-run and executes unconditionally

---

### Requirement: pipeline read/write helpers in xmp_sidecar operate on imago:Detections["pipeline"]
The `xmp_sidecar` module SHALL expose the following functions for reading and writing pipeline step records, all operating on the `"pipeline"` sub-key of `imago:Detections`:

- `read_pipeline_state(xmp_path) -> dict[str, dict]` — returns the full `imago:Detections["pipeline"]` dict, or `{}` if absent
- `read_pipeline_step(xmp_path, step_key) -> dict | None` — returns the record for a single step key, or `None` if absent
- `write_pipeline_steps(xmp_path, updates: dict[str, dict])` — merges the provided step records into the existing `"pipeline"` dict and writes the XMP

These functions SHALL be usable by any pipeline stage (`ai-index`, `detect-regions`, etc.) without duplication.

#### Scenario: Writing step records merges with existing records
- **WHEN** `write_pipeline_steps` is called with records for `ai-index/ocr` and `ai-index/caption`
- **THEN** `imago:Detections["pipeline"]` is updated with the new records while all other existing keys (e.g. `detect-regions/docling`) are preserved unchanged

#### Scenario: Reading a missing step returns None
- **WHEN** `read_pipeline_step` is called for a step key that does not exist in `imago:Detections["pipeline"]`
- **THEN** the function returns `None` without raising an exception

---

### Requirement: detect-regions stage writes its outcome into imago:Detections["pipeline"]
The `detect-regions` stage pipeline step outcome SHALL be written using the shared `write_pipeline_steps` helper under the key `detect-regions/<engine>` (e.g. `detect-regions/docling`). The step entry SHALL follow the same field schema as all other pipeline step records.

#### Scenario: Docling detection completes and writes its step record
- **WHEN** the Docling region detection step runs successfully
- **THEN** `imago:Detections["pipeline"]["detect-regions/docling"]` contains `result: "ok"` and a populated `input_hash`

#### Scenario: detect-regions step record coexists with ai-index step records
- **WHEN** both `detect-regions` and `ai-index` have run for the same image
- **THEN** `imago:Detections["pipeline"]` contains entries for both `detect-regions/docling` and each `ai-index/<step>` without conflict
