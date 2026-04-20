## Context

The `ai-index` stage currently executes as a single monolithic pass: OCR → people detection → caption → GPS → locations-shown → objects → date-estimate, all gated by a single "is this sidecar current?" check. Settings or upstream data changes trigger a complete rerun of all steps. The existing XMP sidecar tracks one `imago:Processing` blob with a single settings signature; there is no per-step completion record.

The pipeline already has a `write_pipeline_step` / `read_pipeline_step` API in `xmp_sidecar.py` that other stages (e.g. `detect-regions`) use, but `ai-index` does not use it — it uses bespoke signature fields inside `imago:Detections`.

## Goals / Non-Goals

**Goals:**
- Define a named step graph for `ai-index` with explicit dependency edges
- Store per-step completion records (step name, settings hash, timestamp) in the `"pipeline"` sub-key of the `imago:Detections` JSON blob
- On each run, evaluate staleness per step and skip steps whose inputs haven't changed
- Invalidate downstream steps when an upstream step reruns
- Allow individual steps to be targeted for forced re-run via CLI flag
- Existing XMP sidecars are forward-compatible: missing step records → step treated as "not run"

**Non-Goals:**
- Parallelising step execution (steps run sequentially within a file; parallelism is across files)
- Changing what any step computes — only the dispatch and recording contract changes
- Persisting intermediate step outputs to disk as separate files (XMP detections payload continues to be the single record)
- Parallelising steps within a file

## Decisions

### 1. Step dependency tree

```
ocr ──────┐
          ├──► caption ──► locations ──┐
people ───┘         │                  ├──► propagate-to-crops
                    └──► date-estimate │
                                       │
(people also feeds propagate-to-crops)─┘

objects   (independent)
```

Edges mean "downstream step is forced stale when upstream reruns". `propagate-to-crops` runs last and pushes the final `locations` and `people` outputs into each crop XMP sidecar under `_Photos/`. Steps with no incoming edges (`ocr`, `people`, `objects`) are evaluated purely by their own input hash.

### 2. Step graph defined in code, not config

The step graph (names, deps, input hash functions, output field keys) is declared as a `dict[str, StepDef]` in a new `ai_index_steps.py` module. Steps are not user-configurable; the graph is a code contract.

**Alternative considered**: YAML/TOML config file for the step graph. Rejected — the hash functions and output field keys need to reference Python objects; externalising the graph would require a separate eval/plugin system for little gain.

### 3. `locations` step consolidates primary GPS and named locations into one AI call

The former `gps` and `locations-shown` steps are merged into a single `locations` step. One AI call receives the caption text + image and returns both the primary location query (for GPS coordinates) and a list of named location queries (for locations shown in the image). All queries are resolved via Nominatim.

This avoids two separate AI round-trips for what is fundamentally one question: "what places does this image show?" The primary GPS coordinate is the most prominent/singular location; named locations is the fuller list.

**Alternative considered**: Keep `gps` and `locations-shown` as separate steps. Rejected — they use the same model, same image, and same caption context; splitting them doubles the AI calls with no benefit.

The prompt SHALL explain to the model that Nominatim accepts free-form natural-language place name queries in any language (e.g. `"Eiffel Tower, Paris, France"`, `"Cafe Paris, New York"`), and that it MUST NOT return raw coordinates.

**Alternative considered**: Have AI return raw lat/lon directly. Rejected — models hallucinate coordinates; Nominatim resolution from a named query is more reliable and auditable.

### 4. `propagate-to-crops` step replaces `--include-view` for metadata sync

Rather than requiring a separate `ai-index --include-view` run to update crop sidecars, a `propagate-to-crops` step runs automatically as the final step of every page's ai-index pass. It:

1. Discovers the page's crop files from the MWG-RS region list in the page XMP
2. For each crop, writes the updated `locations` output (GPS fields) and the matching `people` output (person names for that region) to the crop's XMP sidecar
3. Records its own step entry in each crop's `imago:Detections["pipeline"]` under key `ai-index/propagate-to-crops`

This means crops no longer need to be discovered and processed as standalone images for GPS and people metadata — those fields are always driven from the parent page. Crops can still be run through ai-index independently for crop-specific analysis (OCR, objects), but that is opt-in.

The step's input hash covers the `locations` output hash + `people` output hash + the set of crop paths for the page (so adding or removing a crop triggers re-propagation).

**Alternative considered**: Keep `--include-view` as the mechanism. Rejected — it requires the user to remember to run a second pass, and there is no staleness tracking to know which crops need updating.

### 5. Input hash per step, not global settings signature

Each step declares an `input_hash_fn(settings, sidecar_state) -> str` that extracts only the fields relevant to that step. For example, the `ocr` step hashes only `ocr_engine`, `ocr_model`, `ocr_language`; the `caption` step hashes `caption_engine`, `caption_model`, `people_payload_hash` (from upstream step output).

This means changing the caption model does not invalidate OCR, and changing the cast store only invalidates `people` and downstream steps.

**Alternative considered**: Hash the entire `RenderSettings` object for every step. Rejected — this defeats the purpose; any settings change reruns everything.

### 6. XMP pipeline record is stored in the `"pipeline"` sub-key of `imago:Detections`

Step completion is written into the existing `imago:Detections` JSON blob (the same field used by the detections payload) under a top-level `"pipeline"` key. This is the same location `write_pipeline_step` / `read_pipeline_state` already read and write. Step keys are namespaced as `ai-index/<step-name>`, e.g. `ai-index/ocr`, `ai-index/caption`. Each entry stores:
```json
{
  "timestamp": "2026-04-19T15:30:00Z",
  "input_hash": "<sha256-prefix>",
  "result": "ok" | "skipped" | "error",
  "model": "<model-id>"    // optional, where applicable
}
```

**Alternative considered**: A separate `imago:AIIndexSteps` XMP field. Rejected — keeping all pipeline step records inside `imago:Detections["pipeline"]` (where `detect-regions` already writes) avoids XMP namespace proliferation and reuses the existing helpers.

### 7. Staleness rule: a step is stale if its `input_hash` differs OR any upstream step reran this session

Each step is evaluated in topological order. If a step reruns, it sets a `reran` flag in the in-memory execution context. Downstream steps check `any(dep.reran for dep in upstream_steps)` OR `recorded_hash != current_hash` to decide whether to execute.

This ensures consistency: if `people` reruns (cast store changed), `caption` always reruns even if the caption model hasn't changed, because caption incorporates people names.

### 8. Step runner wraps existing sub-functions

The existing `_run_image_analysis` sub-functions (`_run_ocr`, `_run_people`, `_run_caption`, etc.) are not rewritten. A new `StepRunner` class wraps each sub-function:
1. Compute `input_hash`
2. Check recorded XMP step state
3. If not stale, load previous output from `imago:Detections` payload and return it
4. If stale, call the sub-function, update `imago:Detections` payload slice, record the step
5. Mark step as `reran=True`

The final XMP write happens once after all steps complete, as today.

## Risks / Trade-offs

- **Hash collisions** → SHA-256 prefix (16 hex chars) is used; collision probability is negligible for this use case
- **Step record drift** → If a step's `input_hash_fn` changes (e.g. we add a new settings field to OCR), existing sidecars will hash-miss and rerun once, then stabilise. This is the desired behaviour.
- **Partial step record on crash** → If the process crashes between steps, some steps have records and others don't. On next run, recorded steps are skipped (if not stale) and unrecorded steps run. This is safe because intermediate results are held in memory, not written to XMP until the final write; a crash means the final write never happened, so the sidecar reflects the last complete run. The new step records are written only after the final payload write succeeds.
- **Fast path removal** → `_process_people_update` and `_process_gps_update` are removed and replaced by the step graph. Files previously routed through these paths will run a full step-graph evaluation on the next invocation; steps whose hashes haven't changed will skip immediately, so the cost is negligible.

## Migration Plan

1. **Migrate existing pipeline step records** — existing `detect-regions` entries use `{"completed": <timestamp>}`. On read, normalise old-format entries to the new schema (`completed` → `timestamp`, `result` inferred as `"ok"`). Write all new entries in the new schema. A one-off migration pass over all XMP sidecars updates existing records in place.
2. Implement `ai_index_steps.py` with step graph and `StepRunner`
3. Update `xmp_sidecar.py` with `write_pipeline_steps` and schema-normalising `read_pipeline_state`
4. Wire `StepRunner` into `_run_image_analysis` in `ai_index_runner.py`
5. Remove `_process_people_update` and `_process_gps_update`; route all processing through the step graph
6. Update `ai_index.py` CLI to accept `--steps <step>[,<step>]` for forced re-run of specific steps
7. Existing sidecars with no `"pipeline"` key: no action needed — missing step records cause steps to run once on next invocation

**Rollback**: Step records are additive to `imago:Detections`; old code ignores the `"pipeline"` key. The schema migration (step 1) is the only irreversible part — old `{"completed": ts}` entries are rewritten.

## Resolved Decisions

- **`date-estimate` is a step in the graph** with `input_hash` covering `lmstudio_date_estimate_model` + `ocr_hash` + `caption_output_hash`. When the engine is not LMStudio or the model is not configured, the step records `result: "not-applicable"` with `input_hash: ""`. An empty hash ensures the step reruns if the engine is later enabled.
- **`objects` follows the same pattern**: when no object detection model is configured, records `result: "not-applicable"` with `input_hash: ""`.
- **Skipped steps do not update `timestamp`**: on skip, the existing record is left unchanged. This preserves the original run time for debugging.
