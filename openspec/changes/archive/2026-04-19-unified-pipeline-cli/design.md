## Context

The photoalbums CLI has grown organically: each new processing step was added as a new top-level subcommand (`detect-view-regions`, `crop-regions`, `face-refresh`, `ctm-apply`, `ai`, `render`, `render-pipeline`) and a matching justfile target. `render-pipeline` partially unified the rendering steps but left AI indexing separate and has no visibility into what will run or why steps are skipped. The result is a fragmented surface that forces users to know the internal ordering and pick the right subcommand for each partial re-run scenario.

## Goals / Non-Goals

**Goals:**
- Single `photoalbums.py process` command that runs all steps end-to-end in the correct order
- Step plan printed before execution (numbered list of steps, skip reason if applicable)
- Per-step status lines as execution proceeds (`[1/7] render ... done`, `[3/7] detect-regions ... skipped (complete)`)
- Completion summary table (step name, status, count/detail)
- `--skip <step>` to opt out of one or more steps (repeatable)
- `--redo <step>` to force-rerun one or more steps without affecting others (repeatable)
- All existing per-step flags forwarded under the same `process` command
- Existing individual subcommands remain as-is for backward compatibility
- Justfile consolidated from 10+ targets to `photoalbums-process` plus a few convenience aliases

**Non-Goals:**
- Changing the processing logic inside any individual step
- GUI or interactive TUI
- Parallel step execution (steps are inherently sequential per-page)
- Removing existing subcommands (kept for scripting compatibility)

## Decisions

### 1. Step registry in `photoalbums/lib/pipeline.py`

The `PipelineStep` dataclass and `PIPELINE_STEPS` list live in a single dedicated module. `commands.py` and `cli.py` import from it — no duplication. Adding, removing, or reordering a step requires editing one file.

Each step has: `id` (slug used in `--skip`/`--redo`/`--step`), `label`, `run_fn`, `skip_check_fn`, `depends_on` (list of step ids for cascade staleness), and `redo_clears` (pipeline-state keys cleared on `--redo`). This avoids ad-hoc `if skip_X` blocks and makes ordering and dependencies explicit and auditable.

*Alternative considered*: a dict keyed by step id. Rejected because ordering is implicit and hard to see at a glance.

### 2. `--redo` clears pipeline state then delegates to the same `run_fn`

`--redo detect-regions` internally calls `clear_pipeline_steps(xmp, ["view_regions"])` before running `detect_regions`. This reuses existing force/clear logic already in each step rather than a separate code path.

*Alternative considered*: `--redo` sets an internal `force=True` override only for that step. Same effect, but wording "redo" more clearly communicates intent to the user than "force".

### 3. Step plan display before any execution

Before iterating pages the pipeline prints a numbered block:

```
Pipeline steps (7 total):
  [1] render
  [2] propagate-metadata
  [3] detect-regions
  [4] crop-regions             (skipped: --skip crop-regions)
  [5] face-refresh
  [6] ctm-apply
  [7] ai-index                 (redo forced)
Album: Egypt_1975_B00, 34 pages
```

This replaces the current silent entry into processing.

### 4. Justfile consolidation

Keep `photoalbums-process` as the main target with `*args`. Retain `photoalbums-ai` and `photoalbums-map` as convenience wrappers since they have distinct mental models (AI indexing has its own flags; map is a long-running server). Remove the rest, replacing them with comments pointing to `photoalbums-process --skip`/`--redo`.

### 5. No changes to individual step implementations

`run_render_pipeline`, `run_detect_view_regions`, `run_crop_regions`, `run_face_refresh`, `run_ctm_apply`, `run_ai_index` are not modified. The new `run_process_pipeline` in `commands.py` calls them in sequence, coordinating skip/redo logic at the orchestration layer.

## Risks / Trade-offs

- [Duplicate skip/redo logic] Each step already has its own skip logic; the orchestrator adds a second layer. → Mitigation: orchestrator-level skip is opt-in only via explicit `--skip`; the step's own skip logic still handles idempotency within a step.
- [Breaking justfile muscle memory] Users who script against specific justfile targets will need to update. → Mitigation: keep `photoalbums-ai` and `photoalbums-map`; add a comment in the justfile documenting the migration.
- [Step id namespace] Users must type step ids correctly in `--skip`/`--redo`. → Mitigation: the process command validates step ids and prints the valid list if an unknown id is given.

## Migration Plan

1. Add `run_process_pipeline` to `commands.py` (new function, no changes to existing functions)
2. Add `process` subparser to `cli.py`
3. Update justfile (remove stale targets, add `photoalbums-process`)
4. Update AGENTS.md / usage docs if they reference removed justfile targets
