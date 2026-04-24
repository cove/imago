## Context

The existing pipeline in `photoalbums/commands.py::run_process_pipeline()` orchestrates 7 steps:
1. render — Stitch scans to page JPEGs
2. propagate-metadata — Copy XMP fields from archive to page
3. detect-regions — Find photo bounding boxes
4. crop-regions — Extract detected regions
5. face-refresh — Update face region metadata
6. ai-index — Run OCR, captions, GPS, object detection
7. verify-crops — Review crops against pages

Steps have explicit dependencies (e.g., crop-regions depends on detect-regions). The orchestrator:
- Iterates pages → iterates steps → calls step.run_fn()
- Checks staleness via `_check_step_stale()` before each step
- Tracks per-step counters and error handling
- Handles skip/redo flags

This logic is 400+ lines of imperative orchestration with no reuse outside this specific pipeline.

## Goals / Non-Goals

**Goals:**
- Remove hand-rolled orchestration boilerplate by leveraging Hamilton's DAG resolution
- Make steps testable as independent functions
- Reduce coupling between CLI dispatch and pipeline execution
- Maintain 100% feature parity with existing pipeline
- Keep migration path optional (photoalbums3 coexists with photoalbums)

**Non-Goals:**
- Change what any step computes or outputs
- Modify XMP metadata schema
- Parallelize step execution (remain sequential per page)
- Add distributed execution or cloud orchestration (stay local-only)

## Decisions

### 1. Directory Structure: photoalbums3/

Create new `photoalbums3/` alongside existing `photoalbums/`:
```
photoalbums3/
├── __init__.py
├── pipeline.py              # Hamilton DAG definition (step functions)
├── steps/
│   ├── render.py            # render step handler
│   ├── detect_regions.py     # detect-regions step handler
│   ├── crop_regions.py       # crop-regions step handler
│   ├── face_refresh.py       # face-refresh step handler
│   ├── ai_index.py           # ai-index step handler (delegates to lib)
│   └── verify_crops.py       # verify-crops step handler
└── runner.py                # Hamilton driver & CLI interface
```

**Rationale**: Keeps new implementation isolated, allows gradual migration, doesn't break existing code.

### 2. Step Functions as Hamilton DAG Nodes

Each step is a pure function in `photoalbums3/steps/*.py`:

```python
# steps/render.py
def render_page(
    archive: Path,
    view_dir: Path,
    photos_dir: Path,
    page_group: tuple,
    force: bool = False,
) -> dict:
    """Render page scans to JPEG.
    
    Input: archive, page_group from list_page_scans()
    Output: {"view_path": Path, "xmp_path": Path, "status": "rendered"|"skipped"|"failed"}
    """
    # ... existing stitch_oversized_pages logic ...
    return {"view_path": view_path, "xmp_path": xmp_path, "status": status}

# steps/detect_regions.py
def detect_regions(
    render_page: dict,  # Depends on output of render_page
    view_dir: Path,
    # ... other params ...
) -> dict:
    """Detect photo regions (depends on render).
    
    Input: render_page output, xmp_path
    Output: {"regions": list[Region], "status": "detected"|"skipped"}
    """
    if render_page["status"] == "skipped":
        return {"status": "skipped"}
    # ... existing detect_regions logic ...
    return {"regions": regions, "status": "detected"}
```

Hamilton resolves dependency order from function parameters: `detect_regions(render_page: dict)` means "run detect_regions after render_page."

### 3. Staleness Checking via Custom Validators

Hamilton supports custom `is_stale` callbacks per function. Define per-step staleness:

```python
def render_page(
    ...,
    is_stale: hamilton.is_stale = hamilton.callable_is_stale(
        lambda: check_render_stale(view_path, xmp_path)
    ),
) -> dict:
    ...
```

Or simpler: implement `is_stale()` as a separate function in each step module that Hamilton calls before execution.

### 4. Page Iteration Model

Pages are processed one at a time (no parallel execution within a page). The runner loops:
```python
for page in all_pages:
    dr = hamilton.Driver(...)
    result = dr.execute(
        ["verify_crops"],  # Final output node
        inputs={"page": page, "force": force, ...},
    )
```

Each page gets its own Hamilton graph execution with page-specific inputs.

### 5. CLI Interface: runner.py

Create a unified runner module that:
- Builds the Hamilton driver
- Implements the same CLI flags as existing `run_process_pipeline()`
- Wraps page iteration and error handling
- Returns exit codes matching existing behavior

```python
# runner.py
def run_pipeline(
    album_id: str,
    photos_root: str,
    page: str | None,
    skip_ids: list[str],
    redo_ids: list[str],
    force: bool,
    # ... other flags ...
) -> int:
    """Run the pipeline using Hamilton."""
    dr = hamilton.Builder()
        .with_modules(photoalbums3.steps)
        .build()
    
    for page_group in iter_pages(...):
        result = dr.execute(
            ["verify_crops"],
            inputs={...},
        )
        # Handle result, track counters, errors
    
    return exit_code
```

### 6. Integration with CLI

Update `photoalbums/cli.py` to add optional flag:

```python
render_parser.add_argument(
    "--use-hamilton",
    action="store_true",
    help="Use Hamilton-based pipeline (experimental)",
)
```

If `--use-hamilton`, delegate to `photoalbums3.runner.run_pipeline()` instead of `commands.run_process_pipeline()`.

### 7. Backward Compatibility

- Existing `photoalbums/` remains unchanged
- XMP sidecar format is identical
- Output files are identical
- New Hamilton pipeline writes the same metadata

Users can opt-in via `--use-hamilton` flag without affecting existing workflows.

## Alternatives Considered

**1. Refactor existing photoalbums with minimal changes**
- Pro: No new directory/package
- Con: Still coupling CLI to orchestration; minimal benefit
- Rejected: Doesn't achieve the goal of separating concerns

**2. Use Prefect instead of Hamilton**
- Pro: More features, larger community
- Con: Heavier (~50MB), overkill for local batch processing
- Rejected: Against project's minimalism philosophy

**3. Implement skip/redo via command-line overrides to Hamilton inputs**
- Pro: Leverages Hamilton's native config
- Con: Less intuitive for users; harder to understand step skipping
- Rejected: Explicit step lists (`--skip render --redo ai-index`) are clearer

## Trade-Offs

- **Explicit vs. Implicit**: Function parameters make dependencies explicit (good for clarity) but require consistent naming
- **Testing**: Individual steps are easier to test as functions, but graph-level integration tests need Hamilton driver setup
- **Error Handling**: Current code tracks per-step errors explicitly; Hamilton requires error handling in each step function
