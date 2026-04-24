# Photoalbums3: DAG-Based Pipeline Explorer

A lightweight Python DAG execution engine + web UI for interactive pipeline debugging and parameter tuning.

## Architecture

**Three-part system:**

1. **pipeline.py** — Your pipeline definition (pure Python)
   - Define steps as functions
   - Declare dependencies via `depends_on`
   - Inline prompts for AI steps
   
2. **engine.py** — DAG execution engine
   - Builds DAG from step definitions
   - Smart re-execution (only runs stale steps)
   - Input/output caching per step
   - Parameter override support
   
3. **server.py + ui.html** — FastAPI server + web UI
   - Shows DAG as interactive graph
   - Click steps to inspect inputs/outputs
   - Edit prompts inline, retry instantly
   - Real-time updates via WebSocket

## Quick Start

### 1. Define your pipeline

Edit `pipeline.py`:

```python
from pipeline import StepDef

def my_step(input_data: dict) -> dict:
    """Do something."""
    return {"result": input_data}

PIPELINE_STEPS = [
    StepDef(
        id="step1",
        label="Process data",
        handler=my_step,
    ),
]
```

### 2. Run the server

```bash
cd photoalbums3
python server.py
```

Open http://localhost:8000 in your browser.

### 3. Execute & explore

1. Enter a page ID (e.g., `page_001`)
2. Click "Run Pipeline"
3. Click steps in the DAG to inspect
4. Edit prompts inline, click "Save & Retry"

## How It Works

### Step Execution Model

1. **Topological sort**: Steps execute in dependency order
2. **Staleness check**: Before running a step:
   - Has the input changed since last run? (hash check)
   - Did any upstream step just re-run?
   - If no → skip (cached output)
   - If yes → execute
3. **State persistence**: All step inputs/outputs saved to disk
4. **Downstream invalidation**: If a step re-runs, all dependents are marked stale

### Input/Output Visibility

Each step records:
- **Inputs**: All parameters passed to the function
- **Outputs**: The returned dict
- **Input hash**: SHA256 of inputs (detects changes)
- **Duration**: How long the step took
- **Status**: pending → running → done/failed/skipped

Click any step to see all of this in the inspector panel.

### Parameter Tuning Workflow

**Scenario**: `detect_regions` AI step produces bad results.

1. **Inspect**: Click the step, see inputs (image path) and outputs (regions)
2. **Edit**: Change the prompt in the UI
3. **Retry**: Click "Save & Retry"
4. **Watch**: Step re-runs with new prompt
5. **Inspect**: See new outputs in real-time
6. **Iterate**: Repeat until happy

The entire flow is visible in one place—no context switching.

## API Reference

### REST Endpoints

**GET /api/pipeline/dag**
Returns the DAG structure (steps and edges).

**POST /api/pipeline/run**
Execute full pipeline.
```
page_id: str (required)
context: dict (required)
final_step: str (optional, default: last step)
force: bool (optional, force re-run all steps)
```

**POST /api/step/execute**
Execute a single step.
```
step_id: str (required)
force: bool (optional)
```

**POST /api/step/override**
Override a step parameter.
```
step_id: str (required)
param: str (required, e.g. "prompt", "model")
value: any (required)
```

**GET /api/state**
Get current pipeline state.

**GET /api/step/{step_id}**
Get details of a single step (execution + definition).

### WebSocket

**WS /ws**
Real-time updates as steps execute.

Messages:
- `{"type": "initial_state", "state": {...}}`
- `{"type": "step_complete", "step": {...}}`
- `{"type": "param_override", "step_id": "...", "param": "...", "value": ...}`

## Integration with Existing Code

Replace your step handlers with calls to existing photoalbums functions:

```python
from photoalbums.lib.stitch_oversized_pages import stitch
from photoalbums.lib.ai_view_regions import detect_regions as detect_regions_impl

def detect_regions(render_page: dict, view_dir: str, prompt: str, model: str):
    """Detect regions."""
    # Call existing code
    regions = detect_regions_impl(
        view_path=render_page["view_path"],
        xmp_path=render_page["xmp_path"],
        prompt=prompt,  # Override with UI value
        model=model,
    )
    return {"regions": regions}
```

## State Persistence

Pipeline state is saved to `/tmp/photoalbums3_state/{page_id}.json`.

Contains:
- Step execution records (inputs, outputs, hashes, timestamps)
- Parameter overrides
- Execution status

Reload the page in the UI to see previous runs.

## Debugging

**Enable verbose logging:**
```python
# In engine.py, print statements show:
[render] RUN: not yet run
[render] DONE (0.45s)
[detect_regions] SKIP: inputs unchanged (hash abc12345)
[ai_index] RUN: inputs changed: def67890 → ghi01234
```

**Inspect state on disk:**
```bash
cat /tmp/photoalbums3_state/page_001.json | jq .steps.ai_index
```

## Performance

- **Caching**: Re-running skipped steps is instant (no computation)
- **Incremental**: Change one prompt, only that step + dependents re-run
- **Disk storage**: State files are small (JSON, ~1-5KB per page)

## Limitations

- **Sequential execution**: Steps run one at a time (no parallelism within a page)
- **Local only**: No distributed/cloud execution
- **Manual page selection**: No bulk processing (by design—focus on single-page debugging)
- **No scheduling**: Not a task scheduler (Prefect/Airflow are for that)

## Future Ideas

- Export step outputs to artifacts (save all crops, detections, etc.)
- A/B compare two prompt versions side-by-side
- Parameter sweep (test 5 prompts, rank by quality)
- Step templates (save good prompts as reusable templates)
- Diff tool (compare outputs of two runs)
