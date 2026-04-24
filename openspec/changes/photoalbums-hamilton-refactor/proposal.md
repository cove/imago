## Why

The current `photoalbums` pipeline orchestration is hand-rolled in `run_process_pipeline()` (1,491 lines in commands.py, ~400 lines of orchestration logic). It manually handles:
- Step dependency resolution
- Staleness checking via custom logic
- Skip/redo flags and forced re-runs
- Per-step execution and error handling
- Progress tracking and logging

This creates significant boilerplate with no reuse, making the codebase cumbersome and hard to extend. The orchestration logic is tightly coupled to CLI dispatch, making it difficult to test independently or reuse elsewhere.

## What Changes

Replace the hand-rolled orchestration with **Hamilton**, an actively maintained lightweight Python DAG orchestration framework:

- **Step definitions** are pure Python functions with explicit inputs/outputs; Hamilton automatically resolves the DAG from function signatures
- **Dependency resolution** is automatic from function parameters, eliminating manual step dependency declarations
- **Staleness checking** is built-in via custom `is_stale` checks per function
- **CLI dispatch** is simplified to function calls rather than orchestration loops
- **Step composition** is cleaner: each step is a testable function with no side effects

This refactoring creates **photoalbums3** as a cleaner, more maintainable implementation of the same pipeline with ~60% less orchestration boilerplate.

## Capabilities

### New Capabilities

- **Hamilton DAG Functions**: Replace hardcoded step list with function-based step definitions in `photoalbums3/pipeline.py`
- **Automatic Dependency Resolution**: Step execution order is derived from function signatures, eliminating manual `depends_on` lists
- **Stale-Check Decorators**: Per-step staleness is defined via callable decorators instead of inline checks
- **Step Handler Modules**: Each major step (render, detect-regions, crop-regions, etc.) is a focused function in its own module

### Modified Capabilities

- **CLI dispatch**: Simplified from nested conditionals to direct function calls
- **run_process_pipeline**: Replaced with Hamilton graph execution
- **Progress reporting**: Handled via Hamilton's built-in task runner

## Impact

- **New**: `photoalbums3/` directory with Hamilton-based pipeline
- **Modified**: `photoalbums/cli.py` gains optional `--use-hamilton` flag to opt into new pipeline
- **Unchanged**: All underlying step logic (stitch, detect-regions, crop, ai-index, etc.) remains the same
- **Migration path**: Existing `photoalbums/` remains functional; `photoalbums3` can coexist during transition

## Constraints

- Hamilton must be kept as a lightweight dependency (currently <100KB, 3 required dependencies)
- No changes to XMP metadata schema or step outputs
- Backward compatibility: existing XMP sidecars continue to work with new pipeline
