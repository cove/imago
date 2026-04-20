## Why

The photoalbums pipeline is split across many disconnected `justfile` targets and CLI subcommands (`detect-view-regions`, `crop-regions`, `face-refresh`, `render-pipeline`, `ctm`, `ctm-apply`, `ai`, `render`, etc.), making it hard to understand the full processing sequence, know which steps have run, or selectively redo individual steps. There is no way to see a step plan before running, and partial re-runs require memorizing which subcommand maps to which step.

## What Changes

- Replace the fragmented justfile targets and individual subcommands with a single `process` subcommand under `photoalbums.py`
- The `process` command runs all pipeline steps in order: render → propagate-metadata → detect-regions → crop-regions → face-refresh → ctm-apply → ai-index
- Steps can be individually skipped (`--skip <step>`) or force-rerun (`--redo <step>`)
- On startup the pipeline prints a numbered step plan showing which steps will run, which are skipped, and why
- On completion a summary table shows which steps ran, were skipped, or failed
- Existing individual subcommands are kept as aliases for backward compatibility but their justfile targets are consolidated

## Capabilities

### New Capabilities
- `unified-process-pipeline`: A single `photoalbums.py process` entry point that orchestrates all pipeline steps with consistent flags, step-level progress reporting, a startup plan display, and a completion summary.

### Modified Capabilities

## Impact

- `photoalbums/cli.py`: new `process` subparser with `--skip`, `--redo`, `--step`, `--photos-root`, `--album`, `--page`, and existing per-step flags (`--force`, `--debug`, `--no-validation`, `--skip-restoration`, `--force-restoration`)
- `photoalbums/commands.py`: new `run_process_pipeline` orchestrator that merges logic currently spread across `run_render_pipeline`, `run_ai_index`, `run_detect_view_regions`, `run_crop_regions`, `run_face_refresh`, `run_ctm_apply`
- `justfile`: consolidate the seven `photoalbums-*` targets into `photoalbums-process` (with optional args) and keep a few convenience aliases
