from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PipelineStep:
    id: str
    label: str
    run_fn: Callable[..., object]
    skip_check_fn: Callable[..., bool] | None
    depends_on: list[str]
    redo_clears: list[str]


def _noop_run(**_kwargs: object) -> None:
    pass


def _make_steps() -> list[PipelineStep]:
    # Steps are defined lazily by id/label/depends_on; run_fn is wired at call time
    # by run_process_pipeline because imports are heavy and optional.
    return [
        PipelineStep(
            id="render",
            label="Stitch/convert archive scans to page view JPEGs",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=[],
            redo_clears=[],
        ),
        PipelineStep(
            id="propagate-metadata",
            label="Copy safe archive XMP fields to page sidecar",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=["render"],
            redo_clears=[],
        ),
        PipelineStep(
            id="detect-regions",
            label="Detect photo bounding boxes and write MWG-RS XMP regions",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=["render"],
            redo_clears=["detect-regions", "view_regions"],
        ),
        PipelineStep(
            id="crop-regions",
            label="Crop detected regions to _Photos/ directory",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=["detect-regions"],
            redo_clears=["crop-regions", "crop_regions"],
        ),
        PipelineStep(
            id="face-refresh",
            label="Update face region metadata on rendered outputs",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=["crop-regions"],
            redo_clears=["face-refresh", "face_regions"],
        ),
        PipelineStep(
            id="ai-index",
            label="Run AI pipeline (OCR, caption, GPS, XMP write)",
            run_fn=_noop_run,
            skip_check_fn=None,
            depends_on=["crop-regions"],
            redo_clears=["ai-index"],
        ),
    ]


PIPELINE_STEPS: list[PipelineStep] = _make_steps()

VALID_STEP_IDS: list[str] = [s.id for s in PIPELINE_STEPS]


def validate_step_ids(ids: list[str], *, flag: str) -> list[str]:
    """Validate step ids against the registry; print error and exit 2 on unknown ids."""
    unknown = [sid for sid in ids if sid not in VALID_STEP_IDS]
    if unknown:
        print(
            f"Error: unknown step id(s) for {flag}: {', '.join(unknown)}\n"
            f"Valid step ids: {', '.join(VALID_STEP_IDS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return ids
