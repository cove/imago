from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class PipelineStep:
    id: str
    label: str
    depends_on: list[str]
    optional: bool = False


def _make_steps() -> list[PipelineStep]:
    return [
        PipelineStep(
            id="scan-ai",
            label="Run AI on archive scan (OCR, YOLO objects, Immich faces, GPS, date, people)",
            depends_on=[],
        ),
        PipelineStep(
            id="render",
            label="Stitch/convert archive scans to page view JPEGs",
            depends_on=["scan-ai"],
        ),
        PipelineStep(
            id="detect-regions",
            label="Detect photo bounding boxes and write MWG-RS XMP regions",
            depends_on=["render"],
        ),
        PipelineStep(
            id="crop-regions",
            label="Crop detected regions to _Photos/ directory",
            depends_on=["detect-regions"],
        ),
        PipelineStep(
            id="face-refresh",
            label="Update face region metadata on rendered outputs",
            depends_on=["crop-regions"],
        ),
        PipelineStep(
            id="immich-face-refresh",
            label="Refresh face region metadata from current Immich assets",
            depends_on=["crop-regions"],
        ),
        PipelineStep(
            id="ai-index",
            label="Run AI pipeline (OCR, caption, GPS, XMP write)",
            depends_on=["face-refresh"],
        ),
        PipelineStep(
            id="sequence-page-dates",
            label="Sequence _Archive/_Pages dates for viewer sort order",
            depends_on=["ai-index"],
        ),
        PipelineStep(
            id="verify-crops",
            label="Review each page's crops against the page image and page/crop XMP context",
            depends_on=["sequence-page-dates"],
            optional=True,
        ),
    ]


PIPELINE_STEPS: list[PipelineStep] = _make_steps()

VALID_STEP_IDS: list[str] = [s.id for s in PIPELINE_STEPS]

OPTIONAL_STEP_IDS: set[str] = {s.id for s in PIPELINE_STEPS if s.optional}


def validate_step_ids(ids: list[str], *, flag: str) -> list[str]:
    """Validate step ids against the registry; print error and exit 2 on unknown ids."""
    unknown = [sid for sid in ids if sid not in VALID_STEP_IDS]
    if unknown:
        print(
            f"Error: unknown step id(s) for {flag}: {', '.join(unknown)}\nValid step ids: {', '.join(VALID_STEP_IDS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return ids
