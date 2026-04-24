"""Pipeline definition in pure Python.

Define your DAG here. Each function is a step. Dependencies are inferred from
function parameters (if a param matches another step's name, it depends on it).
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import hashlib
import json


@dataclass
class StepDef:
    """Step definition: id, handler function, optional prompt."""
    id: str
    label: str
    handler: Callable
    depends_on: list[str] = None
    prompt: Optional[str] = None
    model: str = "claude-3-5-sonnet"

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


# ============================================================================
# STEP DEFINITIONS
# ============================================================================

def render_page(archive_path: str, page_group: tuple, force: bool = False) -> dict:
    """Render page scans to JPEG."""
    # TODO: Call actual stitch_oversized_pages
    return {
        "view_path": f"/tmp/render_{hash(page_group)}_V.jpg",
        "xmp_path": f"/tmp/render_{hash(page_group)}_V.xmp",
        "status": "rendered",
    }


def detect_regions(
    render_page: dict,
    view_dir: str,
    photos_dir: str,
    prompt: str = None,
    model: str = "claude-3-5-sonnet",
) -> dict:
    """Detect photo regions with AI."""
    # TODO: Call actual ai_view_regions.detect_regions
    return {
        "regions": [
            {"x": 100, "y": 50, "w": 400, "h": 300},
            {"x": 520, "y": 50, "w": 400, "h": 300},
        ],
        "status": "detected",
    }


def crop_regions(
    detect_regions: dict,
    render_page: dict,
    view_dir: str,
    photos_dir: str,
) -> dict:
    """Extract detected regions to files."""
    # TODO: Call actual crop_page_regions
    return {
        "crops_written": len(detect_regions.get("regions", [])),
        "status": "cropped",
    }


def ai_index(
    crop_regions: dict,
    render_page: dict,
    view_dir: str,
    photos_dir: str,
    force: bool = False,
) -> dict:
    """Run OCR, captions, GPS, object detection."""
    # TODO: Call actual ai_index_runner
    return {
        "ocr_text": "Sample OCR output...",
        "caption": "A photo album page showing family photos",
        "status": "indexed",
    }


def verify_crops(
    ai_index: dict,
    crop_regions: dict,
    view_dir: str,
    photos_dir: str,
) -> dict:
    """Verify crop metadata."""
    # TODO: Call actual verify_crops logic
    return {
        "verified": True,
        "status": "verified",
    }


# ============================================================================
# PIPELINE STEPS (order matters, or use depends_on for clarity)
# ============================================================================

PIPELINE_STEPS = [
    StepDef(
        id="render",
        label="Render page scans to JPEG",
        handler=render_page,
    ),
    StepDef(
        id="detect_regions",
        label="Detect photo regions",
        handler=detect_regions,
        depends_on=["render"],
        prompt="""Analyze this photo album page image.
Identify distinct photos with bounding boxes.
Return JSON array of regions: [{"x": ..., "y": ..., "w": ..., "h": ...}]""",
    ),
    StepDef(
        id="crop_regions",
        label="Extract detected regions",
        handler=crop_regions,
        depends_on=["detect_regions"],
    ),
    StepDef(
        id="ai_index",
        label="Run OCR, captions, GPS",
        handler=ai_index,
        depends_on=["crop_regions"],
        prompt="""Extract text and generate a caption for this photo.
Return JSON: {"ocr_text": "...", "caption": "..."}""",
    ),
    StepDef(
        id="verify_crops",
        label="Verify crop metadata",
        handler=verify_crops,
        depends_on=["ai_index"],
    ),
]

STEPS_BY_ID = {step.id: step for step in PIPELINE_STEPS}


def get_pipeline_dag() -> dict:
    """Return DAG structure for UI."""
    return {
        "steps": [
            {
                "id": step.id,
                "label": step.label,
                "depends_on": step.depends_on,
                "has_prompt": step.prompt is not None,
            }
            for step in PIPELINE_STEPS
        ],
        "edges": [
            {"from": dep, "to": step.id}
            for step in PIPELINE_STEPS
            for dep in step.depends_on
        ],
    }
