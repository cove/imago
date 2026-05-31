"""Thin adapter exposing Docling OCR text for the ``docling`` OCR engine.

This deliberately does NOT build its own Docling converter or pipeline. Per project
direction we use Docling's own standard pipeline as-is (the same one the detect-regions
step runs) rather than deconstructing it. All OCR work happens in
``_docling_pipeline.run_docling_pipeline`` with ``do_ocr=True``; this module just
requests that pass and returns the text it recognized, along with Docling's native
text regions (text + pixel bbox) for downstream overlay/association use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ai_model_settings import (
    default_docling_backend,
    default_docling_device,
    default_docling_preset,
    default_docling_retries,
)
from .image_limits import get_image_dimensions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DoclingOcrResult:
    text: str
    text_regions: list = field(default_factory=list)
    picture_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    debug_payload: dict[str, Any] = field(default_factory=dict)


def run_docling_ocr(image_path: str | Path, img_w: int = 0, img_h: int = 0) -> DoclingOcrResult:
    """Run Docling's standard pipeline with OCR enabled and return recognized text."""
    from ._docling_pipeline import run_docling_pipeline  # pylint: disable=import-outside-toplevel

    path = Path(image_path)
    if not img_w or not img_h:
        img_w, img_h = get_image_dimensions(path)

    pipeline_result = run_docling_pipeline(
        path,
        img_w=img_w,
        img_h=img_h,
        preset=default_docling_preset(),
        backend=default_docling_backend(),
        device=default_docling_device(),
        retries=default_docling_retries(),
        do_ocr=True,
    )
    picture_boxes = [
        (region.x, region.y, region.width, region.height) for region in pipeline_result.regions
    ]
    return DoclingOcrResult(
        text=pipeline_result.ocr_text,
        text_regions=list(pipeline_result.text_regions),
        picture_boxes=picture_boxes,
        debug_payload=dict(pipeline_result.debug_payload),
    )
