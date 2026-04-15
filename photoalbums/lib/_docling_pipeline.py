"""Run the Docling VLM pipeline for photo region detection.

Uses Docling's DocumentConverter with VlmPipeline configured from the selected
preset. Converts DoclingDocument picture items into RegionResult objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Lazy import to avoid pulling in heavy docling deps at module load time
_RegionResult = None


def _get_region_result():
    global _RegionResult
    if _RegionResult is None:
        from .ai_view_regions import RegionResult  # pylint: disable=import-outside-toplevel

        _RegionResult = RegionResult
    return _RegionResult


def _resolve_caption(doc, picture_item) -> str:
    """Return the text of the first caption associated with the picture, or ''."""
    for ref in picture_item.captions:
        cref = str(ref.cref or "")
        # cref format: '#/texts/3' or '#/pictures/0' etc.
        parts = cref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] == "texts":
            try:
                idx = int(parts[1])
                text_item = doc.texts[idx]
                text = str(text_item.text or "").strip()
                if text:
                    return text
            except (IndexError, ValueError):
                pass
    return ""


def run_docling_pipeline(
    image_path: str | Path,
    img_w: int,
    img_h: int,
    preset: str,
) -> list:
    """Process an image through Docling and return RegionResult objects."""
    from docling.datamodel.base_models import InputFormat  # pylint: disable=import-outside-toplevel
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions  # pylint: disable=import-outside-toplevel
    from docling.document_converter import DocumentConverter, ImageFormatOption  # pylint: disable=import-outside-toplevel
    from docling.pipeline.vlm_pipeline import VlmPipeline  # pylint: disable=import-outside-toplevel
    from docling_core.types.doc import DocItemLabel  # pylint: disable=import-outside-toplevel

    RegionResult = _get_region_result()

    vlm_options = VlmConvertOptions.from_preset(preset)
    pipeline_options = VlmPipelineOptions(vlm_options=vlm_options)
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    source_path = Path(image_path)
    log.info("Running Docling VLM pipeline (preset=%r) on %s", preset, image_path)
    result = converter.convert(str(source_path))
    doc = result.document

    regions: list = []
    for item, _level in doc.iterate_items():
        if item.label != DocItemLabel.PICTURE:
            continue
        if not item.prov:
            log.debug("Skipping picture item with no provenance: %r", item)
            continue

        bbox = item.prov[0].bbox
        x = round(bbox.l * img_w)
        y = round(bbox.t * img_h)
        width = max(1, round((bbox.r - bbox.l) * img_w))
        height = max(1, round((bbox.b - bbox.t) * img_h))

        caption_hint = _resolve_caption(doc, item)

        regions.append(
            RegionResult(
                index=len(regions),
                x=x,
                y=y,
                width=width,
                height=height,
                caption_hint=caption_hint,
            )
        )

    if not regions:
        log.warning("run_docling_pipeline: no picture items found in DoclingDocument for %s", image_path)

    return regions
