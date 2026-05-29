"""Run the local Docling standard image pipeline for page photo region detection."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_RegionResult = None
_converter_cache: dict[tuple[str, str, bool], Any] = {}

# Docling DocItemLabel values (lowercased) that carry recognized text we keep for OCR.
_TEXT_LABELS = {
    "caption",
    "text",
    "title",
    "section_header",
    "paragraph",
    "list_item",
    "footnote",
    "page_header",
    "page_footer",
    "handwritten_text",
    "reference",
}
# When do_ocr is enabled, render the page at this scale before OCR (sharper line crops).
DEFAULT_DOCLING_OCR_IMAGE_SCALE = 2.0


@dataclass(frozen=True)
class DoclingTextRegion:
    text: str
    label: str
    # x, y, width, height in ORIGINAL image pixels, top-left origin.
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DoclingPipelineResult:
    regions: list
    debug_payload: dict[str, Any]
    # Populated only when run_docling_pipeline is called with do_ocr=True.
    text_regions: list[DoclingTextRegion] = field(default_factory=list)
    ocr_text: str = ""


class DoclingPipelineRuntimeError(RuntimeError):
    def __init__(self, message: str, *, debug_payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.debug_payload = dict(debug_payload or {})


@contextmanager
def _suppress_docling_info_logs():
    logger_names = (
        "docling",
        "docling_core",
        "RapidOCR",
        "rapidocr",
    )
    saved: list[tuple[logging.Logger, int]] = []
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            saved.append((logger, logger.level))
            logger.setLevel(max(logger.level, logging.WARNING))
        yield
    finally:
        for logger, level in saved:
            logger.setLevel(level)


def _get_region_result():
    global _RegionResult
    if _RegionResult is None:
        from .ai_view_regions import RegionResult  # pylint: disable=import-outside-toplevel

        _RegionResult = RegionResult
    return _RegionResult


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "export_to_dict"):
        return value.export_to_dict()  # type: ignore[no-any-return]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")  # type: ignore[no-any-return]
        except TypeError:
            return value.model_dump()  # type: ignore[no-any-return]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _build_engine_options(backend: str, device: str):
    from docling.datamodel.accelerator_options import AcceleratorDevice  # pylint: disable=import-outside-toplevel
    from docling.datamodel.vlm_engine_options import (  # pylint: disable=import-outside-toplevel
        AutoInlineVlmEngineOptions,
        MlxVlmEngineOptions,
        TransformersVlmEngineOptions,
    )

    normalized_backend = str(backend or "").strip().lower() or "auto_inline"
    normalized_device = str(device or "").strip().lower() or "auto"
    if normalized_backend == "auto_inline":
        return AutoInlineVlmEngineOptions()
    if normalized_backend == "transformers":
        engine_device = None if normalized_device == AcceleratorDevice.AUTO.value else normalized_device
        return TransformersVlmEngineOptions(device=engine_device)
    if normalized_backend == "mlx":
        return MlxVlmEngineOptions()
    raise RuntimeError(f"Docling pipeline failed due to: unsupported local backend '{backend}'")


def _build_debug_payload(
    convert_result,
    *,
    pipeline_options,
    preset: str,
    backend: str,
    device: str,
    attempts: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _to_jsonable(convert_result)
    document = getattr(convert_result, "document", None)
    if document is not None:
        payload["document"] = _to_jsonable(document)
    payload["config"] = {
        "pipeline_kind": "standard_image",
        "preset": preset,
        "backend": backend,
        "device": device,
        "attempts": attempts,
        "pipeline_options": _to_jsonable(pipeline_options),
    }
    if errors:
        payload["runtime_errors"] = list(errors)
    return payload


def _get_converter(backend: str, device: str, do_ocr: bool = False):
    cache_key = (backend, device, bool(do_ocr))
    if cache_key in _converter_cache:
        return _converter_cache[cache_key]
    from docling.backend.image_backend import ImageDocumentBackend  # pylint: disable=import-outside-toplevel
    from docling.datamodel.accelerator_options import AcceleratorOptions  # pylint: disable=import-outside-toplevel
    from docling.datamodel.base_models import InputFormat  # pylint: disable=import-outside-toplevel
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # pylint: disable=import-outside-toplevel
    from docling.document_converter import DocumentConverter, PdfFormatOption  # pylint: disable=import-outside-toplevel

    pipeline_kwargs: dict[str, Any] = {
        "do_ocr": bool(do_ocr),
        "accelerator_options": AcceleratorOptions(device=device),
    }
    if do_ocr:
        # Let Docling's standard pipeline detect and recognize text in the same pass.
        pipeline_kwargs["generate_page_images"] = True
        pipeline_kwargs["images_scale"] = DEFAULT_DOCLING_OCR_IMAGE_SCALE
    pipeline_options = PdfPipelineOptions(**pipeline_kwargs)
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ImageDocumentBackend,
            )
        }
    )
    log.info(
        "Docling DocumentConverter initialized (backend=%r, device=%r, do_ocr=%s)",
        backend, device, bool(do_ocr),
    )
    _converter_cache[cache_key] = (converter, pipeline_options)
    return converter, pipeline_options


def _extract_caption_hint(item: Any, doc: Any) -> str:
    if not (hasattr(item, "captions") and item.captions):
        return ""
    caption_ref = item.captions[0]
    if not (hasattr(caption_ref, "cref") and caption_ref.cref):
        return ""
    text_idx_str = caption_ref.cref.split("/")[-1]
    try:
        text_idx = int(text_idx_str)
        if hasattr(doc, "texts") and 0 <= text_idx < len(doc.texts):
            return str(getattr(doc.texts[text_idx], "text", "")).strip()
    except (ValueError, IndexError, AttributeError) as exc:
        log.debug("Failed to resolve caption text for ref %r: %s", caption_ref, exc)
    return ""


def _iter_docling_picture_regions(doc: Any, img_h: int, RegionResult: Any) -> list:
    from docling_core.types.doc import DocItemLabel  # pylint: disable=import-outside-toplevel

    regions = []
    for item, _level in doc.iterate_items():
        if item.label != DocItemLabel.PICTURE:
            continue
        if not item.prov:
            log.debug("Skipping picture item with no provenance: %r", item)
            continue
        prov = item.prov[0]
        page_height = float(img_h)
        if getattr(prov, "page_no", None) in getattr(doc, "pages", {}):
            page_height = float(doc.pages[prov.page_no].size.height)
        bbox = prov.bbox.to_top_left_origin(page_height=page_height)
        left, top, right, bottom = bbox.as_tuple()
        regions.append(RegionResult(
            index=len(regions),
            x=max(0, round(left)),
            y=max(0, round(top)),
            width=max(1, round(right - left)),
            height=max(1, round(bottom - top)),
            caption_hint=_extract_caption_hint(item, doc),
        ))
    return regions


def _bbox_to_pixels(prov: Any, doc: Any, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert a Docling provenance bbox to (x, y, w, h) in original image pixels, top-left origin."""
    page_width = float(img_w)
    page_height = float(img_h)
    page_no = getattr(prov, "page_no", None)
    pages = getattr(doc, "pages", {}) or {}
    if page_no in pages:
        size = pages[page_no].size
        page_width = float(getattr(size, "width", img_w)) or float(img_w)
        page_height = float(getattr(size, "height", img_h)) or float(img_h)
    bbox = prov.bbox.to_top_left_origin(page_height=page_height)
    left, top, right, bottom = bbox.as_tuple()
    scale_x = float(img_w) / page_width if page_width else 1.0
    scale_y = float(img_h) / page_height if page_height else 1.0
    x = max(0, round(min(left, right) * scale_x))
    y = max(0, round(min(top, bottom) * scale_y))
    w = max(1, round(abs(right - left) * scale_x))
    h = max(1, round(abs(bottom - top) * scale_y))
    return (x, y, w, h)


def _iter_docling_text_regions(doc: Any, img_w: int, img_h: int) -> list[DoclingTextRegion]:
    from docling_core.types.doc import DocItemLabel  # pylint: disable=import-outside-toplevel

    del DocItemLabel  # imported to ensure docling_core is present; labels compared as strings
    text_regions: list[DoclingTextRegion] = []
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        label_value = str(getattr(label, "value", label) or "").lower()
        if label_value not in _TEXT_LABELS:
            continue
        if not getattr(item, "prov", None):
            continue
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        bbox = _bbox_to_pixels(item.prov[0], doc, img_w, img_h)
        text_regions.append(DoclingTextRegion(text=text, label=label_value, bbox=bbox))
    # Reading order: top-to-bottom, then left-to-right.
    text_regions.sort(key=lambda region: (region.bbox[1], region.bbox[0]))
    return text_regions


def run_docling_pipeline(
    image_path: str | Path,
    img_w: int,
    img_h: int,
    preset: str,
    *,
    backend: str = "auto_inline",
    device: str = "auto",
    retries: int = 3,
    do_ocr: bool = False,
) -> DoclingPipelineResult:
    """Process an image through the same Docling image path the CLI standard pipeline uses.

    When ``do_ocr`` is True, the same pass also recognizes text; the result carries
    ``text_regions`` (text + pixel bbox per region) and a joined ``ocr_text`` string."""
    RegionResult = _get_region_result()
    source_path = Path(image_path)
    max_attempts = max(1, int(retries))
    errors: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    _build_engine_options(backend, device)
    converter, pipeline_options = _get_converter(backend, device, do_ocr=do_ocr)

    convert_result = None
    regions: list = []
    for attempt_number in range(1, max_attempts + 1):
        try:
            log.info(
                "Running Docling standard image pipeline (preset=%r, backend=%r, device=%r, attempt=%d/%d) on %s",
                preset, backend, device, attempt_number, max_attempts, source_path,
            )
            with _suppress_docling_info_logs():
                convert_result = converter.convert(str(source_path))
        except Exception as exc:
            last_exc = exc
            errors.append({"attempt": attempt_number, "error": str(exc)})
            if attempt_number < max_attempts:
                log.warning("Docling pipeline runtime failure for %s on attempt %d/%d: %s",
                            source_path, attempt_number, max_attempts, exc)
            continue

        regions = _iter_docling_picture_regions(convert_result.document, img_h, RegionResult)
        if regions:
            break
        if attempt_number < max_attempts:
            log.warning("run_docling_pipeline: no picture items found for %s on attempt %d/%d, retrying",
                        source_path, attempt_number, max_attempts)
            errors.append({"attempt": attempt_number, "error": "no_regions"})

    if convert_result is None:
        debug_payload = {
            "config": {"preset": preset, "backend": backend, "device": device,
                       "attempts": max_attempts, "pipeline_options": _to_jsonable(pipeline_options)},
            "runtime_errors": list(errors),
        }
        raise DoclingPipelineRuntimeError(f"Docling pipeline failed due to: {last_exc}",
                                          debug_payload=debug_payload) from last_exc

    if not regions:
        log.warning("run_docling_pipeline: no picture items found in DoclingDocument for %s", image_path)

    text_regions: list[DoclingTextRegion] = []
    ocr_text = ""
    if do_ocr:
        text_regions = _iter_docling_text_regions(convert_result.document, img_w, img_h)
        ocr_text = "\n".join(region.text for region in text_regions).strip()

    debug_payload = _build_debug_payload(
        convert_result, pipeline_options=pipeline_options, preset=preset,
        backend=backend, device=device, attempts=len(errors) + 1, errors=errors,
    )
    debug_payload["region_count"] = len(regions)
    if do_ocr:
        debug_payload["text_region_count"] = len(text_regions)
        debug_payload["text_regions"] = [
            {"text": region.text, "label": region.label, "bbox": list(region.bbox)}
            for region in text_regions
        ]
    return DoclingPipelineResult(
        regions=regions,
        debug_payload=debug_payload,
        text_regions=text_regions,
        ocr_text=ocr_text,
    )
