"""Run the local Docling standard image pipeline for page photo region detection."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_RegionResult = None
_converter_cache: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class DoclingPipelineResult:
    regions: list
    debug_payload: dict[str, Any]


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


def _get_converter(backend: str, device: str):
    cache_key = (backend, device)
    if cache_key in _converter_cache:
        return _converter_cache[cache_key]
    from docling.datamodel.accelerator_options import AcceleratorOptions  # pylint: disable=import-outside-toplevel
    from docling.datamodel.base_models import InputFormat  # pylint: disable=import-outside-toplevel
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # pylint: disable=import-outside-toplevel
    from docling.document_converter import DocumentConverter, PdfFormatOption  # pylint: disable=import-outside-toplevel
    from docling.backend.image_backend import ImageDocumentBackend  # pylint: disable=import-outside-toplevel

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        accelerator_options=AcceleratorOptions(device=device),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ImageDocumentBackend,
            )
        }
    )
    log.info("Docling DocumentConverter initialized (backend=%r, device=%r)", backend, device)
    _converter_cache[cache_key] = (converter, pipeline_options)
    return converter, pipeline_options


def run_docling_pipeline(
    image_path: str | Path,
    img_w: int,
    img_h: int,
    preset: str,
    *,
    backend: str = "auto_inline",
    device: str = "auto",
    retries: int = 3,
) -> DoclingPipelineResult:
    """Process an image through the same Docling image path the CLI standard pipeline uses."""
    from docling_core.types.doc import DocItemLabel  # pylint: disable=import-outside-toplevel

    RegionResult = _get_region_result()
    source_path = Path(image_path)
    max_attempts = max(1, int(retries))
    errors: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    _build_engine_options(backend, device)
    converter, pipeline_options = _get_converter(backend, device)

    convert_result = None
    regions: list = []
    for attempt_number in range(1, max_attempts + 1):
        try:
            log.info(
                "Running Docling standard image pipeline (preset=%r, backend=%r, device=%r, attempt=%d/%d) on %s",
                preset,
                backend,
                device,
                attempt_number,
                max_attempts,
                source_path,
            )
            with _suppress_docling_info_logs():
                convert_result = converter.convert(str(source_path))
        except Exception as exc:
            last_exc = exc
            errors.append({"attempt": attempt_number, "error": str(exc)})
            if attempt_number < max_attempts:
                log.warning(
                    "Docling pipeline runtime failure for %s on attempt %d/%d: %s",
                    source_path,
                    attempt_number,
                    max_attempts,
                    exc,
                )
            continue

        doc = convert_result.document
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
            regions.append(
                RegionResult(
                    index=len(regions),
                    x=max(0, round(left)),
                    y=max(0, round(top)),
                    width=max(1, round(right - left)),
                    height=max(1, round(bottom - top)),
                )
            )

        if regions:
            break
        if attempt_number < max_attempts:
            log.warning(
                "run_docling_pipeline: no picture items found for %s on attempt %d/%d, retrying",
                source_path,
                attempt_number,
                max_attempts,
            )
            errors.append({"attempt": attempt_number, "error": "no_regions"})

    if convert_result is None:
        debug_payload = {
            "config": {
                "preset": preset,
                "backend": backend,
                "device": device,
                "attempts": max_attempts,
                "pipeline_options": _to_jsonable(pipeline_options),
            },
            "runtime_errors": list(errors),
        }
        message = f"Docling pipeline failed due to: {last_exc}"
        raise DoclingPipelineRuntimeError(message, debug_payload=debug_payload) from last_exc

    if not regions:
        log.warning("run_docling_pipeline: no picture items found in DoclingDocument for %s", image_path)

    debug_payload = _build_debug_payload(
        convert_result,
        pipeline_options=pipeline_options,
        preset=preset,
        backend=backend,
        device=device,
        attempts=len(errors) + 1,
        errors=errors,
    )
    debug_payload["region_count"] = len(regions)
    return DoclingPipelineResult(regions=regions, debug_payload=debug_payload)
