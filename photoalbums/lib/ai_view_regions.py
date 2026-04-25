"""Detect page photo regions with Docling and persist them via XMP."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .ai_model_settings import (
    default_docling_backend,
    default_docling_device,
    default_docling_preset,
    default_docling_retries,
    default_caption_matching_model,
    default_lmstudio_base_url,
    default_view_region_model,
)
from .prompt_debug import debug_root_for_image_path

if TYPE_CHECKING:
    from .prompt_debug import PromptDebugSession

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
_MAX_SINGLE_REGION_PAGE_FRACTION = 0.90


@dataclass(frozen=True)
class RegionResult:
    index: int
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    caption_hint: str = ""
    location_hint: str = ""
    location_payload: dict[str, object] = field(default_factory=dict)
    person_names: list[str] = field(default_factory=list)


@dataclass
class RegionWithCaption:
    region: RegionResult
    caption: str
    caption_ambiguous: bool = False


@dataclass(frozen=True)
class RegionFailure:
    region_index: int
    reason: str
    severity: str
    overlap_with: int | None = None
    overlap_fraction: float | None = None
    page_fraction: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    kept: list[RegionResult]
    failures: list[RegionFailure]


def pixel_to_mwgrs(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def associate_captions(
    regions: list[RegionResult],
    captions: list[dict],
    img_width: int,
) -> list[RegionWithCaption]:
    ambiguity_threshold = img_width * 0.10

    def region_centre(region: RegionResult) -> tuple[float, float]:
        return region.x + region.width / 2.0, region.y + region.height / 2.0

    def caption_centre(caption: dict) -> tuple[float, float] | None:
        try:
            cx = float(caption["x"]) + float(caption["w"]) / 2.0
            cy = float(caption["y"]) + float(caption["h"]) / 2.0
            return cx, cy
        except (KeyError, TypeError, ValueError):
            return None

    def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5

    results: list[RegionWithCaption] = []
    for caption in captions:
        centre = caption_centre(caption)
        if centre is None:
            text = str(caption.get("text") or "").strip()
            return [RegionWithCaption(region, text, caption_ambiguous=True) for region in regions]

        distances = [distance(region_centre(region), centre) for region in regions]
        if len(distances) < 2:
            best_idx = 0
        else:
            sorted_distances = sorted(distances)
            if sorted_distances[1] - sorted_distances[0] < ambiguity_threshold:
                text = str(caption.get("text") or "").strip()
                return [RegionWithCaption(region, text, caption_ambiguous=True) for region in regions]
            best_idx = distances.index(sorted_distances[0])
        results.append(RegionWithCaption(regions[best_idx], str(caption.get("text") or "").strip()))

    assigned = {row.region.index for row in results}
    for region in regions:
        if region.index not in assigned:
            results.append(RegionWithCaption(region, ""))
    results.sort(key=lambda row: row.region.index)
    return results
def validate_region_set(
    regions: list[RegionResult],
    *,
    img_w: int,
    img_h: int,
) -> ValidationResult:
    img_area = img_w * img_h
    failures: list[RegionFailure] = []
    clamped: list[RegionResult] = []

    for region in regions:
        left = max(0, region.x)
        top = max(0, region.y)
        right = min(img_w, region.x + region.width)
        bottom = min(img_h, region.y + region.height)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            failures.append(RegionFailure(region_index=region.index, reason="zero_area", severity="hard"))
            continue
        if img_area > 0 and (width * height) / img_area >= _MAX_SINGLE_REGION_PAGE_FRACTION:
            failures.append(RegionFailure(region_index=region.index, reason="full_page", severity="hard"))
            continue
        clamped.append(replace(region, x=left, y=top, width=width, height=height))

    return ValidationResult(valid=len(failures) == 0, kept=sorted(clamped, key=lambda region: region.index), failures=failures)


def validate_regions_for_write(
    regions: list[RegionResult],
    *,
    img_w: int,
    img_h: int,
) -> list[RegionResult]:
    return validate_region_set(regions, img_w=img_w, img_h=img_h).kept


def _xmp_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".xmp")


def _has_xmp_regions(xmp_path: Path) -> bool:
    if not xmp_path.is_file():
        return False
    try:
        return "mwg-rs:RegionList" in xmp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _read_regions_from_xmp(xmp_path: Path, img_w: int, img_h: int) -> list[RegionResult]:
    import xml.etree.ElementTree as ET  # pylint: disable=import-outside-toplevel

    MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
    STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    DC_NS = "http://purl.org/dc/elements/1.1/"
    IMAGO_NS = "https://imago.local/ns/1.0/"

    try:
        tree = ET.parse(str(xmp_path))
    except ET.ParseError:
        return []

    results: list[RegionResult] = []
    for li in tree.iter(f"{{{RDF_NS}}}li"):
        cx_t = li.get(f"{{{STAREA_NS}}}x")
        cy_t = li.get(f"{{{STAREA_NS}}}y")
        nw_t = li.get(f"{{{STAREA_NS}}}w")
        nh_t = li.get(f"{{{STAREA_NS}}}h")
        if not all((cx_t, cy_t, nw_t, nh_t)):
            continue
        try:
            cx = float(cx_t)
            cy = float(cy_t)
            nw = float(nw_t)
            nh = float(nh_t)
        except (TypeError, ValueError):
            continue
        px = max(0, int(round((cx - nw / 2.0) * img_w)))
        py = max(0, int(round((cy - nh / 2.0) * img_h)))
        pw = max(1, int(round(nw * img_w)))
        ph = max(1, int(round(nh * img_h)))
        caption = str(li.get(f"{{{MWGRS_NS}}}Name") or "").strip()
        if not caption:
            desc_el = li.find(f".//{{{DC_NS}}}description")
            if desc_el is not None:
                text_el = desc_el.find(f".//{{{RDF_NS}}}li")
                if text_el is not None and text_el.text:
                    caption = text_el.text.strip()
        if not caption:
            caption = str(li.get(f"{{{IMAGO_NS}}}CaptionHint") or "").strip()
        results.append(
            RegionResult(
                index=len(results),
                x=px,
                y=py,
                width=pw,
                height=ph,
                caption_hint=caption,
            )
        )
    return results


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    with Image.open(str(image_path)) as img:
        return img.size


def _failed_regions_debug_path(image_path: str | Path, attempt_number: int | None = None) -> Path:
    path = Path(image_path)
    filename = f"{path.stem}.view-regions.failed-boxes.jpg"
    if attempt_number is not None:
        filename = f"{path.stem}.view-regions.failed-boxes.attempt-{attempt_number:02d}.jpg"
    return debug_root_for_image_path(path) / filename


def _accepted_regions_debug_path(image_path: str | Path, attempt_number: int | None = None) -> Path:
    path = Path(image_path)
    filename = f"{path.stem}.view-regions.accepted-boxes.jpg"
    if attempt_number is not None:
        filename = f"{path.stem}.view-regions.accepted-boxes.attempt-{attempt_number:02d}.jpg"
    return debug_root_for_image_path(path) / filename


def _region_association_overlay_path(image_path: str | Path, attempt_number: int | None = None) -> Path:
    path = Path(image_path)
    filename = f"{path.stem}.view-regions.association-overlay.jpg"
    if attempt_number is not None:
        filename = f"{path.stem}.view-regions.association-overlay.attempt-{attempt_number:02d}.jpg"
    return debug_root_for_image_path(path) / filename


def _docling_raw_debug_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    return debug_root_for_image_path(path) / f"{path.stem}.view-regions.docling.json"


def _failed_regions_debug_paths(image_path: str | Path) -> list[Path]:
    path = Path(image_path)
    return list(debug_root_for_image_path(path).glob(f"{path.stem}.view-regions.failed-boxes*.jpg"))


def _accepted_regions_debug_paths(image_path: str | Path) -> list[Path]:
    path = Path(image_path)
    return list(debug_root_for_image_path(path).glob(f"{path.stem}.view-regions.accepted-boxes*.jpg"))


def _region_association_overlay_paths(image_path: str | Path) -> list[Path]:
    path = Path(image_path)
    return list(debug_root_for_image_path(path).glob(f"{path.stem}.view-regions.association-overlay*.jpg"))


def _clear_failed_regions_debug_image(image_path: str | Path) -> None:
    for debug_path in _failed_regions_debug_paths(image_path):
        try:
            if debug_path.is_file():
                debug_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove failed-region debug image %s: %s", debug_path, exc)


def _clear_accepted_regions_debug_image(image_path: str | Path) -> None:
    for debug_path in _accepted_regions_debug_paths(image_path):
        try:
            if debug_path.is_file():
                debug_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove accepted-region debug image %s: %s", debug_path, exc)


def _clear_region_association_overlay_image(image_path: str | Path) -> None:
    for debug_path in _region_association_overlay_paths(image_path):
        try:
            if debug_path.is_file():
                debug_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove region-association overlay image %s: %s", debug_path, exc)


def _clear_regions_debug_images(image_path: str | Path) -> None:
    _clear_failed_regions_debug_image(image_path)
    _clear_accepted_regions_debug_image(image_path)
    _clear_region_association_overlay_image(image_path)


def _write_accepted_regions_debug_image(
    image_path: str | Path,
    regions: list[RegionResult],
    *,
    attempt_number: int | None = None,
) -> Path | None:
    from .ai_view_region_render import render_regions_debug  # pylint: disable=import-outside-toplevel

    overlay_regions: list[dict[str, object]] = []
    for region in sorted(regions, key=lambda item: item.index):
        label_parts: list[str] = []
        caption_hint = str(region.caption_hint or "").strip()
        if caption_hint:
            label_parts.append(caption_hint)
        if region.person_names:
            label_parts.append(", ".join(str(name).strip() for name in region.person_names if str(name).strip()))
        overlay_regions.append(
            {
                "index": region.index,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "caption": " | ".join(label_parts),
            }
        )
    output_path = _accepted_regions_debug_path(image_path, attempt_number=attempt_number)
    try:
        render_regions_debug(image_path, overlay_regions, output_path)
    except Exception as exc:
        log.warning("Failed to write accepted-region debug image %s: %s", output_path, exc)
        return None
    return output_path


def _write_region_association_overlay_image(
    image_path: str | Path,
    regions: list[RegionResult],
    *,
    attempt_number: int | None = None,
) -> Path | None:
    from .ai_view_region_render import render_regions_overlay  # pylint: disable=import-outside-toplevel

    overlay_regions = [
        {
            "index": region.index,
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height,
        }
        for region in sorted(regions, key=lambda item: item.index)
    ]
    output_path = _region_association_overlay_path(image_path, attempt_number=attempt_number)
    try:
        render_regions_overlay(image_path, overlay_regions, output_path)
    except Exception as exc:
        log.warning("Failed to write region-association overlay image %s: %s", output_path, exc)
        return None
    return output_path


def _write_failed_regions_debug_image(
    image_path: str | Path,
    regions: list[RegionResult],
    failures: list[RegionFailure],
    *,
    attempt_number: int | None = None,
) -> Path | None:
    if not regions:
        return None

    from .ai_view_region_render import render_regions_debug  # pylint: disable=import-outside-toplevel

    page_level_notes: list[str] = []
    notes_by_index: dict[int, list[str]] = {}
    for failure in failures:
        if failure.reason == "overlap":
            other = f" with #{failure.overlap_with + 1}" if failure.overlap_with is not None else ""
            note = f"overlap{other}"
        elif failure.reason == "zero_area":
            note = "zero area"
        elif failure.reason == "full_page":
            note = "covers >=90% of page"
        else:
            note = failure.reason.replace("_", " ")
        if failure.region_index >= 0:
            notes_by_index.setdefault(failure.region_index, []).append(note)
        else:
            page_level_notes.append(note)

    overlay_regions: list[dict[str, object]] = []
    for idx, region in enumerate(sorted(regions, key=lambda item: item.index)):
        notes = list(notes_by_index.get(region.index, []))
        if idx == 0 and page_level_notes:
            notes = page_level_notes + notes
        overlay_regions.append(
            {
                "index": region.index,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "caption": "; ".join(notes),
            }
        )

    output_path = _failed_regions_debug_path(image_path, attempt_number=attempt_number)
    try:
        render_regions_debug(image_path, overlay_regions, output_path)
    except Exception as exc:
        log.warning("Failed to write failed-region debug image %s: %s", output_path, exc)
        return None
    return output_path


def _write_docling_raw_debug_artifact(image_path: str | Path, payload: dict[str, object]) -> Path | None:
    output_path = _docling_raw_debug_path(image_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("Failed to write Docling debug artifact %s: %s", output_path, exc)
        return None
    return output_path


def _apply_lmstudio_captions(
    regions: list[RegionResult],
    image_path: Path,
    base_url: str,
) -> list[RegionResult]:
    from ._caption_matching import (  # pylint: disable=import-outside-toplevel
        assign_captions_from_lmstudio,
        call_lmstudio_caption_matching,
    )
    from .ai_geocode import NominatimGeocoder  # pylint: disable=import-outside-toplevel
    from .ai_index_scan import _page_scan_filenames  # pylint: disable=import-outside-toplevel
    from .ai_location import _resolve_location_payload  # pylint: disable=import-outside-toplevel
    from .ai_render_settings import find_archive_dir_for_image  # pylint: disable=import-outside-toplevel
    from .xmp_sidecar import read_locations_shown  # pylint: disable=import-outside-toplevel

    caption_model = default_caption_matching_model()
    overlay_path = _write_region_association_overlay_image(image_path, regions)
    if not caption_model:
        log.debug("LM Studio caption matching skipped: caption_matching_model not configured in ai_models.toml")
        return regions

    locations_shown: list[dict] = []
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is not None and archive_dir.is_dir():
        for scan_name in _page_scan_filenames(image_path):
            locations_shown = read_locations_shown((archive_dir / scan_name).with_suffix(".xmp"))
            if locations_shown:
                break

    captions = call_lmstudio_caption_matching(
        overlay_path or image_path,
        base_url=base_url or default_lmstudio_base_url(),
        model=caption_model,
        locations_shown=locations_shown,
    )
    if not captions:
        return regions
    matched_regions = assign_captions_from_lmstudio(regions, captions)
    if len(locations_shown) < 2:
        return matched_regions

    geocoder = NominatimGeocoder()
    resolved_regions: list[RegionResult] = []
    for region in matched_regions:
        location_text = str(region.location_hint or "").strip()
        location_payload = (
            _resolve_location_payload(
                geocoder=geocoder,
                gps_latitude="",
                gps_longitude="",
                location_name=location_text,
            )
            if location_text
            else {}
        )
        resolved_regions.append(replace(region, location_payload=location_payload))
    return resolved_regions


def _detect_regions_docling(
    path: Path,
    *,
    xmp_path: Path,
    model: str,
    img_w: int,
    img_h: int,
    force: bool,
    base_url: str = "",
    prompt_debug: PromptDebugSession | None,
    skip_validation: bool,
    write_debug: bool = False,
) -> list[RegionResult]:
    from ._docling_pipeline import (  # pylint: disable=import-outside-toplevel
        DoclingPipelineRuntimeError,
        run_docling_pipeline,
    )
    from .xmp_sidecar import read_pipeline_step, write_pipeline_steps, xmp_datetime_now  # pylint: disable=import-outside-toplevel

    _STEP_KEY = "detect-regions/docling"

    def _write_step(result: str) -> None:
        write_pipeline_steps(xmp_path, {_STEP_KEY: {"timestamp": xmp_datetime_now(), "result": result, "input_hash": "", "model": model}})

    if not force:
        existing_step = read_pipeline_step(xmp_path, _STEP_KEY) or read_pipeline_step(xmp_path, "view_regions") or {}
        existing_result = str(existing_step.get("result") or "").strip()
        if existing_result in {"no_regions", "validation_failed", "failed"}:
            log.info("Skipping Docling detection for %s: pipeline step already recorded result=%r", path, existing_result)
            return []

    try:
        pipeline_result = run_docling_pipeline(
            path,
            img_w=img_w,
            img_h=img_h,
            preset=default_docling_preset(),
            backend=default_docling_backend(),
            device=default_docling_device(),
            retries=default_docling_retries(),
        )
    except DoclingPipelineRuntimeError as exc:
        if prompt_debug is not None and exc.debug_payload:
            _write_docling_raw_debug_artifact(path, exc.debug_payload)
        _write_step("failed")
        log.error("Docling pipeline failed for %s: %s", path, exc)
        return []
    except Exception as exc:
        _write_step("failed")
        log.error("Docling pipeline failed for %s: %s", path, exc)
        return []

    if prompt_debug is not None and pipeline_result.debug_payload:
        _write_docling_raw_debug_artifact(path, pipeline_result.debug_payload)

    if not pipeline_result.regions:
        _write_step("no_regions")
        log.info("Docling: no regions detected for %s", path)
        return []

    if skip_validation:
        final_regions = _apply_lmstudio_captions(pipeline_result.regions, path, base_url)
        _write_step("regions_found")
        if write_debug:
            _write_accepted_regions_debug_image(path, final_regions)
        return final_regions

    validation = validate_region_set(pipeline_result.regions, img_w=img_w, img_h=img_h)
    if validation.failures:
        _write_step("validation_failed")
        if write_debug:
            _write_failed_regions_debug_image(path, pipeline_result.regions, validation.failures)
        reasons = ", ".join(failure.reason for failure in validation.failures)
        log.error(
            "Docling: region validation failed for %s (%d failure(s): %s); use --force to retry",
            path,
            len(validation.failures),
            reasons,
        )
        return []

    final_regions = _apply_lmstudio_captions(validation.kept, path, base_url)
    _write_step("regions_found")
    if write_debug:
        _write_accepted_regions_debug_image(path, final_regions)
    return final_regions


def detect_regions(
    image_path: str | Path,
    *,
    force: bool = False,
    model: str = "",
    base_url: str = "",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    album_context: str = "",
    page_caption: str = "",
    people_roster: dict[str, str] | None = None,
    prompt_debug: PromptDebugSession | None = None,
    skip_validation: bool = False,
    write_debug: bool = False,
) -> list[RegionResult]:
    del timeout, album_context, page_caption, people_roster

    path = Path(image_path)
    xmp_path = _xmp_path_for(path)

    if not force and _has_xmp_regions(xmp_path):
        try:
            img_w, img_h = _image_dimensions(path)
            cached = _read_regions_from_xmp(xmp_path, img_w, img_h)
            if cached:
                if write_debug:
                    _clear_regions_debug_images(path)
                _write_accepted_regions_debug_image(path, cached)
                _write_region_association_overlay_image(path, cached)
                return cached
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to read cached XMP regions for %s: %s", path, exc)

    resolved_model = str(model or "").strip() or str(default_view_region_model() or "").strip()
    if "docling" not in resolved_model.lower():
        raise RuntimeError(f"View region detection failed due to: non-Docling model configured for regions: {resolved_model}")

    img_w, img_h = _image_dimensions(path)
    if write_debug:
        _clear_regions_debug_images(path)
    return _detect_regions_docling(
        path,
        xmp_path=xmp_path,
        model=resolved_model,
        img_w=img_w,
        img_h=img_h,
        force=force,
        base_url=str(base_url or "").strip(),
        prompt_debug=prompt_debug,
        skip_validation=skip_validation,
        write_debug=write_debug,
    )
