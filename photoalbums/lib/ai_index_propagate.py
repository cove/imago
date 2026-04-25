"""propagate-to-crops step: push GPS and person names into crop XMP sidecars."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .xmp_sidecar import (
    _dedupe,
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    read_region_list,
    write_pipeline_steps,
    write_xmp_sidecar,
    xmp_datetime_now,
)
from .metadata_resolver import resolve_crop_location, resolve_person_in_image
from .metadata_resolver import resolve_crop_locations_shown


def _crop_paths_signature(crop_paths: list[Path]) -> str:
    combined = "|".join(str(p) for p in sorted(str(p) for p in crop_paths))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _get_image_dimensions_safe(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image as _PILImage
        from .image_limits import allow_large_pillow_images
        allow_large_pillow_images(_PILImage)
        with _PILImage.open(str(image_path)) as img:
            return img.size
    except Exception:
        return 0, 0


def _find_crop_paths_for_page(image_path: Path) -> list[Path]:
    """Return existing crop file paths for a page image. Returns [] if not a pages-dir page."""
    from ..naming import is_pages_dir, photos_dir_for_album_dir

    if not is_pages_dir(image_path.parent):
        return []

    photos_dir = photos_dir_for_album_dir(image_path.parent)
    if not photos_dir.is_dir():
        return []

    sidecar_path = image_path.with_suffix(".xmp")
    if not sidecar_path.is_file():
        return []

    img_w, img_h = _get_image_dimensions_safe(image_path)
    if img_w <= 0 or img_h <= 0:
        return []

    try:
        regions = read_region_list(sidecar_path, img_w, img_h)
    except Exception:
        return []

    if not regions:
        return []

    from .ai_photo_crops import _expected_crop_output_paths

    candidates = _expected_crop_output_paths(image_path, photos_dir, len(regions))
    return [p for p in candidates if p.is_file()]


def _read_regions_safe(sidecar_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not sidecar_path.is_file():
        return []
    try:
        return read_region_list(sidecar_path, img_w, img_h)
    except Exception:
        return []


def _region_caption(region_state: dict) -> str:
    return str(region_state.get("caption_hint") or region_state.get("caption") or "")


def _resolve_crop_metadata(
    region_state: dict,
    locations_shown: list,
    page_location: dict[str, Any],
    names_from_region: list[str],
    existing_person_names: list[str],
) -> tuple[dict, list, list[str]]:
    region_override = dict(region_state.get("location_override") or {})
    region_assigned = dict(region_state.get("location_payload") or {})
    caption = _region_caption(region_state)
    crop_location = resolve_crop_location(
        region_location_override=region_override,
        region_location_assigned=region_assigned,
        caption=caption,
        locations_shown=locations_shown,
        page_location=page_location,
    )
    crop_locations_shown = resolve_crop_locations_shown(
        region_location_override=region_override,
        region_location_assigned=region_assigned,
        caption=caption,
        locations_shown=locations_shown,
    )
    new_person_names = resolve_person_in_image(
        _dedupe(names_from_region + existing_person_names),
        locations_shown=locations_shown,
        location_payload=crop_location,
    )
    return crop_location, crop_locations_shown, new_person_names


def _build_detections_payload(
    existing_state: dict, crop_location: dict, step_timestamp: str
) -> dict:
    existing_detections = dict(existing_state.get("detections") or {})
    if crop_location:
        existing_detections["location"] = crop_location
    existing_pipeline = dict(existing_detections.get("pipeline") or {})
    existing_pipeline["ai-index/propagate-to-crops"] = {
        "timestamp": step_timestamp,
        "input_hash": "",
        "result": "ok",
    }
    existing_detections["pipeline"] = existing_pipeline
    return existing_detections


def _str_field(d: dict, key: str) -> str:
    return str(d.get(key) or "")


def _write_propagated_crop(
    crop_xmp: Path,
    existing_state: dict,
    crop_location: dict,
    crop_locations_shown: list,
    new_person_names: list[str],
    step_timestamp: str,
) -> None:
    detections_payload = _build_detections_payload(existing_state, crop_location, step_timestamp)
    write_xmp_sidecar(
        crop_xmp,
        person_names=new_person_names,
        subjects=list(existing_state.get("subjects") or []),
        title=_str_field(existing_state, "title"),
        title_source=_str_field(existing_state, "title_source"),
        description=_str_field(existing_state, "description"),
        ocr_text=_str_field(existing_state, "ocr_text"),
        parent_ocr_text=_str_field(existing_state, "parent_ocr_text"),
        ocr_lang=_str_field(existing_state, "ocr_lang"),
        author_text=_str_field(existing_state, "author_text"),
        scene_text=_str_field(existing_state, "scene_text"),
        album_title=_str_field(existing_state, "album_title"),
        gps_latitude=_str_field(crop_location, "gps_latitude").strip(),
        gps_longitude=_str_field(crop_location, "gps_longitude").strip(),
        location_city=_str_field(crop_location, "city").strip(),
        location_state=_str_field(crop_location, "state").strip(),
        location_country=_str_field(crop_location, "country").strip(),
        location_sublocation=_str_field(crop_location, "sublocation").strip(),
        locations_shown=crop_locations_shown,
        source_text=_str_field(existing_state, "source_text"),
        detections_payload=detections_payload,
        create_date=_str_field(existing_state, "create_date"),
        dc_date=list(existing_state.get("dc_date_values") or []),
        date_time_original=_str_field(existing_state, "date_time_original"),
        ocr_ran=bool(existing_state.get("ocr_ran", False)),
        people_detected=bool(new_person_names),
        people_identified=bool(new_person_names),
    )


def _propagate_one_crop(
    crop_xmp: Path,
    region_state: dict,
    names_from_region: list[str],
    locations_shown: list,
    page_location: dict[str, Any],
    step_timestamp: str,
) -> bool:
    if not crop_xmp.is_file():
        return False
    existing_state = read_ai_sidecar_state(crop_xmp)
    if not isinstance(existing_state, dict):
        return False
    existing_person_names = read_person_in_image(crop_xmp)
    crop_location, crop_locations_shown, new_person_names = _resolve_crop_metadata(
        region_state,
        locations_shown,
        page_location,
        names_from_region,
        existing_person_names,
    )
    _write_propagated_crop(
        crop_xmp,
        existing_state,
        crop_location,
        crop_locations_shown,
        new_person_names,
        step_timestamp,
    )
    return True


def run_propagate_to_crops(
    image_path: Path,
    *,
    location_payload: dict[str, Any],
    people_payload: list[dict[str, Any]],
) -> dict[str, Any]:
    """Propagate location GPS and person names from page XMP to each crop XMP.

    Returns a dict with a 'crops_updated' count (for diagnostic use).
    Eligible to return None if engine not configured — but this step always runs.
    """
    crop_paths = _find_crop_paths_for_page(image_path)
    if not crop_paths:
        return {"crops_updated": 0}

    sidecar_path = image_path.with_suffix(".xmp")
    img_w, img_h = _get_image_dimensions_safe(image_path)
    regions = _read_regions_safe(sidecar_path, img_w, img_h)
    region_person_names: list[list[str]] = [list(r.get("person_names") or []) for r in regions]
    locations_shown = read_locations_shown(sidecar_path)
    step_timestamp = xmp_datetime_now()

    crops_updated = 0
    for i, crop_path in enumerate(crop_paths):
        names_from_region = region_person_names[i] if i < len(region_person_names) else []
        region_state = regions[i] if i < len(regions) else {}
        if _propagate_one_crop(
            crop_path.with_suffix(".xmp"),
            region_state,
            names_from_region,
            locations_shown,
            location_payload,
            step_timestamp,
        ):
            crops_updated += 1

    return {"crops_updated": crops_updated}
