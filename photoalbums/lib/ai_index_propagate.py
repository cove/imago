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
    sidecar_path = image_path.with_suffix(".xmp")
    img_w, img_h = _get_image_dimensions_safe(image_path)

    crop_paths = _find_crop_paths_for_page(image_path)
    if not crop_paths:
        return {"crops_updated": 0}

    # Build person names keyed by region index from MWG-RS region names
    regions: list[dict] = []
    try:
        regions = read_region_list(sidecar_path, img_w, img_h) if sidecar_path.is_file() else []
    except Exception:
        regions = []

    # Map region index → person names from imago:PersonNames (face recognition results)
    region_person_names: list[list[str]] = [list(r.get("person_names") or []) for r in regions]
    locations_shown = read_locations_shown(sidecar_path)

    step_timestamp = xmp_datetime_now()
    crops_updated = 0

    for i, crop_path in enumerate(crop_paths):
        crop_xmp = crop_path.with_suffix(".xmp")
        if not crop_xmp.is_file():
            continue

        existing_state = read_ai_sidecar_state(crop_xmp)
        if not isinstance(existing_state, dict):
            continue

        # Person names for this crop from imago:PersonNames on the matching region
        names_from_region = region_person_names[i] if i < len(region_person_names) else []
        existing_person_names = read_person_in_image(crop_xmp)
        region_state = regions[i] if i < len(regions) else {}
        crop_location = resolve_crop_location(
            region_location_override=dict(region_state.get("location_override") or {}),
            region_location_assigned=dict(region_state.get("location_payload") or {}),
            caption=str(region_state.get("caption_hint") or region_state.get("caption") or ""),
            locations_shown=locations_shown,
            page_location=location_payload,
        )
        crop_locations_shown = resolve_crop_locations_shown(
            region_location_override=dict(region_state.get("location_override") or {}),
            region_location_assigned=dict(region_state.get("location_payload") or {}),
            caption=str(region_state.get("caption_hint") or region_state.get("caption") or ""),
            locations_shown=locations_shown,
        )
        new_person_names = resolve_person_in_image(
            _dedupe(names_from_region + existing_person_names),
            locations_shown=locations_shown,
            location_payload=crop_location,
        )

        # Existing detections payload — update location
        existing_detections = dict(existing_state.get("detections") or {})
        if crop_location:
            existing_detections["location"] = crop_location

        # Pipeline record for this crop
        existing_pipeline = dict(existing_detections.get("pipeline") or {})
        existing_pipeline["ai-index/propagate-to-crops"] = {
            "timestamp": step_timestamp,
            "input_hash": "",
            "result": "ok",
        }
        existing_detections["pipeline"] = existing_pipeline

        write_xmp_sidecar(
            crop_xmp,
            person_names=new_person_names,
            subjects=list(existing_state.get("subjects") or []),
            title=str(existing_state.get("title") or ""),
            title_source=str(existing_state.get("title_source") or ""),
            description=str(existing_state.get("description") or ""),
            ocr_text=str(existing_state.get("ocr_text") or ""),
            parent_ocr_text=str(existing_state.get("parent_ocr_text") or ""),
            ocr_lang=str(existing_state.get("ocr_lang") or ""),
            author_text=str(existing_state.get("author_text") or ""),
            scene_text=str(existing_state.get("scene_text") or ""),
            album_title=str(existing_state.get("album_title") or ""),
            gps_latitude=str(crop_location.get("gps_latitude") or "").strip(),
            gps_longitude=str(crop_location.get("gps_longitude") or "").strip(),
            location_city=str(crop_location.get("city") or "").strip(),
            location_state=str(crop_location.get("state") or "").strip(),
            location_country=str(crop_location.get("country") or "").strip(),
            location_sublocation=str(crop_location.get("sublocation") or "").strip(),
            locations_shown=crop_locations_shown,
            source_text=str(existing_state.get("source_text") or ""),
            detections_payload=existing_detections,
            create_date=str(existing_state.get("create_date") or ""),
            dc_date=list(existing_state.get("dc_date_values") or []),
            date_time_original=str(existing_state.get("date_time_original") or ""),
            ocr_ran=bool(existing_state.get("ocr_ran", False)),
            people_detected=bool(new_person_names),
            people_identified=bool(new_person_names),
        )
        crops_updated += 1

    return {"crops_updated": crops_updated}
