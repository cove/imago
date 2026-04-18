from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai_album_titles import _derived_name_match
from .ai_location import _xmp_gps_to_decimal
from .ai_render_settings import find_archive_dir_for_image
from .xmpmm_provenance import read_derived_from
from ..exiftool_utils import read_tag
from ..naming import DERIVED_VIEW_RE, is_photos_dir, pages_dir_for_album_dir
from .xmp_sidecar import _dedupe, _normalize_xmp_datetime, read_ai_sidecar_state

MIN_EXISTING_SIDECAR_BYTES = 100


def has_valid_sidecar(path: Path) -> bool:
    sidecar_path = path.with_suffix(".xmp")
    try:
        return sidecar_path.is_file() and int(sidecar_path.stat().st_size) > MIN_EXISTING_SIDECAR_BYTES
    except FileNotFoundError:
        return False


def has_current_sidecar(path: Path) -> bool:
    sidecar_path = path.with_suffix(".xmp")
    try:
        if not has_valid_sidecar(path):
            return False
        return int(sidecar_path.stat().st_mtime_ns) >= int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return False


def _sidecar_current_for_paths(sidecar_path: Path, source_paths: list[Path]) -> bool:
    try:
        if not sidecar_path.is_file():
            return False
        sidecar_mtime_ns = int(sidecar_path.stat().st_mtime_ns)
        latest_source_mtime_ns = max(int(path.stat().st_mtime_ns) for path in source_paths)
        return sidecar_mtime_ns >= latest_source_mtime_ns
    except (FileNotFoundError, ValueError):
        return False


def _xmp_timestamp_from_path(path: Path) -> str:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return ""
    text = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat()
    return text.replace("+00:00", "Z")


def read_embedded_create_date(path: Path) -> str:
    for tag in (
        "XMP-xmp:CreateDate",
        "XMP-exif:DateTimeOriginal",
        "EXIF:DateTimeOriginal",
        "EXIF:CreateDate",
    ):
        normalized = _normalize_xmp_datetime(str(read_tag(path, tag) or "").strip())
        if normalized:
            return normalized
    return ""


def _dc_source_scan_names(source_text: str) -> list[str]:
    names: list[str] = []
    for part in str(source_text or "").split(";"):
        candidate = Path(str(part or "").strip()).name
        if candidate.lower().endswith(".tif"):
            names.append(candidate)
    return _dedupe(names)


def _is_derived_image_path(image_path: Path) -> bool:
    return _derived_name_match(image_path) is not None or DERIVED_VIEW_RE.search(Path(image_path).stem) is not None


def _resolve_derived_source_sidecar_state(
    image_path: Path,
    sidecar_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not _is_derived_image_path(image_path) or not isinstance(sidecar_state, dict):
        return None
    derived_from = read_derived_from(image_path.with_suffix(".xmp"))
    derived_source_path = str(derived_from.get("source_path") or "").strip()
    if derived_source_path:
        source_candidate = Path(derived_source_path)
        source_sidecar_candidates: list[Path] = []
        if source_candidate.is_absolute():
            source_sidecar_candidates.append(source_candidate.with_suffix(".xmp"))
        else:
            source_sidecar_candidates.append((image_path.parent / source_candidate).with_suffix(".xmp"))
            source_name = source_candidate.name
            if source_name:
                source_sidecar_candidates.append((image_path.parent / source_name).with_suffix(".xmp"))
            if is_photos_dir(image_path.parent):
                view_dir = pages_dir_for_album_dir(image_path.parent)
                source_sidecar_candidates.append((view_dir / source_candidate).with_suffix(".xmp"))
                if source_name:
                    source_sidecar_candidates.append((view_dir / source_name).with_suffix(".xmp"))
            archive_dir = find_archive_dir_for_image(image_path)
            if archive_dir is not None and archive_dir.is_dir():
                source_sidecar_candidates.append((archive_dir / source_candidate.name).with_suffix(".xmp"))
        for source_sidecar_path in source_sidecar_candidates:
            if not source_sidecar_path.is_file():
                continue
            source_state = read_ai_sidecar_state(source_sidecar_path)
            if isinstance(source_state, dict):
                return source_state
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is None or not archive_dir.is_dir():
        return None
    scan_names = _dc_source_scan_names(str(sidecar_state.get("source_text") or ""))
    if not scan_names:
        return None
    source_state = read_ai_sidecar_state((archive_dir / scan_names[0]).with_suffix(".xmp"))
    return source_state if isinstance(source_state, dict) else None


def _resolve_derived_source_ocr_text(image_path: Path, sidecar_state: dict[str, Any] | None) -> str:
    source_state = _resolve_derived_source_sidecar_state(image_path, sidecar_state)
    if not isinstance(source_state, dict):
        return ""
    return str(source_state.get("ocr_text") or "").strip()


def _sidecar_location_payload(sidecar_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sidecar_state, dict):
        return {}
    detections = sidecar_state.get("detections")
    location = dict(detections.get("location") or {}) if isinstance(detections, dict) else {}
    gps_latitude = _xmp_gps_to_decimal(sidecar_state.get("gps_latitude"), axis="lat")
    gps_longitude = _xmp_gps_to_decimal(sidecar_state.get("gps_longitude"), axis="lon")
    if gps_latitude and not str(location.get("gps_latitude") or "").strip():
        location["gps_latitude"] = gps_latitude
    if gps_longitude and not str(location.get("gps_longitude") or "").strip():
        location["gps_longitude"] = gps_longitude
    if str(sidecar_state.get("location_city") or "").strip() and not str(location.get("city") or "").strip():
        location["city"] = str(sidecar_state.get("location_city") or "").strip()
    if str(sidecar_state.get("location_state") or "").strip() and not str(location.get("state") or "").strip():
        location["state"] = str(sidecar_state.get("location_state") or "").strip()
    if str(sidecar_state.get("location_country") or "").strip() and not str(location.get("country") or "").strip():
        location["country"] = str(sidecar_state.get("location_country") or "").strip()
    if (
        str(sidecar_state.get("location_sublocation") or "").strip()
        and not str(location.get("sublocation") or "").strip()
    ):
        location["sublocation"] = str(sidecar_state.get("location_sublocation") or "").strip()
    return location


def _effective_sidecar_location_payload(image_path: Path, sidecar_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sidecar_state, dict):
        return {}
    if _is_derived_image_path(image_path):
        source_location = _sidecar_location_payload(_resolve_derived_source_sidecar_state(image_path, sidecar_state))
        if source_location:
            return source_location
    return _sidecar_location_payload(sidecar_state)


def _effective_sidecar_ocr_text(image_path: Path, sidecar_state: dict[str, Any] | None) -> str:
    if not isinstance(sidecar_state, dict):
        return ""
    if _is_derived_image_path(image_path):
        return _resolve_derived_source_ocr_text(image_path, sidecar_state)
    return str(sidecar_state.get("ocr_text") or "").strip()


def _resolve_xmp_text_layers(
    *,
    image_path: Path,
    ocr_text: str,
    page_like: bool,
    ocr_authority_source: str = "",
    author_text: str = "",
    scene_text: str = "",
) -> dict[str, str]:
    del image_path, ocr_text, page_like, ocr_authority_source
    clean_author = str(author_text or "").strip()
    clean_scene = str(scene_text or "").strip()

    return {
        "author_text": clean_author,
        "scene_text": clean_scene,
    }


def _compute_xmp_title(
    *,
    image_path: Path,
    explicit_title: str,
    title_source: str = "",
    author_text: str = "",
) -> tuple[str, str]:
    clean_title = str(explicit_title or "").strip()
    clean_source = str(title_source or "").strip()

    if clean_title and clean_source:
        return clean_title, clean_source
    return "", ""
