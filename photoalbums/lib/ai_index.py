from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .ai_album_titles import (
    _album_identity_key,
    _album_title_valid_in_sidecars,
    _base_page_name_match,
    _derived_name_match,
    _expand_album_title_dependencies,
    _is_album_title_source_candidate,
    _iter_album_cover_sidecars,
    _looks_like_album_title_page,
    _require_album_title_for_title_page,
    _resolve_album_printed_title_hint,
    _resolve_album_title_from_sidecars,
    _resolve_album_title_hint,
    _resolve_title_page_album_title,
    _scan_name_match,
    _store_album_printed_title_hint,
    _title_page_match,
)
from .album_sets import find_archive_set_by_photos_root
from .ai_caption import (
    CaptionEngine,
    DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
    clean_text,
    _normalize_gps_value,
    normalize_lmstudio_base_url,
    resolve_caption_model,
)
from .ai_date import DateEstimateEngine
from .ai_model_settings import default_lmstudio_base_url, default_ocr_model
from .ai_ocr import OCREngine, extract_keywords
from .ai_page_layout import prepare_image_layout
from .ai_geocode import NominatimGeocoder
from .ai_location import (
    _extract_explicit_gps_from_text,
    _has_legacy_ai_locations_shown_gps,
    _merge_location_estimates,
    _resolve_location_metadata,
    _resolve_location_payload,
    _resolve_locations_shown,
    _xmp_gps_to_decimal,
)
from .ai_processing_locks import (
    BATCH_LOCK_SUFFIX,
    JOB_ID_ENV,
    PROCESSING_LOCK_SUFFIX,
    _acquire_batch_processing_lock,
    _acquire_image_processing_lock,
    _release_batch_processing_lock,
    _release_image_processing_lock,
)
from .ai_render_settings import (
    find_archive_dir_for_image,
    load_render_settings,
    resolve_effective_settings,
)
from .ai_sidecar_state import (
    MIN_EXISTING_SIDECAR_BYTES,
    _compute_xmp_title,
    _dc_source_scan_names,
    _effective_sidecar_location_payload,
    _effective_sidecar_ocr_text,
    _is_derived_image_path,
    _resolve_derived_source_sidecar_state,
    _resolve_xmp_text_layers,
    _sidecar_current_for_paths,
    _sidecar_location_payload,
    _xmp_timestamp_from_path,
    has_current_sidecar,
    has_valid_sidecar,
    read_embedded_create_date,
)
from .prompt_debug import PromptDebugSession
from ..common import PHOTO_ALBUMS_DIR
from ..exiftool_utils import read_tag
from ..naming import (
    DERIVED_VIEW_RE,
    SCAN_TIFF_RE,
    parse_album_filename,
)
from .xmp_sidecar import (
    _dedupe,
    _normalize_xmp_datetime,
    _resolve_date_time_original,
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
)
from .xmp_review import load_ai_xmp_review

# Re-exports from extracted modules — keep backward compatibility for tests and callers.
from .ai_index_args import (  # noqa: F401
    DEFAULT_CAST_STORE,
    DEFAULT_CREATOR_TOOL,
    IMAGE_EXTENSIONS,
    _absolute_cli_path,
    _explicit_cli_flags,
    _resolve_caption_prompt,
    parse_args,
)
from .ai_index_engine_cache import (  # noqa: F401
    PROCESSOR_SIGNATURE,
    _init_caption_engine,
    _init_date_engine,
    _init_object_detector,
    _init_people_matcher,
    _settings_signature,
)
from .ai_index_analysis import (  # noqa: F401
    AI_MODEL_MAX_SOURCE_BYTES,
    ArchiveScanOCRAuthority,
    ImageAnalysis,
    _build_caption_metadata,
    _caption_people_name_score,
    _estimate_people_from_detections,
    _get_image_dimensions,
    _merge_people_estimates,
    _merge_people_matches,
    _prepare_ai_model_image,
    _refresh_detection_model_metadata,
    _resolve_people_count_metadata,
    _run_image_analysis,
    _serialize_people_matches,
)
from .ai_index_scan import (  # noqa: F401
    _aggregate_best_rows,
    _bounds_offset,
    _build_dc_source,
    _build_flat_page_description,
    _build_flat_payload,
    _dc_source_needs_refresh,
    _hash_text,
    _layout_payload,
    _page_scan_filenames,
    _resolve_archive_scan_authoritative_ocr,
    _scan_group_paths,
    _scan_group_signature,
    _scan_number,
    _scan_page_key,
)


def _format_eta(completed_times: list[float], remaining: int) -> str:
    if not completed_times or remaining <= 0:
        return ""
    avg = sum(completed_times) / len(completed_times)
    total_seconds = int(avg * remaining)
    if total_seconds < 60:
        return f"eta:{total_seconds}s"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"eta:{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"eta:{hours}h{mins:02d}m"


def _progress_ticker(prefix: str, _interval: float = 0.5):
    """Returns (stop, set_step). Prints each step as a new line."""

    def set_step(name: str) -> None:
        print(f"  {prefix}  [{name}]", flush=True)

    def stop() -> None:
        pass

    return stop, set_step


def _display_work_label(image_path: Path) -> str:
    if _scan_name_match(image_path):
        collection, year, book, page = parse_album_filename(image_path.name)
        if collection != "Unknown":
            return f"{collection}_{year}_B{book}_P{int(page):02d}"
    return image_path.name


def _format_reprocess_reasons(reasons: list[str]) -> str:
    clean = _dedupe([str(reason or "").strip() for reason in reasons])
    return ", ".join(clean)


def _apply_shard(files: list[Path], shard_count: int, shard_index: int) -> list[Path]:
    if shard_count <= 1:
        return list(files)
    selected: list[Path] = []
    for path in files:
        album_key = _album_identity_key(path)
        digest = hashlib.sha1(album_key.encode("utf-8")).digest()
        if int.from_bytes(digest[:8], "big") % shard_count == shard_index:
            selected.append(path)
    return selected


def _compute_people_positions(people_matches: list, image_path: Path) -> dict[str, str]:
    """Return a dict mapping each identified person's name to a position label.

    Uses the face bbox (absolute pixels in the image's coordinate space) and
    the image dimensions to produce a human-readable location like 'upper-left'.
    """
    from ._caption_prompts import (
        _position_label,
    )  # pylint: disable=import-outside-toplevel

    try:
        from PIL import Image as _PILImage  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(_PILImage)
        with _PILImage.open(str(image_path)) as _img:
            img_w, img_h = _img.size
    except Exception:
        return {}
    positions: dict[str, str] = {}
    for match in people_matches:
        name = str(getattr(match, "name", "") or "").strip()
        bbox = list(getattr(match, "bbox", None) or [])
        if not name or len(bbox) < 4 or img_w <= 0 or img_h <= 0:
            continue
        x, y, w, h = bbox[:4]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        label = _position_label(float(cx), float(cy))
        if label:
            positions[name] = label
    return positions


def _format_people_step_label(step: str, names: list[str]) -> str:
    clean_names = _dedupe([str(name or "").strip() for name in names])
    names_text = ", ".join(clean_names) if clean_names else "none"
    return f"{step} {len(clean_names)}: {names_text}"


JOB_ARTIFACTS_ENV = "IMAGO_JOB_ARTIFACTS"
CAST_STORE_RETRY_ATTEMPTS = 6
CAST_STORE_RETRY_DELAY_SECONDS = 0.5
TITLE_PAGE_LOCATION_SOURCE = "title_page_location_config"


def _is_retryable_cast_store_write_error(exc: Exception) -> bool:
    if not isinstance(exc, OSError):
        return False
    lower = str(exc or "").strip().lower()
    if not lower:
        return False
    if getattr(exc, "winerror", None) not in {5, 32} and not isinstance(exc, PermissionError):
        return False
    return any(name in lower for name in ("faces.jsonl", "review_queue.jsonl", "people.json"))


def _match_people_with_cast_store_retry(
    *,
    people_matcher: Any,
    image_path: Path,
    source_path: Path,
    bbox_offset: tuple[int, int],
    hint_text: str,
) -> list[Any]:
    last_exc: Exception | None = None
    for attempt in range(CAST_STORE_RETRY_ATTEMPTS):
        try:
            return people_matcher.match_image(
                image_path,
                source_path=source_path,
                bbox_offset=bbox_offset,
                hint_text=hint_text,
            )
        except Exception as exc:
            if not _is_retryable_cast_store_write_error(exc) or attempt >= CAST_STORE_RETRY_ATTEMPTS - 1:
                raise
            last_exc = exc
            time.sleep(CAST_STORE_RETRY_DELAY_SECONDS)
    if last_exc is not None:
        raise last_exc
    return []


def discover_images(
    photos_root: Path,
    *,
    include_archive: bool,
    include_view: bool,
    extensions: set[str],
) -> list[Path]:
    files: list[Path] = []
    for path in photos_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        parent_names = {parent.name for parent in path.parents}
        in_archive = any(name.endswith("_Archive") for name in parent_names)
        in_view = any(name.endswith("_View") for name in parent_names)
        if in_archive and include_archive:
            files.append(path)
            continue
        if in_view and include_view:
            files.append(path)
            continue
    files.sort()
    return files


def append_job_artifact(record: dict[str, Any]) -> None:
    artifact_file = str(os.environ.get(JOB_ARTIFACTS_ENV) or "").strip()
    if not artifact_file:
        return
    path = Path(artifact_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _emit_prompt_debug_artifact(prompt_debug: PromptDebugSession | None, *, dry_run: bool) -> None:
    if prompt_debug is None or not prompt_debug.has_steps():
        return
    append_job_artifact(prompt_debug.to_artifact())


def _append_geocode_artifact(*, image_path: Path, record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        return
    append_job_artifact(
        {
            "kind": "photoalbums_geocode",
            "image_path": str(image_path),
            "label": _display_work_label(image_path),
            **record,
        }
    )


def _is_archive_file(image_path: Path) -> bool:
    return str(image_path.parent.name or "").endswith("_Archive")


def _page_sort_key(image_path: Path) -> tuple[str, int, int, str]:
    album_key = _album_identity_key(image_path)
    _collection, _year, _book, page_str = parse_album_filename(image_path.name)
    try:
        page_number = int(page_str)
    except ValueError:
        page_number = 0
    if _scan_name_match(image_path):
        kind_rank = 0
    elif _derived_name_match(image_path):
        kind_rank = 1
    else:
        kind_rank = 2
    return album_key, page_number, kind_rank, image_path.name.casefold()


def _coalesce_archive_processing_files(files: list[Path]) -> list[Path]:
    scan_groups: dict[str, list[Path]] = {}
    passthrough: list[Path] = []
    for image_path in files:
        if _is_archive_file(image_path) and _scan_name_match(image_path):
            page_key = _scan_page_key(image_path)
            if page_key is None:
                passthrough.append(image_path)
                continue
            scan_groups.setdefault(page_key, []).append(image_path)
            continue
        passthrough.append(image_path)

    selected: list[Path] = []
    missing_s01_pages: list[str] = []
    for _page_key, group_paths in sorted(scan_groups.items()):
        group_paths = sorted(group_paths, key=_scan_number)
        primary_scan = next((path for path in group_paths if _scan_number(path) == 1), None)
        if primary_scan is None:
            missing_s01_pages.append(" + ".join(path.name for path in group_paths))
            continue
        selected.append(primary_scan)

    if missing_s01_pages:
        raise RuntimeError("Missing S01 scan for page(s): " + "; ".join(missing_s01_pages))

    selected.extend(passthrough)
    selected.sort(key=_page_sort_key)
    return selected


def _filter_files_by_tree(files: list[Path], *, include_archive: bool, include_view: bool) -> list[Path]:
    filtered: list[Path] = []
    for image_path in files:
        parent_names = {parent.name for parent in image_path.parents}
        in_archive = any(name.endswith("_Archive") for name in parent_names)
        in_view = any(name.endswith("_View") for name in parent_names)
        if in_archive and include_archive:
            filtered.append(image_path)
            continue
        if in_view and include_view:
            filtered.append(image_path)
            continue
    return filtered


def _format_location_hint_from_state(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return ""
    parts = [
        str(state.get("location_sublocation") or "").strip(),
        str(state.get("location_city") or "").strip(),
        str(state.get("location_state") or "").strip(),
        str(state.get("location_country") or "").strip(),
    ]
    return ", ".join(part for part in parts if part)


def _resolve_upstream_page_sidecar_state(image_path: Path) -> dict[str, Any] | None:
    if not _derived_name_match(image_path):
        return None
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is None or not archive_dir.is_dir():
        return None
    scan_filenames = _page_scan_filenames(image_path)
    if not scan_filenames:
        return None
    primary_scan_name = next(
        (
            scan_name
            for scan_name in scan_filenames
            if (match := _scan_name_match(scan_name)) is not None and int(match.group("scan")) == 1
        ),
        "",
    )
    if not primary_scan_name:
        raise RuntimeError(f"Missing S01 scan for page context: {image_path}")
    sidecar_path = (archive_dir / primary_scan_name).with_suffix(".xmp")
    state = read_ai_sidecar_state(sidecar_path)
    return state if isinstance(state, dict) else None


def _contextualize_ocr_text(ocr_text: str, *, context_ocr_text: str = "", context_location_hint: str = "") -> str:
    parts = [str(ocr_text or "").strip()]
    clean_context_ocr = str(context_ocr_text or "").strip()
    if clean_context_ocr:
        parts.append(f"Parent page OCR hint (context only):\n{clean_context_ocr}")
    clean_location_hint = str(context_location_hint or "").strip()
    if clean_location_hint:
        parts.append(f"Parent page location hint (context only):\n{clean_location_hint}")
    return "\n\n".join(part for part in parts if part)


def _mirror_page_sidecars(primary_scan_path: Path) -> None:
    if not _scan_name_match(primary_scan_path):
        return
    sibling_scans = _scan_group_paths(primary_scan_path)
    if len(sibling_scans) <= 1:
        return
    source_sidecar = primary_scan_path.with_suffix(".xmp")
    if not source_sidecar.is_file():
        raise RuntimeError(f"Page sidecar missing for copy step: {source_sidecar}")
    for sibling_path in sibling_scans:
        if sibling_path == primary_scan_path:
            continue
        shutil.copy2(source_sidecar, sibling_path.with_suffix(".xmp"))


def _artifact_sidecar_paths(image_path: Path, sidecar_path: Path) -> list[Path]:
    if _scan_name_match(image_path):
        return [path.with_suffix(".xmp") for path in _scan_group_paths(image_path)]
    return [sidecar_path]


def _append_xmp_job_artifact(image_path: Path, sidecar_path: Path) -> None:
    sidecar_paths = _artifact_sidecar_paths(image_path, sidecar_path)
    append_job_artifact(
        {
            "kind": "photoalbums_xmp",
            "image_path": str(image_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_paths": [str(path) for path in sidecar_paths],
            "label": _display_work_label(image_path),
        }
    )


def _configured_title_page_location_payload(
    image_path: Path,
    title_page_location: dict[str, str] | None,
) -> dict[str, Any]:
    if not isinstance(title_page_location, dict):
        return {}
    _, _, _, page_str = parse_album_filename(image_path.name)
    if not page_str.isdigit() or int(page_str) != 1:
        return {}
    latitude = str(title_page_location.get("gps_latitude") or "").strip()
    longitude = str(title_page_location.get("gps_longitude") or "").strip()
    if not latitude or not longitude:
        return {}
    payload: dict[str, Any] = {
        "gps_latitude": float(latitude),
        "gps_longitude": float(longitude),
        "map_datum": "WGS-84",
        "source": TITLE_PAGE_LOCATION_SOURCE,
    }
    address = str(title_page_location.get("address") or "").strip()
    if address:
        payload["query"] = address
        payload["display_name"] = address
    for key in ("city", "state", "country", "sublocation"):
        value = str(title_page_location.get(key) or "").strip()
        if value:
            payload[key] = value
    return payload


def _apply_title_page_location_config(
    *,
    image_path: Path,
    location_payload: dict[str, Any] | None,
    detections_payload: dict[str, Any] | None = None,
    title_page_location: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    loc = dict(location_payload or {})
    configured = _configured_title_page_location_payload(image_path, title_page_location)
    if configured:
        loc = configured
    elif str(loc.get("source") or "").strip() == TITLE_PAGE_LOCATION_SOURCE:
        loc = {}
    if not isinstance(detections_payload, dict):
        return loc, detections_payload
    detections = dict(detections_payload)
    if loc:
        detections["location"] = dict(loc)
    elif "location" in detections:
        del detections["location"]
    return loc, detections


def needs_processing(
    path: Path,
    sidecar_state: dict[str, Any] | None,
    force: bool,
    *,
    reprocess_required: bool = False,
) -> bool:
    if force:
        return True
    if _is_album_title_source_candidate(path) and isinstance(sidecar_state, dict):
        ocr = str(sidecar_state.get("ocr_text") or "").strip()
        title = str(sidecar_state.get("album_title") or "").strip()
        if ocr and title == ocr:
            return True
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    sidecar_path = path.with_suffix(".xmp")
    if not reprocess_required and has_current_sidecar(path):
        return False
    if reprocess_required:
        return True
    if sidecar_state is not None:
        if str(sidecar_state.get("processor_signature") or "") != PROCESSOR_SIGNATURE:
            return True
        recorded_size = int(sidecar_state.get("size") or -1)
        recorded_mtime = int(sidecar_state.get("mtime_ns") or -1)
        if int(stat.st_size) != recorded_size or int(stat.st_mtime_ns) != recorded_mtime:
            return True
        return not has_current_sidecar(path)
    if not has_valid_sidecar(path):
        return True
    return int(sidecar_path.stat().st_mtime_ns) < int(stat.st_mtime_ns)


def _sidecar_has_lmstudio_caption_error(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    detections = state.get("detections")
    if not isinstance(detections, dict):
        return False
    caption = detections.get("caption")
    if not isinstance(caption, dict):
        return False
    error_text = str(caption.get("error") or "").strip()
    if not error_text:
        return False
    requested_engine = str(caption.get("requested_engine") or "").strip().lower()
    effective_engine = str(caption.get("effective_engine") or "").strip().lower()
    return "lmstudio" in {requested_engine, effective_engine}


def _sidecar_has_people_to_refresh(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    detections = state.get("detections")
    if isinstance(detections, dict):
        people = detections.get("people")
        if isinstance(people, list) and any(isinstance(person, dict) for person in people):
            return True
    if state.get("people_identified") is True:
        return True
    people_detected = state.get("people_detected")
    if people_detected is not None:
        return bool(people_detected)
    return False


def _date_estimate_input_hash(ocr_text: str, album_title: str) -> str:
    clean_ocr = str(ocr_text or "").strip()
    clean_album_title = str(album_title or "").strip()
    if not clean_ocr and not clean_album_title:
        return ""
    return _hash_text(
        json.dumps(
            {
                "ocr_text": clean_ocr,
                "album_title": clean_album_title,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _dc_date_value(sidecar_state: dict[str, Any] | None) -> str | list[str]:
    if not isinstance(sidecar_state, dict):
        return ""
    raw_values = sidecar_state.get("dc_date_values")
    if isinstance(raw_values, list):
        values = [str(item or "").strip() for item in raw_values if str(item or "").strip()]
        if values:
            return values
    return str(sidecar_state.get("dc_date") or "").strip()


def _has_dc_date(value: str | list[str]) -> bool:
    if isinstance(value, list):
        return any(str(item or "").strip() for item in value)
    return bool(str(value or "").strip())


def _dc_date_needs_refresh(
    image_path: Path,
    sidecar_state: dict[str, Any] | None,
    *,
    enabled: bool,
) -> bool:
    if not isinstance(sidecar_state, dict):
        return False
    current_dc_date = _dc_date_value(sidecar_state)
    current_date_time_original = str(sidecar_state.get("date_time_original") or "").strip()
    if _has_dc_date(current_dc_date):
        return _resolve_date_time_original(dc_date=current_dc_date) != current_date_time_original
    if not enabled:
        return False
    current_hash = _date_estimate_input_hash(
        _effective_sidecar_ocr_text(image_path, sidecar_state),
        str(sidecar_state.get("album_title") or ""),
    )
    if not current_hash:
        return False
    return current_hash != str(sidecar_state.get("date_estimate_input_hash") or "").strip()


def _resolve_dc_date(
    *,
    existing_dc_date: str | list[str],
    ocr_text: str,
    album_title: str,
    image_path: Path,
    date_engine: DateEstimateEngine | None,
    prompt_debug: PromptDebugSession | None,
) -> str | list[str]:
    if isinstance(existing_dc_date, list):
        clean_existing = [str(item or "").strip() for item in existing_dc_date if str(item or "").strip()]
        if clean_existing:
            return clean_existing
    else:
        clean_existing = str(existing_dc_date or "").strip()
        if clean_existing:
            return clean_existing
    if date_engine is None:
        return ""
    input_hash = _date_estimate_input_hash(ocr_text, album_title)
    if not input_hash:
        return ""
    result = date_engine.estimate(
        ocr_text=ocr_text,
        album_title=album_title,
        source_path=image_path,
        debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
        debug_step="date_estimate",
    )
    if str(result.error or "").strip():
        raise RuntimeError(f"Date estimate failed: {result.error}")
    return str(result.date or "").strip()


def _write_sidecar_and_record(
    sidecar_path: Path,
    image_path: Path,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str = "",
    location_payload: dict[str, Any],
    source_text: str = "",
    ocr_text: str,
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    detections_payload: dict[str, Any] | None = None,
    subphotos: list[dict[str, Any]] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    create_date: str = "",
    dc_date: str | list[str] = "",
    date_time_original: str = "",
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
    title_page_location: dict[str, str] | None = None,
) -> None:
    """Write XMP sidecar and record the artifact.  Derives history_when and image
    dimensions from image_path; unpacks GPS fields from location_payload."""
    img_w, img_h = _get_image_dimensions(image_path)
    loc, detections_payload = _apply_title_page_location_config(
        image_path=image_path,
        location_payload=location_payload,
        detections_payload=detections_payload,
        title_page_location=title_page_location,
    )
    write_xmp_sidecar(
        sidecar_path,
        creator_tool=creator_tool,
        person_names=person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=description,
        album_title=album_title,
        gps_latitude=str(loc.get("gps_latitude") or ""),
        gps_longitude=str(loc.get("gps_longitude") or ""),
        location_city=str(loc.get("city") or ""),
        location_state=str(loc.get("state") or ""),
        location_country=str(loc.get("country") or ""),
        location_sublocation=str(loc.get("sublocation") or ""),
        source_text=source_text,
        ocr_text=ocr_text,
        ocr_lang=ocr_lang,
        author_text=author_text,
        scene_text=scene_text,
        detections_payload=detections_payload,
        subphotos=subphotos,
        stitch_key=stitch_key,
        ocr_authority_source=ocr_authority_source,
        create_date=create_date,
        dc_date=dc_date,
        date_time_original=date_time_original,
        history_when=_xmp_timestamp_from_path(image_path),
        image_width=img_w,
        image_height=img_h,
        ocr_ran=ocr_ran,
        people_detected=people_detected,
        people_identified=people_identified,
        locations_shown=detections_payload.get("locations_shown") if detections_payload else None,
    )
    _append_xmp_job_artifact(image_path, sidecar_path)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_flags = _explicit_cli_flags(argv)
    requested_caption_prompt = _resolve_caption_prompt(
        str(getattr(args, "caption_prompt", "")),
        str(getattr(args, "caption_prompt_file", "")),
    )
    photos_root = _absolute_cli_path(args.photos_root)
    archive_set = find_archive_set_by_photos_root(photos_root)
    title_page_location = archive_set.title_page_location if archive_set is not None else None
    stdout_only = bool(args.stdout)
    reprocess_mode = str(args.reprocess_mode)
    force_processing = bool(args.force or stdout_only or reprocess_mode == "all")
    dry_run = bool(args.dry_run or stdout_only)
    shard_count = int(args.shard_count or 1)
    shard_index = int(args.shard_index or 0)

    def emit_info(message: str) -> None:
        if not stdout_only:
            print(message)

    def emit_error(message: str) -> None:
        print(message, file=sys.stderr if stdout_only else sys.stdout, flush=True)

    if not photos_root.is_dir():
        raise SystemExit(f"Photo root is not a directory: {photos_root}")
    if shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit("--shard-index must be between 0 and --shard-count - 1")

    include_archive = bool(args.include_archive)
    include_view = bool(args.include_view)
    if not include_archive and not include_view:
        include_archive = True
        include_view = False

    ext_set = {
        (item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}")
        for item in str(args.extensions or "").split(",")
        if item.strip()
    }
    if not ext_set:
        ext_set = set(IMAGE_EXTENSIONS)

    single_photo = str(args.photo or "").strip()
    if single_photo:
        if shard_count > 1:
            raise SystemExit("--shard-count > 1 is only supported for multi-photo discovery runs")
        photo_path = _absolute_cli_path(single_photo)
        if not photo_path.is_file():
            raise SystemExit(f"Photo not found: {photo_path}")
        files = [photo_path]
        force_processing = True
    else:
        files = discover_images(
            photos_root,
            include_archive=include_archive,
            include_view=include_view,
            extensions=ext_set,
        )
        album_filter = str(args.album or "").strip()
        if album_filter:
            album_lower = album_filter.casefold()
            files = [f for f in files if album_lower in f.parent.name.casefold()]
        files = _coalesce_archive_processing_files(files)
        photo_offset = int(args.photo_offset or 0)
        if photo_offset > 0:
            files = files[photo_offset:]
        if args.max_images and args.max_images > 0:
            files = files[: int(args.max_images)]
        files = _apply_shard(files, shard_count, shard_index)

    original_file_count = len(files)
    files = _expand_album_title_dependencies(files, ext_set)
    if not single_photo:
        files = _filter_files_by_tree(
            files,
            include_archive=include_archive,
            include_view=include_view,
        )

    emit_info(f"Discovered {len(files)} image files")
    if len(files) > original_file_count:
        emit_info(f"Added {len(files) - original_file_count} title-page dependency files")
    if not files:
        return 0

    batch_lock_path: Path | None = None
    allow_concurrent_shards = not single_photo and shard_count > 1
    if not single_photo and not allow_concurrent_shards:
        try:
            batch_lock_path = _acquire_batch_processing_lock(photos_root)
        except RuntimeError as exc:
            emit_error(str(exc))
            return 1

    default_caption_max_tokens = int(args.caption_max_tokens)
    if "--caption-max-tokens" not in explicit_flags and str(args.caption_engine) == "lmstudio":
        default_caption_max_tokens = max(default_caption_max_tokens, int(DEFAULT_LMSTUDIO_MAX_NEW_TOKENS))

    defaults = {
        "skip": False,
        "enable_people": not bool(args.disable_people),
        "enable_objects": not bool(args.disable_objects),
        "ocr_engine": str(args.ocr_engine),
        "ocr_lang": str(args.ocr_lang),
        "ocr_model": str(args.ocr_model),
        "caption_engine": str(args.caption_engine),
        "caption_model": resolve_caption_model(str(args.caption_engine), str(args.caption_model)),
        "caption_prompt": str(requested_caption_prompt),
        "caption_max_tokens": int(default_caption_max_tokens),
        "caption_temperature": float(args.caption_temperature),
        "caption_max_edge": int(args.caption_max_edge),
        "lmstudio_base_url": normalize_lmstudio_base_url(str(args.lmstudio_base_url)),
        "people_threshold": float(args.people_threshold),
        "object_threshold": float(args.object_threshold),
        "min_face_size": int(args.min_face_size),
        "model": str(args.model),
        "creator_tool": str(args.creator_tool),
    }

    archive_settings_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
    people_matcher_cache: dict[tuple[str, float, int], Any] = {}
    object_detector_cache: dict[tuple[str, float], Any] = {}
    ocr_engine_cache: dict[tuple[str, str, str, str], OCREngine] = {}
    caption_engine_cache: dict[tuple[str, str, str, int, float, str, int], CaptionEngine] = {}
    date_engine_cache: dict[tuple[str, str, int, float, str], DateEstimateEngine] = {}
    archive_scan_ocr_cache: dict[str, ArchiveScanOCRAuthority] = {}
    printed_album_title_cache: dict[str, str] = {}
    geocoder = NominatimGeocoder()
    stitch_cap_td = tempfile.TemporaryDirectory(prefix="imago-stitch-cap-")
    stitch_cap_dir = Path(stitch_cap_td.name)

    processed = 0
    skipped = 0
    failures = 0
    completed_times: list[float] = []

    def _get_date_engine(effective_settings: dict[str, Any]) -> DateEstimateEngine:
        date_key = (
            str(effective_settings.get("caption_engine", defaults["caption_engine"])),
            str(effective_settings.get("caption_model", defaults["caption_model"])),
            int(effective_settings.get("caption_max_tokens", defaults["caption_max_tokens"])),
            0.0,
            str(effective_settings.get("lmstudio_base_url", defaults["lmstudio_base_url"])),
        )
        date_engine = date_engine_cache.get(date_key)
        if date_engine is None:
            date_engine = _init_date_engine(
                engine=date_key[0],
                model_name=date_key[1],
                max_tokens=int(date_key[2]),
                temperature=0.0,
                lmstudio_base_url=date_key[4],
            )
            date_engine_cache[date_key] = date_engine
        return date_engine

    for idx, image_path in enumerate(files, 1):
        sidecar_path = image_path.with_suffix(".xmp")
        existing_xmp_people = read_person_in_image(sidecar_path)
        archive_dir = find_archive_dir_for_image(image_path)
        settings_file: Path | None = None
        loaded_settings: dict[str, Any] | None = None
        if archive_dir is not None and not args.ignore_render_settings:
            key = str(archive_dir.resolve())
            cached = archive_settings_cache.get(key)
            if cached is None:
                path, payload = load_render_settings(
                    archive_dir,
                    defaults=defaults,
                    create=False,
                )
                cached = (path, payload)
                archive_settings_cache[key] = cached
            settings_file, loaded_settings = cached

        effective = resolve_effective_settings(
            image_path,
            defaults=defaults,
            loaded=loaded_settings,
        )
        if args.disable_people:
            effective["enable_people"] = False
        if args.disable_objects:
            effective["enable_objects"] = False
        if "--ocr-engine" in explicit_flags:
            effective["ocr_engine"] = str(args.ocr_engine)
        if "--ocr-model" in explicit_flags:
            effective["ocr_model"] = str(args.ocr_model)
        if "--caption-engine" in explicit_flags:
            effective["caption_engine"] = str(args.caption_engine)
        if "--caption-model" in explicit_flags:
            effective["caption_model"] = str(args.caption_model)
        if (
            "--caption-prompt" in explicit_flags
            or "--local-prompt" in explicit_flags
            or "--local-prompt" in explicit_flags
            or "--qwen-prompt" in explicit_flags
            or "--caption-prompt-file" in explicit_flags
            or "--local-prompt-file" in explicit_flags
            or "--local-prompt-file" in explicit_flags
            or "--qwen-prompt-file" in explicit_flags
        ):
            effective["caption_prompt"] = str(requested_caption_prompt)
        if "--caption-max-tokens" in explicit_flags:
            effective["caption_max_tokens"] = int(args.caption_max_tokens)
        if "--caption-temperature" in explicit_flags:
            effective["caption_temperature"] = float(args.caption_temperature)
        if "--caption-max-edge" in explicit_flags:
            effective["caption_max_edge"] = int(args.caption_max_edge)
        if "--lmstudio-base-url" in explicit_flags:
            effective["lmstudio_base_url"] = normalize_lmstudio_base_url(str(args.lmstudio_base_url))
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", defaults["caption_engine"])),
            str(effective.get("caption_model", defaults["caption_model"])),
        )
        settings_sig = _settings_signature(effective)
        creator_tool = str(effective.get("creator_tool", args.creator_tool))
        date_estimation_enabled = (
            str(effective.get("caption_engine", defaults["caption_engine"])).strip().lower() == "lmstudio"
        )

        existing_sidecar_valid = has_valid_sidecar(image_path)
        existing_sidecar_current = has_current_sidecar(image_path) if existing_sidecar_valid else False
        existing_sidecar_state: dict | None = None
        source_refresh_required = False
        if existing_sidecar_valid:
            existing_sidecar_state = read_ai_sidecar_state(sidecar_path)

        existing_sidecar_complete = False
        reprocess_required = False
        reprocess_reasons: list[str] = []
        date_refresh_required = False
        if existing_sidecar_valid and not existing_sidecar_current:
            reprocess_reasons.append("sidecar_older_than_image")
        if _sidecar_has_lmstudio_caption_error(existing_sidecar_state):
            reprocess_required = True
            reprocess_reasons.append("lmstudio_caption_error")
        if existing_sidecar_valid:
            existing_sidecar_complete = sidecar_has_expected_ai_fields(
                sidecar_path,
                creator_tool=creator_tool,
                enable_people=bool(effective.get("enable_people", True)),
                enable_objects=bool(effective.get("enable_objects", True)),
                ocr_engine=str(effective.get("ocr_engine", defaults["ocr_engine"])),
                caption_engine=str(effective.get("caption_engine", defaults["caption_engine"])),
            )
            source_refresh_required = _dc_source_needs_refresh(image_path, existing_sidecar_state)
            if source_refresh_required:
                reprocess_reasons.append("dc_source_stale")
            date_refresh_required = _dc_date_needs_refresh(
                image_path,
                existing_sidecar_state,
                enabled=date_estimation_enabled,
            )
            if date_refresh_required:
                reprocess_reasons.append("timeline_date_missing")
        caption_engine_name = str(effective.get("caption_engine", defaults["caption_engine"])).strip().lower()
        location_shown_missing = False
        location_shown_gps_dirty = False
        if existing_sidecar_complete and existing_sidecar_state is not None and caption_engine_name == "lmstudio":
            det = existing_sidecar_state.get("detections") or {}
            detected_locations = list(det.get("locations_shown") or []) if isinstance(det, dict) else []
            written_locations = read_locations_shown(sidecar_path)
            location_shown_ran = isinstance(det, dict) and det.get("location_shown_ran") is True
            location_shown_missing = (isinstance(det, dict) and det.get("location_shown_ran") is not True) or (
                (location_shown_ran or bool(detected_locations)) and not written_locations
            )
            location_shown_gps_dirty = _has_legacy_ai_locations_shown_gps(existing_sidecar_state)
        gps_repair_requested = (
            existing_sidecar_current
            and existing_sidecar_complete
            and existing_sidecar_state is not None
            and (location_shown_missing or location_shown_gps_dirty)
            and not reprocess_required
            and not source_refresh_required
            and not date_refresh_required
        )

        if (
            existing_sidecar_current
            and existing_sidecar_complete
            and not reprocess_required
            and not source_refresh_required
            and not date_refresh_required
            and not force_processing
            and not gps_repair_requested
        ):
            skipped += 1
            if args.verbose and not stdout_only:
                print(f"[{idx}/{len(files)}] skip  {image_path.name} (current xmp)")
            continue

        people_matcher = None
        current_cast_signature = ""
        if bool(effective.get("enable_people", True)):
            people_key = (
                str(Path(args.cast_store).resolve()),
                float(effective.get("people_threshold", defaults["people_threshold"])),
                int(effective.get("min_face_size", defaults["min_face_size"])),
            )
            people_matcher = people_matcher_cache.get(people_key)
            if people_matcher is None:
                people_matcher = _init_people_matcher(
                    cast_store=Path(args.cast_store),
                    min_similarity=float(people_key[1]),
                    min_face_size=int(people_key[2]),
                )
                people_matcher_cache[people_key] = people_matcher
            current_cast_signature = str(people_matcher.store_signature())

        existing_sidecar_ocr_hash = _hash_text(str((existing_sidecar_state or {}).get("ocr_text") or ""))
        multi_scan_group_paths = _scan_group_paths(image_path)
        archive_stitched_ocr_required = (
            str(effective.get("ocr_engine", defaults["ocr_engine"])).strip().lower() != "none"
            and len(multi_scan_group_paths) > 1
        )
        multi_scan_group_signature = (
            _scan_group_signature(multi_scan_group_paths) if archive_stitched_ocr_required else ""
        )

        people_update_only = False
        if existing_sidecar_valid and not existing_sidecar_complete:
            reprocess_required = True
            reprocess_reasons.append("sidecar_incomplete")
        existing_album_title = str((existing_sidecar_state or {}).get("album_title") or "").strip()
        if not existing_album_title and (
            _is_album_title_source_candidate(image_path) or _resolve_album_title_from_sidecars(image_path)
        ):
            reprocess_required = True
            reprocess_reasons.append("missing_album_title")
        if archive_stitched_ocr_required:
            sidecar_source = str((existing_sidecar_state or {}).get("ocr_authority_source") or "").strip()
            sidecar_signature = str((existing_sidecar_state or {}).get("ocr_authority_signature") or "").strip()
            sidecar_hash = str((existing_sidecar_state or {}).get("ocr_authority_hash") or "").strip()
            sidecar_has_current_stitched_authority = (
                sidecar_source == "archive_stitched"
                and bool(existing_sidecar_ocr_hash)
                and _sidecar_current_for_paths(sidecar_path, multi_scan_group_paths)
            )
            sidecar_matches_stitched_authority = (
                sidecar_source == "archive_stitched"
                and sidecar_signature == multi_scan_group_signature
                and bool(sidecar_hash)
                and sidecar_hash == existing_sidecar_ocr_hash
            )
            if not sidecar_matches_stitched_authority and not sidecar_has_current_stitched_authority:
                reprocess_required = True
                reprocess_reasons.append("missing_stitched_authority")
        if existing_sidecar_state is not None:
            old_sig = str(existing_sidecar_state.get("settings_signature") or "")
            if old_sig != settings_sig and not (existing_sidecar_current and existing_sidecar_complete):
                reprocess_required = True
                reprocess_reasons.append("settings_signature_mismatch")
            elif bool(effective.get("enable_people", True)):
                if str(existing_sidecar_state.get("cast_store_signature") or "") != current_cast_signature:
                    if _sidecar_has_people_to_refresh(existing_sidecar_state):
                        people_update_only = True
                        reprocess_reasons.append("cast_store_signature_changed")

        needs_full = needs_processing(
            image_path,
            existing_sidecar_state,
            force_processing and not gps_repair_requested,
            reprocess_required=reprocess_required,
        )

        gps_update_only = False
        if gps_repair_requested:
            gps_update_only = True
            if location_shown_missing:
                reprocess_reasons.append("missing_location_shown")
            if location_shown_gps_dirty:
                reprocess_reasons.append("location_shown_ai_gps_stale")
        if (
            not gps_update_only
            and reprocess_mode == "gps"
            and not needs_full
            and existing_sidecar_complete
            and existing_sidecar_state is not None
        ):
            gps_update_only = True
        if (
            not gps_update_only
            and not needs_full
            and not source_refresh_required
            and not date_refresh_required
            and existing_sidecar_complete
            and existing_sidecar_state is not None
            and caption_engine_name == "lmstudio"
        ):
            if location_shown_missing:
                gps_update_only = True
                reprocess_reasons.append("missing_location_shown")
            if location_shown_gps_dirty:
                gps_update_only = True
                reprocess_reasons.append("location_shown_ai_gps_stale")
        if (
            not needs_full
            and not people_update_only
            and not gps_update_only
            and not isinstance(existing_sidecar_state, dict)
        ):
            skipped += 1
            if args.verbose and not stdout_only:
                print(f"[{idx}/{len(files)}] skip  {image_path.name}")
            continue

        if bool(effective.get("skip", False)):
            skipped += 1
            if args.verbose and not stdout_only:
                print(f"[{idx}/{len(files)}] skip  {image_path.name} (render_settings skip=true)")
            continue

        if reprocess_mode not in ("unprocessed", "all"):
            _reasons_set = set(reprocess_reasons)
            _mode_match: bool
            if reprocess_mode == "new_only":
                _mode_match = existing_sidecar_state is None
            elif reprocess_mode == "errors_only":
                _mode_match = bool(_reasons_set & {"lmstudio_caption_error", "sidecar_incomplete"})
            elif reprocess_mode == "outdated":
                _mode_match = "sidecar_older_than_image" in _reasons_set
            elif reprocess_mode == "cast_changed":
                _mode_match = "cast_store_signature_changed" in _reasons_set
            elif reprocess_mode == "gps":
                _mode_match = gps_update_only
            else:
                _mode_match = True
            if not _mode_match:
                skipped += 1
                if args.verbose and not stdout_only:
                    print(f"[{idx}/{len(files)}] skip  {image_path.name} (reprocess_mode={reprocess_mode})")
                continue

        if existing_sidecar_valid and not stdout_only:
            reason_text = _format_reprocess_reasons(reprocess_reasons)
            if needs_full and reason_text:
                print(
                    f"  [{idx}/{len(files)}]  {image_path.name}  [reprocess: {reason_text}]",
                    flush=True,
                )
            elif people_update_only and reason_text:
                print(
                    f"  [{idx}/{len(files)}]  {image_path.name}  [update: {reason_text}]",
                    flush=True,
                )
            elif (source_refresh_required or date_refresh_required) and reason_text:
                print(
                    f"  [{idx}/{len(files)}]  {image_path.name}  [refresh: {reason_text}]",
                    flush=True,
                )

        # ── Fast path: cast changed but only people+caption need updating ──────
        try:
            lock_path = _acquire_image_processing_lock(image_path)
        except RuntimeError as exc:
            if allow_concurrent_shards and "already processing" in str(exc):
                skipped += 1
                if args.verbose and not stdout_only:
                    print(f"[{idx}/{len(files)}] skip  {image_path.name} ({exc})")
            else:
                failures += 1
                emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")
            continue

        if not needs_full and not people_update_only and not gps_update_only:
            state = existing_sidecar_state
            if isinstance(state, dict):
                file_start = time.monotonic()
                prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
                try:
                    review = load_ai_xmp_review(sidecar_path)
                    refresh_ocr_text = _effective_sidecar_ocr_text(
                        image_path,
                        review if isinstance(review, dict) else None,
                    )
                    refresh_location = _effective_sidecar_location_payload(
                        image_path,
                        review if isinstance(review, dict) else None,
                    )
                    refresh_detections = (
                        dict(review.get("detections") or {}) if isinstance(review.get("detections"), dict) else {}
                    )
                    if refresh_location:
                        refresh_detections["location"] = refresh_location
                    refresh_location, refresh_detections = _apply_title_page_location_config(
                        image_path=image_path,
                        location_payload=refresh_location,
                        detections_payload=refresh_detections,
                        title_page_location=title_page_location,
                    )
                    if not dry_run:
                        refresh_gps_lat = str(refresh_location.get("gps_latitude") or "").strip()
                        refresh_gps_lon = str(refresh_location.get("gps_longitude") or "").strip()
                        if not refresh_gps_lat:
                            refresh_gps_lat = _xmp_gps_to_decimal(review.get("gps_latitude"), axis="lat")
                        if not refresh_gps_lon:
                            refresh_gps_lon = _xmp_gps_to_decimal(review.get("gps_longitude"), axis="lon")
                        refresh_page_like = bool(review.get("subphotos")) or (
                            str((refresh_detections.get("caption") or {}).get("effective_engine") or "").strip()
                            == "page-summary"
                        )
                        text_layers = _resolve_xmp_text_layers(
                            image_path=image_path,
                            ocr_text=refresh_ocr_text,
                            page_like=refresh_page_like,
                            ocr_authority_source=str(review.get("ocr_authority_source") or ""),
                            author_text=str(review.get("author_text") or ""),
                            scene_text=str(review.get("scene_text") or ""),
                        )
                        xmp_title, xmp_title_source = _compute_xmp_title(
                            image_path=image_path,
                            explicit_title=str(review.get("title") or ""),
                            title_source=str(review.get("title_source") or ""),
                            author_text=str(text_layers.get("author_text") or ""),
                        )
                        stat = image_path.stat()
                        summary = dict(review.get("summary") or {})
                        refresh_subphotos = review.get("subphotos")
                        refresh_analysis_mode = str(
                            (existing_sidecar_state or {}).get("analysis_mode")
                            or (
                                "page_subphotos"
                                if isinstance(refresh_subphotos, list) and refresh_subphotos
                                else "single_image"
                            )
                        )
                        refresh_album_title = _require_album_title_for_title_page(
                            image_path=image_path,
                            album_title=_resolve_title_page_album_title(
                                image_path=image_path,
                                album_title=(
                                    str(review.get("album_title") or "").strip()
                                    or _resolve_album_title_hint(image_path)
                                ),
                                ocr_text=refresh_ocr_text,
                            ),
                            context="refresh",
                        )
                        date_engine = (
                            _get_date_engine(effective)
                            if date_estimation_enabled and not _has_dc_date(_dc_date_value(review))
                            else None
                        )
                        refresh_dc_date = _resolve_dc_date(
                            existing_dc_date=_dc_date_value(review),
                            ocr_text=refresh_ocr_text,
                            album_title=refresh_album_title,
                            image_path=image_path,
                            date_engine=date_engine,
                            prompt_debug=prompt_debug,
                        )
                        refresh_date_time_original = _resolve_date_time_original(
                            dc_date=refresh_dc_date,
                            date_time_original=str(review.get("date_time_original") or ""),
                        )
                        if refresh_detections is None:
                            refresh_detections = {}
                        refresh_detections["processing"] = {
                            "processor_signature": PROCESSOR_SIGNATURE,
                            "settings_signature": settings_sig,
                            "cast_store_signature": (
                                current_cast_signature if bool(effective.get("enable_people", True)) else ""
                            ),
                            "size": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                            "date_estimate_input_hash": _date_estimate_input_hash(
                                refresh_ocr_text,
                                refresh_album_title,
                            ),
                            "ocr_authority_signature": str(
                                (existing_sidecar_state or {}).get("ocr_authority_signature") or ""
                            ),
                            "ocr_authority_hash": str((existing_sidecar_state or {}).get("ocr_authority_hash") or ""),
                            "analysis_mode": refresh_analysis_mode,
                        }
                        write_xmp_sidecar(
                            sidecar_path,
                            creator_tool=creator_tool,
                            person_names=list(review.get("person_names") or []),
                            subjects=list(review.get("subjects") or []),
                            title=xmp_title,
                            title_source=xmp_title_source,
                            description=str(review.get("description") or ""),
                            album_title=refresh_album_title,
                            gps_latitude=refresh_gps_lat,
                            gps_longitude=refresh_gps_lon,
                            location_city=str(refresh_location.get("city") or ""),
                            location_state=str(refresh_location.get("state") or ""),
                            location_country=str(refresh_location.get("country") or ""),
                            location_sublocation=str(refresh_location.get("sublocation") or ""),
                            source_text=_build_dc_source(
                                refresh_album_title,
                                image_path,
                                _page_scan_filenames(image_path),
                            ),
                            ocr_text=refresh_ocr_text,
                            author_text=str(text_layers.get("author_text") or ""),
                            scene_text=str(text_layers.get("scene_text") or ""),
                            detections_payload=refresh_detections,
                            stitch_key=str(review.get("stitch_key") or ""),
                            ocr_authority_source=str(review.get("ocr_authority_source") or ""),
                            create_date=(
                                str(review.get("create_date") or "").strip() or read_embedded_create_date(image_path)
                            ),
                            dc_date=refresh_dc_date,
                            date_time_original=refresh_date_time_original,
                            history_when=_xmp_timestamp_from_path(image_path),
                            image_width=_get_image_dimensions(image_path)[0],
                            image_height=_get_image_dimensions(image_path)[1],
                            ocr_ran=bool(review.get("ocr_ran")),
                            people_detected=bool(review.get("people_detected")),
                            people_identified=bool(review.get("people_identified")),
                            ocr_lang=str(review.get("ocr_lang") or ""),
                            locations_shown=refresh_detections.get("locations_shown") if refresh_detections else None,
                        )

                    if not dry_run:
                        _append_xmp_job_artifact(image_path, sidecar_path)
                    _emit_prompt_debug_artifact(prompt_debug, dry_run=dry_run)
                    processed += 1
                    completed_times.append(time.monotonic() - file_start)
                    if not stdout_only:
                        eta_str = _format_eta(completed_times, len(files) - idx)
                        eta_part = f"  {eta_str}" if eta_str else ""
                        print(
                            f"[{idx}/{len(files)}]{eta_part}  ok    {image_path.name}  [refresh]",
                            flush=True,
                        )
                except Exception as exc:
                    failures += 1
                    _emit_prompt_debug_artifact(prompt_debug, dry_run=dry_run)
                    emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")
                _release_image_processing_lock(lock_path)
                continue

        if not needs_full and people_update_only and not stdout_only:
            state = existing_sidecar_state
            chain_gps_update_after_people = bool(gps_update_only)
            if not isinstance(state, dict):
                needs_full = True  # fall through to full processing
            else:
                file_start = time.monotonic()
                det = state.get("detections") or {}
                existing_people_rows = [r for r in list(det.get("people") or []) if isinstance(r, dict)]
                existing_caption_payload = dict(det.get("caption") or {})
                existing_ocr_text = _effective_sidecar_ocr_text(image_path, state)
                existing_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
                existing_object_rows = [r for r in list(det.get("objects") or []) if isinstance(r, dict)]
                existing_object_labels = [str(r.get("label") or "") for r in existing_object_rows if r.get("label")]
                existing_location = _effective_sidecar_location_payload(image_path, state)

                eta_str = _format_eta(completed_times, len(files) - idx + 1)
                eta_part = f"  {eta_str}" if eta_str else ""
                prefix = f"[{idx}/{len(files)}]{eta_part}  {_display_work_label(image_path)}"
                print(prefix, flush=True)
                _pu_stop, _pu_step = _progress_ticker(prefix)

                try:
                    _pu_step("people")
                    pu_people_matches = (
                        _match_people_with_cast_store_retry(
                            people_matcher=people_matcher,
                            image_path=image_path,
                            source_path=image_path,
                            bbox_offset=(0, 0),
                            hint_text=existing_ocr_text,
                        )
                        if people_matcher
                        else []
                    )
                    pu_faces_detected = (
                        (
                            _v
                            if isinstance(
                                _v := getattr(people_matcher, "last_faces_detected", 0),
                                int,
                            )
                            else 0
                        )
                        if people_matcher
                        else 0
                    )
                    pu_people_match_names = _dedupe([r.name for r in pu_people_matches])
                    _pu_step(_format_people_step_label("people", pu_people_match_names))
                    pu_person_names = _dedupe(pu_people_match_names + existing_xmp_people)
                    pu_album_title = _resolve_album_title_hint(image_path)
                    pu_printed_title = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
                    pu_people_payload = _serialize_people_matches(pu_people_matches)
                    pu_prompt_debug = None
                    people_names_changed = pu_person_names != existing_xmp_people
                    if not people_names_changed:
                        pu_updated_det = {
                            **det,
                            "people": pu_people_payload or existing_people_rows,
                            "caption": existing_caption_payload,
                        }
                    else:
                        caption_key = (
                            str(effective.get("caption_engine", defaults["caption_engine"])),
                            str(effective.get("caption_model", defaults["caption_model"])),
                            str(effective.get("caption_prompt", defaults["caption_prompt"])),
                            int(effective.get("caption_max_tokens", defaults["caption_max_tokens"])),
                            float(effective.get("caption_temperature", defaults["caption_temperature"])),
                            str(effective.get("lmstudio_base_url", defaults["lmstudio_base_url"])),
                            int(effective.get("caption_max_edge", defaults["caption_max_edge"])),
                        )
                        pu_caption_engine = caption_engine_cache.get(caption_key)
                        if pu_caption_engine is None:
                            pu_caption_engine = _init_caption_engine(
                                engine=caption_key[0],
                                model_name=caption_key[1],
                                caption_prompt=caption_key[2],
                                max_tokens=int(caption_key[3]),
                                temperature=float(caption_key[4]),
                                lmstudio_base_url=caption_key[5],
                                max_image_edge=int(caption_key[6]),
                                stream=True,
                            )
                            caption_engine_cache[caption_key] = pu_caption_engine
                        pu_people_positions = _compute_people_positions(pu_people_matches, image_path)
                        _pu_step("caption")
                        pu_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
                        with _prepare_ai_model_image(image_path) as pu_model_path:
                            pu_caption_out = pu_caption_engine.generate(
                                image_path=pu_model_path,
                                people=pu_person_names,
                                objects=existing_object_labels,
                                ocr_text=existing_ocr_text,
                                source_path=image_path,
                                album_title=pu_album_title,
                                printed_album_title=pu_printed_title,
                                people_positions=pu_people_positions,
                                debug_recorder=pu_prompt_debug.record,
                                debug_step="caption_refresh",
                            )
                            pu_faces_detected = (
                                (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                                if people_matcher
                                else 0
                            )
                        (
                            pu_local_people_present,
                            pu_local_estimated_people_count,
                        ) = _estimate_people_from_detections(
                            people_matches=pu_people_matches,
                            people_names=pu_person_names,
                            object_labels=existing_object_labels,
                            faces_detected=pu_faces_detected,
                        )
                        (
                            pu_people_present,
                            pu_estimated_people_count,
                        ) = _resolve_people_count_metadata(
                            requested_caption_engine=str(caption_key[0]),
                            caption_engine=pu_caption_engine,
                            model_image_path=pu_model_path,
                            people=pu_person_names,
                            objects=existing_object_labels,
                            ocr_text=existing_ocr_text,
                            source_path=image_path,
                            album_title=pu_album_title,
                            printed_album_title=pu_printed_title,
                            people_positions=pu_people_positions,
                            local_people_present=pu_local_people_present,
                            local_estimated_people_count=pu_local_estimated_people_count,
                            prompt_debug=pu_prompt_debug,
                            debug_step="people_count_refresh",
                        )
                        _emit_prompt_debug_artifact(pu_prompt_debug, dry_run=dry_run)
                        pu_caption_payload = _build_caption_metadata(
                            requested_engine=str(caption_key[0]),
                            effective_engine=str(pu_caption_out.engine),
                            fallback=bool(pu_caption_out.fallback),
                            error=str(pu_caption_out.error or ""),
                            engine_error=str(getattr(pu_caption_out, "engine_error", "") or ""),
                            model=str(caption_key[1] if caption_key[0] in {"local", "lmstudio"} else ""),
                            people_present=pu_people_present,
                            estimated_people_count=pu_estimated_people_count,
                        )
                        pu_ocr_model = str(
                            dict(det.get("ocr") or {}).get("model")
                            or (
                                effective.get("ocr_model", defaults["ocr_model"])
                                if str(effective.get("ocr_engine", defaults["ocr_engine"])).strip().lower()
                                in {"local", "lmstudio"}
                                else ""
                            )
                        )
                        pu_updated_det = _refresh_detection_model_metadata(
                            {
                                **det,
                                "people": pu_people_payload,
                                "caption": pu_caption_payload,
                            },
                            ocr_model=pu_ocr_model,
                            caption_model=(
                                str(pu_caption_engine.effective_model_name)
                                if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
                                else ""
                            ),
                        )
                    pu_subjects = _dedupe(
                        existing_object_labels + existing_ocr_keywords + ([pu_album_title] if pu_album_title else [])
                    )
                    pu_source_text = _build_dc_source(pu_album_title, image_path, _page_scan_filenames(image_path))

                    pu_people_detected = pu_faces_detected > 0 or len(pu_person_names) > 0
                    pu_people_identified = len(pu_person_names) > 0

                    if not dry_run:
                        pu_album_title = _require_album_title_for_title_page(
                            image_path=image_path,
                            album_title=_resolve_title_page_album_title(
                                image_path=image_path,
                                album_title=pu_album_title,
                                ocr_text=existing_ocr_text,
                            ),
                            context="people update",
                        )
                        date_engine = (
                            _get_date_engine(effective)
                            if date_estimation_enabled and not _has_dc_date(_dc_date_value(state))
                            else None
                        )
                        pu_dc_date = _resolve_dc_date(
                            existing_dc_date=_dc_date_value(state),
                            ocr_text=existing_ocr_text,
                            album_title=pu_album_title,
                            image_path=image_path,
                            date_engine=date_engine,
                            prompt_debug=pu_prompt_debug,
                        )
                        pu_date_time_original = _resolve_date_time_original(
                            dc_date=pu_dc_date,
                            date_time_original=str(state.get("date_time_original") or ""),
                        )
                        pu_source_text = _build_dc_source(pu_album_title, image_path, _page_scan_filenames(image_path))
                        pu_page_like = (
                            str((pu_updated_det.get("caption") or {}).get("effective_engine") or "").strip()
                            == "page-summary"
                        )
                        text_layers = _resolve_xmp_text_layers(
                            image_path=image_path,
                            ocr_text=existing_ocr_text,
                            page_like=pu_page_like,
                            ocr_authority_source=str(state.get("ocr_authority_source") or ""),
                            author_text=str(state.get("author_text") or ""),
                            scene_text=str(state.get("scene_text") or ""),
                        )
                        xmp_title, xmp_title_source = _compute_xmp_title(
                            image_path=image_path,
                            explicit_title=str(state.get("title") or ""),
                            title_source=str(state.get("title_source") or ""),
                            author_text=str(text_layers.get("author_text") or ""),
                        )
                        current_cast_signature = str(people_matcher.store_signature())
                        pu_proc = dict((pu_updated_det.get("processing") or {}))
                        pu_proc["cast_store_signature"] = current_cast_signature
                        if date_estimation_enabled or pu_dc_date:
                            pu_proc["date_estimate_input_hash"] = _date_estimate_input_hash(
                                existing_ocr_text,
                                pu_album_title,
                            )
                        if existing_location:
                            pu_updated_det["location"] = existing_location
                        pu_updated_det = {**pu_updated_det, "processing": pu_proc}
                        _write_sidecar_and_record(
                            sidecar_path,
                            image_path,
                            creator_tool=creator_tool,
                            person_names=pu_person_names,
                            subjects=pu_subjects,
                            title=xmp_title,
                            title_source=xmp_title_source,
                            description=str(state.get("description") or ""),
                            album_title=pu_album_title,
                            location_payload=existing_location,
                            source_text=pu_source_text,
                            ocr_text=existing_ocr_text,
                            author_text=str(text_layers.get("author_text") or ""),
                            scene_text=str(text_layers.get("scene_text") or ""),
                            detections_payload=pu_updated_det,
                            stitch_key=str(state.get("stitch_key") or ""),
                            ocr_authority_source=str(state.get("ocr_authority_source") or ""),
                            create_date=(
                                str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)
                            ),
                            dc_date=pu_dc_date,
                            date_time_original=pu_date_time_original,
                            ocr_ran=bool(state.get("ocr_ran") or True),
                            people_detected=pu_people_detected,
                            people_identified=pu_people_identified,
                            title_page_location=title_page_location,
                        )

                    if chain_gps_update_after_people and not dry_run:
                        existing_sidecar_state = read_ai_sidecar_state(sidecar_path)
                        existing_xmp_people = read_person_in_image(sidecar_path)
                    people_update_only = False
                    _pu_stop()
                    if not chain_gps_update_after_people:
                        processed += 1
                        completed_times.append(time.monotonic() - file_start)
                        eta_str2 = _format_eta(completed_times, len(files) - idx)
                        eta_part2 = f"  {eta_str2}" if eta_str2 else ""
                        print(
                            f"[{idx}/{len(files)}]{eta_part2}  ok    {image_path.name}",
                            flush=True,
                        )
                except Exception as exc:
                    failures += 1
                    _pu_stop()
                    emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")

                if not needs_full and not chain_gps_update_after_people:
                    _release_image_processing_lock(lock_path)
                    continue

        # ── GPS location step (re-run only location estimate + geocode) ───────
        if not needs_full and gps_update_only and not stdout_only:
            state = existing_sidecar_state
            if not isinstance(state, dict):
                _release_image_processing_lock(lock_path)
                continue
            file_start = time.monotonic()
            det = state.get("detections") or {}
            gps_ocr_text = _effective_sidecar_ocr_text(image_path, state)
            gps_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
            gps_people_names = _dedupe(
                [
                    str(r.get("name") or "")
                    for r in list(det.get("people") or [])
                    if isinstance(r, dict) and r.get("name")
                ]
            )
            gps_object_labels = [
                str(r.get("label") or "")
                for r in list(det.get("objects") or [])
                if isinstance(r, dict) and r.get("label")
            ]
            gps_album_title = str(state.get("album_title") or "").strip()
            gps_printed_title = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
            gps_existing_location_name = str((dict(det.get("location") or {})).get("query") or "").strip()

            eta_str = _format_eta(completed_times, len(files) - idx + 1)
            eta_part = f"  {eta_str}" if eta_str else ""
            prefix = f"[{idx}/{len(files)}]{eta_part}  {_display_work_label(image_path)}"
            print(prefix, flush=True)
            _gps_stop, _gps_step = _progress_ticker(prefix)

            try:
                caption_key = (
                    str(effective.get("caption_engine", defaults["caption_engine"])),
                    str(effective.get("caption_model", defaults["caption_model"])),
                    str(effective.get("caption_prompt", defaults["caption_prompt"])),
                    int(effective.get("caption_max_tokens", defaults["caption_max_tokens"])),
                    float(effective.get("caption_temperature", defaults["caption_temperature"])),
                    str(effective.get("lmstudio_base_url", defaults["lmstudio_base_url"])),
                    int(effective.get("caption_max_edge", defaults["caption_max_edge"])),
                )
                gps_caption_engine = caption_engine_cache.get(caption_key)
                if gps_caption_engine is None:
                    gps_caption_engine = _init_caption_engine(
                        engine=caption_key[0],
                        model_name=caption_key[1],
                        caption_prompt=caption_key[2],
                        max_tokens=int(caption_key[3]),
                        temperature=float(caption_key[4]),
                        lmstudio_base_url=caption_key[5],
                        max_image_edge=int(caption_key[6]),
                        stream=True,
                    )
                    caption_engine_cache[caption_key] = gps_caption_engine

                gps_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
                _gps_step("location")
                with _prepare_ai_model_image(image_path) as gps_model_path:
                    gps_latitude, gps_longitude, location_name = _resolve_location_metadata(
                        requested_caption_engine=str(caption_key[0]),
                        caption_engine=gps_caption_engine,
                        model_image_path=gps_model_path,
                        people=gps_people_names,
                        objects=gps_object_labels,
                        ocr_text=gps_ocr_text,
                        source_path=image_path,
                        album_title=gps_album_title,
                        printed_album_title=gps_printed_title,
                        people_positions={},
                        fallback_location_name=gps_existing_location_name,
                        prompt_debug=gps_prompt_debug,
                        debug_step="location_gps_step",
                    )
                    _gps_step("locations_shown")
                    gps_locations_shown, gps_locations_shown_ran = _resolve_locations_shown(
                        requested_caption_engine=str(caption_key[0]),
                        caption_engine=gps_caption_engine,
                        model_image_path=gps_model_path,
                        ocr_text=gps_ocr_text,
                        source_path=image_path,
                        album_title=gps_album_title,
                        printed_album_title=gps_printed_title,
                        geocoder=geocoder,
                        prompt_debug=gps_prompt_debug,
                        debug_step="locations_shown_gps_step",
                        artifact_recorder=(
                            lambda record: _append_geocode_artifact(image_path=image_path, record=record)
                        ),
                    )
                gps_location_payload = _resolve_location_payload(
                    geocoder=geocoder,
                    gps_latitude=gps_latitude,
                    gps_longitude=gps_longitude,
                    location_name=location_name,
                    artifact_recorder=(lambda record: _append_geocode_artifact(image_path=image_path, record=record)),
                    artifact_step="location_gps_step",
                )
                gps_location_payload, _ = _apply_title_page_location_config(
                    image_path=image_path,
                    location_payload=gps_location_payload,
                    title_page_location=title_page_location,
                )
                _emit_prompt_debug_artifact(gps_prompt_debug, dry_run=dry_run)

                if not dry_run:
                    gps_updated_det = {**det}
                    if gps_location_payload:
                        gps_updated_det["location"] = gps_location_payload
                    elif "location" in gps_updated_det:
                        del gps_updated_det["location"]
                    gps_updated_det["locations_shown"] = gps_locations_shown
                    gps_updated_det["location_shown_ran"] = gps_locations_shown_ran
                    gps_subjects = _dedupe(
                        gps_object_labels + gps_ocr_keywords + ([gps_album_title] if gps_album_title else [])
                    )
                    xmp_title, xmp_title_source = _compute_xmp_title(
                        image_path=image_path,
                        explicit_title=str(state.get("title") or ""),
                        title_source=str(state.get("title_source") or ""),
                        author_text=str(state.get("author_text") or ""),
                    )
                    _write_sidecar_and_record(
                        sidecar_path,
                        image_path,
                        creator_tool=creator_tool,
                        person_names=list(existing_xmp_people),
                        subjects=gps_subjects,
                        title=xmp_title,
                        title_source=xmp_title_source,
                        description=str(state.get("description") or ""),
                        album_title=gps_album_title,
                        location_payload=gps_location_payload,
                        source_text=str(state.get("source_text") or ""),
                        ocr_text=gps_ocr_text,
                        ocr_lang=str(state.get("ocr_lang") or ""),
                        author_text=str(state.get("author_text") or ""),
                        scene_text=str(state.get("scene_text") or ""),
                        detections_payload=gps_updated_det,
                        stitch_key=str(state.get("stitch_key") or ""),
                        ocr_authority_source=str(state.get("ocr_authority_source") or ""),
                        create_date=(
                            str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)
                        ),
                        dc_date=_dc_date_value(state),
                        date_time_original=str(state.get("date_time_original") or ""),
                        ocr_ran=bool(state.get("ocr_ran")),
                        people_detected=bool(state.get("people_detected")),
                        people_identified=bool(state.get("people_identified")),
                        title_page_location=title_page_location,
                    )

                processed += 1
                completed_times.append(time.monotonic() - file_start)
                _gps_stop()
                eta_str2 = _format_eta(completed_times, len(files) - idx)
                eta_part2 = f"  {eta_str2}" if eta_str2 else ""
                print(
                    f"[{idx}/{len(files)}]{eta_part2}  ok    {image_path.name}  [gps]",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                _gps_stop()
                emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")

            if not needs_full:
                _release_image_processing_lock(lock_path)
                continue
        # ─────────────────────────────────────────────────────────────────────

        file_start = time.monotonic()
        stop_ticker = None
        set_step = None
        if not stdout_only:
            eta_str = _format_eta(completed_times, len(files) - idx + 1)
            eta_part = f"  {eta_str}" if eta_str else ""
            prefix = f"[{idx}/{len(files)}]{eta_part}  {_display_work_label(image_path)}"
            print(prefix, flush=True)
            stop_ticker, set_step = _progress_ticker(prefix)
        album_title_hint = _resolve_album_title_hint(image_path)
        printed_album_title_hint = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
        upstream_page_state = _resolve_upstream_page_sidecar_state(image_path)
        upstream_context_ocr = str((upstream_page_state or {}).get("ocr_text") or "").strip()
        upstream_location_hint = _format_location_hint_from_state(upstream_page_state)
        if not album_title_hint:
            album_title_hint = str((upstream_page_state or {}).get("album_title") or "").strip()
        if not printed_album_title_hint:
            printed_album_title_hint = str((upstream_page_state or {}).get("album_title") or "").strip()
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))

        try:
            object_detector = None
            if bool(effective.get("enable_objects", True)):
                object_key = (
                    str(effective.get("model", defaults["model"])),
                    float(effective.get("object_threshold", defaults["object_threshold"])),
                )
                object_detector = object_detector_cache.get(object_key)
                if object_detector is None:
                    object_detector = _init_object_detector(
                        model_name=str(object_key[0]),
                        confidence=float(object_key[1]),
                    )
                    object_detector_cache[object_key] = object_detector

            caption_key = (
                str(effective.get("caption_engine", defaults["caption_engine"])),
                str(effective.get("caption_model", defaults["caption_model"])),
                str(effective.get("caption_prompt", defaults["caption_prompt"])),
                int(effective.get("caption_max_tokens", defaults["caption_max_tokens"])),
                float(effective.get("caption_temperature", defaults["caption_temperature"])),
                str(effective.get("lmstudio_base_url", defaults["lmstudio_base_url"])),
                int(effective.get("caption_max_edge", defaults["caption_max_edge"])),
            )
            caption_engine = caption_engine_cache.get(caption_key)
            if caption_engine is None:
                caption_engine = _init_caption_engine(
                    engine=caption_key[0],
                    model_name=caption_key[1],
                    caption_prompt=caption_key[2],
                    max_tokens=int(caption_key[3]),
                    temperature=float(caption_key[4]),
                    lmstudio_base_url=caption_key[5],
                    max_image_edge=int(caption_key[6]),
                    stream=not stdout_only,
                )
                caption_engine_cache[caption_key] = caption_engine

            ocr_key = (
                str(effective.get("ocr_engine", defaults["ocr_engine"])),
                str(effective.get("ocr_lang", defaults["ocr_lang"])),
                str(effective.get("ocr_model", defaults["ocr_model"])),
                normalize_lmstudio_base_url(str(effective.get("lmstudio_base_url", defaults["lmstudio_base_url"]))),
            )
            ocr_engine = ocr_engine_cache.get(ocr_key)
            if ocr_engine is None:
                ocr_engine = OCREngine(
                    engine=ocr_key[0],
                    language=ocr_key[1],
                    model_name=ocr_key[2],
                    base_url=ocr_key[3],
                )
                ocr_engine_cache[ocr_key] = ocr_engine

            scan_ocr_authority: ArchiveScanOCRAuthority | None = None
            if archive_stitched_ocr_required:
                scan_ocr_authority = _resolve_archive_scan_authoritative_ocr(
                    image_path=image_path,
                    group_paths=multi_scan_group_paths,
                    group_signature=multi_scan_group_signature,
                    cache=archive_scan_ocr_cache,
                    ocr_engine=ocr_engine,
                    step_fn=set_step,
                    stitched_image_dir=stitch_cap_dir,
                    debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
                )

            with prepare_image_layout(
                image_path,
                split_mode="off",
            ) as layout:
                person_names: list[str]
                subjects: list[str]
                description: str
                ocr_text: str
                payload: dict[str, Any]
                subphotos_xml: list[dict[str, Any]] | None = None
                people_count = 0
                object_count = 0
                analysis_mode = "single_image"
                split_applied = False
                subphoto_count = 0
                _scan_filenames = _page_scan_filenames(image_path)
                printed_album_title_hint = album_title_hint

                _stitched_cap_path = scan_ocr_authority.stitched_image_path if scan_ocr_authority is not None else None
                derived_ocr_override = _effective_sidecar_ocr_text(image_path, existing_sidecar_state)
                analysis_target = _stitched_cap_path or (layout.content_path if layout.page_like else image_path)
                people_analysis_source = (
                    analysis_target
                    if scan_ocr_authority is not None
                    else (layout.content_path if layout.page_like else image_path)
                )
                analysis = _run_image_analysis(
                    image_path=analysis_target,
                    people_image_path=people_analysis_source,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine=str(caption_key[0]),
                    ocr_engine_name=ocr_key[0],
                    ocr_language=ocr_key[1],
                    people_source_path=people_analysis_source,
                    people_bbox_offset=(_bounds_offset(layout.content_bounds) if layout.page_like else (0, 0)),
                    caption_source_path=(image_path if layout.page_like else analysis_target),
                    album_title=album_title_hint,
                    printed_album_title=printed_album_title_hint,
                    geocoder=geocoder,
                    step_fn=set_step,
                    extra_people_names=existing_xmp_people,
                    is_page_scan=layout.page_like,
                    ocr_text_override=(
                        scan_ocr_authority.ocr_text
                        if scan_ocr_authority is not None
                        else (derived_ocr_override or None)
                    ),
                    context_ocr_text=upstream_context_ocr,
                    context_location_hint=upstream_location_hint,
                    prompt_debug=prompt_debug,
                    title_page_location=title_page_location,
                )
                resolved_album_title = analysis.album_title or album_title_hint
                resolved_printed_album_title = resolved_album_title
                _store_album_printed_title_hint(
                    image_path,
                    printed_album_title_cache,
                    resolved_album_title,
                )
                person_names = _dedupe(analysis.people_names + existing_xmp_people)
                subjects = _dedupe(analysis.subjects + ([resolved_album_title] if resolved_album_title else []))
                description = (
                    _build_flat_page_description(analysis=analysis) if layout.page_like else analysis.description
                )
                ocr_text = analysis.ocr_text
                payload = _build_flat_payload(layout, analysis)
                people_count = len(analysis.people_names)
                object_count = len(analysis.object_labels)
                analysis_mode = "page_flat" if layout.page_like else "single_image"
                ocr_authority_hash = str(scan_ocr_authority.ocr_hash) if scan_ocr_authority is not None else ""

                payload = _refresh_detection_model_metadata(
                    payload,
                    ocr_model=(
                        str(ocr_engine.effective_model_name)
                        if str(ocr_key[0]).strip().lower() in {"local", "lmstudio"}
                        else ""
                    ),
                    caption_model=(
                        str(caption_engine.effective_model_name)
                        if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
                        else ""
                    ),
                )

                # Compute per-stage tracking flags for the XMP
                _ocr_ran_flag = str(effective.get("ocr_engine", defaults["ocr_engine"])).lower() != "none"
                _people_detected_flag = analysis.faces_detected > 0 or len(person_names) > 0
                _people_identified_flag = len(person_names) > 0

                if not dry_run:
                    location_payload = dict(payload.get("location") or {}) if isinstance(payload, dict) else {}
                    effective_location_payload = _effective_sidecar_location_payload(image_path, existing_sidecar_state)
                    if effective_location_payload:
                        location_payload = effective_location_payload
                    if location_payload:
                        payload["location"] = location_payload
                    img_w, img_h = _get_image_dimensions(image_path)
                    final_album_title = _require_album_title_for_title_page(
                        image_path=image_path,
                        album_title=_resolve_title_page_album_title(
                            image_path=image_path,
                            album_title=(resolved_album_title or _resolve_album_title_hint(image_path)),
                            ocr_text=ocr_text,
                        ),
                        context="write",
                    )
                    date_engine = (
                        _get_date_engine(effective)
                        if date_estimation_enabled and not _has_dc_date(_dc_date_value(existing_sidecar_state))
                        else None
                    )
                    final_dc_date = _resolve_dc_date(
                        existing_dc_date=_dc_date_value(existing_sidecar_state),
                        ocr_text=ocr_text,
                        album_title=final_album_title,
                        image_path=image_path,
                        date_engine=date_engine,
                        prompt_debug=prompt_debug,
                    )
                    final_date_time_original = _resolve_date_time_original(
                        dc_date=final_dc_date,
                        date_time_original=str((existing_sidecar_state or {}).get("date_time_original") or ""),
                    )
                    text_layers = _resolve_xmp_text_layers(
                        image_path=image_path,
                        ocr_text=ocr_text,
                        page_like=bool(layout.page_like),
                        ocr_authority_source=("archive_stitched" if scan_ocr_authority is not None else ""),
                        author_text=str(analysis.author_text or ""),
                        scene_text=str(analysis.scene_text or ""),
                    )
                    xmp_title, xmp_title_source = _compute_xmp_title(
                        image_path=image_path,
                        explicit_title=str(analysis.title or ""),
                        author_text=str(text_layers.get("author_text") or ""),
                    )
                    # Convert relative photo-region coords (0–1) to pixel bounds for MWG XMP regions
                    subphotos_xml = [
                        {
                            "index": i + 1,
                            "bounds": {
                                "x": round(r["x"] * img_w),
                                "y": round(r["y"] * img_h),
                                "width": round(r["w"] * img_w),
                                "height": round(r["h"] * img_h),
                            },
                            "description": r.get("author_text", ""),
                            "author_text": r.get("author_text", ""),
                            "scene_text": r.get("scene_text", ""),
                            "people": [],
                            "subjects": [],
                        }
                        for i, r in enumerate(analysis.image_regions or [])
                    ] or None
                    if people_matcher is not None:
                        current_cast_signature = str(people_matcher.store_signature())
                    stat = image_path.stat()
                    payload["processing"] = {
                        "processor_signature": PROCESSOR_SIGNATURE,
                        "settings_signature": settings_sig,
                        "cast_store_signature": (
                            current_cast_signature if bool(effective.get("enable_people", True)) else ""
                        ),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "date_estimate_input_hash": (
                            _date_estimate_input_hash(ocr_text, final_album_title)
                            if date_estimation_enabled or final_dc_date
                            else str((existing_sidecar_state or {}).get("date_estimate_input_hash") or "")
                        ),
                        "ocr_authority_signature": (
                            str(scan_ocr_authority.signature) if scan_ocr_authority is not None else ""
                        ),
                        "ocr_authority_hash": ocr_authority_hash,
                        "analysis_mode": str(analysis_mode),
                    }
                    _write_sidecar_and_record(
                        sidecar_path,
                        image_path,
                        creator_tool=creator_tool,
                        person_names=person_names,
                        subjects=subjects,
                        title=xmp_title,
                        title_source=xmp_title_source,
                        description=description,
                        album_title=final_album_title,
                        location_payload=location_payload,
                        source_text=_build_dc_source(
                            final_album_title,
                            image_path,
                            _scan_filenames,
                        ),
                        ocr_text=ocr_text,
                        ocr_lang=str(analysis.ocr_lang or ""),
                        author_text=str(text_layers.get("author_text") or ""),
                        scene_text=str(text_layers.get("scene_text") or ""),
                        detections_payload=payload,
                        subphotos=subphotos_xml,
                        ocr_authority_source=("archive_stitched" if scan_ocr_authority is not None else ""),
                        create_date=read_embedded_create_date(image_path),
                        dc_date=final_dc_date,
                        date_time_original=final_date_time_original,
                        ocr_ran=_ocr_ran_flag,
                        people_detected=_people_detected_flag,
                        people_identified=_people_identified_flag,
                        title_page_location=title_page_location,
                    )

            _emit_prompt_debug_artifact(prompt_debug, dry_run=dry_run)
            processed += 1
            completed_times.append(time.monotonic() - file_start)
            if stop_ticker is not None:
                stop_ticker()
            if stdout_only:
                caption_meta = dict(payload.get("caption") or {}) if isinstance(payload, dict) else {}
                fallback_error = str(caption_meta.get("error") or "").strip()
                if bool(caption_meta.get("fallback")) and fallback_error:
                    emit_error(f"[{idx}/{len(files)}] warn  {image_path.name}: caption fallback: {fallback_error}")
                print(f"{image_path.name}: {description}" if description else image_path.name)
            else:
                eta_str = _format_eta(completed_times, len(files) - idx)
                eta_part = f"  {eta_str}" if eta_str else ""
                print(
                    f"[{idx}/{len(files)}]{eta_part}  ok    {image_path.name}",
                    flush=True,
                )
            _mirror_page_sidecars(image_path)
        except Exception as exc:
            failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=dry_run)
            if stop_ticker is not None:
                stop_ticker()
            emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")
        _release_image_processing_lock(lock_path)

    stitch_failures = 0
    if bool(getattr(args, "stitch_scans", False)):
        emit_info("Scan stitch pass skipped: archive scan OCR stitching now happens during normal processing.")

    if not stdout_only:
        print("\nSummary")
        print(f"- Processed: {processed}")
        print(f"- Skipped:   {skipped}")
        print(f"- Failed:    {failures + stitch_failures}")
    _release_batch_processing_lock(batch_lock_path)
    stitch_cap_td.cleanup()
    return 1 if (failures or stitch_failures) else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
