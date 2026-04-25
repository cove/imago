from __future__ import annotations

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
from ..naming import is_archive_dir, is_pages_dir, is_photos_dir

# Re-exports from extracted modules — keep backward compatibility for tests and callers.
from .ai_index_args import (  # noqa: F401
    DEFAULT_CAST_STORE,
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
    album_keys: list[str] = []
    for path in files:
        album_key = _album_identity_key(path)
        if album_key not in album_keys:
            album_keys.append(album_key)
    album_shards = {album_key: idx % shard_count for idx, album_key in enumerate(album_keys)}
    return [path for path in files if album_shards.get(_album_identity_key(path)) == shard_index]


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
        from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

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
        in_archive = any(is_archive_dir(name) for name in parent_names)
        in_view = any(is_pages_dir(name) for name in parent_names)
        in_photos = any(is_photos_dir(name) for name in parent_names)
        if in_archive and include_archive:
            files.append(path)
            continue
        if include_view and (in_view or in_photos):
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
    return is_archive_dir(image_path.parent)


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
        in_archive = any(is_archive_dir(name) for name in parent_names)
        in_view = any(is_pages_dir(name) for name in parent_names)
        in_photos = any(is_photos_dir(name) for name in parent_names)
        if in_archive and include_archive:
            filtered.append(image_path)
            continue
        if include_view and (in_view or in_photos):
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


def _write_sidecar_and_record(*args: Any, **kwargs: Any) -> None:
    """Re-exported from ai_index_runner for backward compatibility."""
    from .ai_index_runner import _write_sidecar_and_record as _impl

    return _impl(*args, **kwargs)


def run(argv: list[str] | None = None) -> int:
    from .ai_index_runner import IndexRunner

    return IndexRunner(argv).run()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
