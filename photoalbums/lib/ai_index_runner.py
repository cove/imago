from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import PHOTO_ALBUMS_DIR
from .ai_model_settings import default_lmstudio_base_url, default_ocr_model
from .model_store import YOLO_MODEL_DIR

log = logging.getLogger(__name__)

from ..naming import (
    is_archive_dir,
    is_pages_dir,
    is_photos_dir,
    parse_album_filename,
)
from .ai_album_titles import (
    _album_identity_key,
    _derived_name_match,
    _expand_album_title_dependencies,
    _is_album_title_source_candidate,
    _require_album_title_for_title_page,
    _resolve_album_printed_title_hint,
    _resolve_album_title_from_sidecars,
    _resolve_album_title_hint,
    _resolve_title_page_album_title,
    _scan_name_match,
    _store_album_printed_title_hint,
)
from .ai_caption import (
    DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
    CaptionEngine,
    normalize_lmstudio_base_url,
    resolve_caption_model,
)
from .ai_date import DateEstimateEngine
from .ai_geocode import NominatimGeocoder

# Index functions (previously ai_index.py own code)


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
        return

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
    person_hint_count: int = 0,
) -> list[Any]:
    last_exc: Exception | None = None
    for attempt in range(CAST_STORE_RETRY_ATTEMPTS):
        try:
            kwargs = {
                "source_path": source_path,
                "bbox_offset": bbox_offset,
                "hint_text": hint_text,
            }
            if person_hint_count:
                kwargs["person_hint_count"] = person_hint_count
            return people_matcher.match_image(image_path, **kwargs)
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


def _apply_title_page_location_config(
    *,
    image_path: Path,
    location_payload: dict[str, Any] | None,
    detections_payload: dict[str, Any] | None = None,
    title_page_location: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    loc = dict(location_payload or {})
    _, _, _, page_str = parse_album_filename(image_path.name)
    if (page_str.isdigit() and int(page_str) == 1) or (
        str(loc.get("source") or "").strip() == TITLE_PAGE_LOCATION_SOURCE
    ):
        loc = {}
    if not isinstance(detections_payload, dict):
        return loc, detections_payload
    detections = dict(detections_payload)
    if loc:
        detections["location"] = dict(loc)
    elif "location" in detections:
        del detections["location"]
    return loc, detections


def _known_sidecar_needs_reprocess(path: Path, stat: os.stat_result, sidecar_state: dict[str, Any]) -> bool:
    if str(sidecar_state.get("processor_signature") or "") != PROCESSOR_SIGNATURE:
        return True
    recorded_size = int(sidecar_state.get("size") or -1)
    recorded_mtime = int(sidecar_state.get("mtime_ns") or -1)
    if int(stat.st_size) != recorded_size or int(stat.st_mtime_ns) != recorded_mtime:
        return True
    return not has_current_sidecar(path)


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
        return _known_sidecar_needs_reprocess(path, stat, sidecar_state)
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


def _clean_existing_dc_date(existing_dc_date: str | list[str]) -> str | list[str] | None:
    if isinstance(existing_dc_date, list):
        clean = [str(item or "").strip() for item in existing_dc_date if str(item or "").strip()]
        return clean if clean else None
    clean = str(existing_dc_date or "").strip()
    return clean if clean else None


def _estimate_dc_date_from_engine(
    *,
    ocr_text: str,
    album_title: str,
    image_path: Path,
    date_engine: DateEstimateEngine,
    prompt_debug: PromptDebugSession | None,
) -> str:
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


def _resolve_dc_date(
    *,
    existing_dc_date: str | list[str],
    ocr_text: str,
    album_title: str,
    image_path: Path,
    date_engine: DateEstimateEngine | None,
    prompt_debug: PromptDebugSession | None,
) -> str | list[str]:
    cleaned = _clean_existing_dc_date(existing_dc_date)
    if cleaned is not None:
        return cleaned
    if date_engine is None:
        return ""
    return _estimate_dc_date_from_engine(
        ocr_text=ocr_text,
        album_title=album_title,
        image_path=image_path,
        date_engine=date_engine,
        prompt_debug=prompt_debug,
    )


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

from .ai_index_analysis import (
    ArchiveScanOCRAuthority,
    ImageAnalysis,
    _build_caption_metadata,
    _estimate_people_from_detections,
    _get_image_dimensions,
    _prepare_ai_model_image,
    _refresh_detection_model_metadata,
    _resolve_people_count_metadata,
    _run_image_analysis,
    _serialize_people_matches,
)

# CLI argument parsing (previously ai_index_args.py)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"


def _explicit_cli_flags(argv: list[str] | None) -> set[str]:
    flags: set[str] = set()
    for item in list(argv or []):
        text = str(item or "")
        if not text.startswith("--"):
            continue
        flags.add(text.split("=", 1)[0])
    return flags


def _resolve_caption_prompt(prompt_text: str, prompt_file: str) -> str:
    file_text = str(prompt_file or "").strip()
    if file_text:
        path = Path(file_text).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise SystemExit(f"Caption prompt file does not exist: {path}") from exc
        except OSError as exc:
            raise SystemExit(f"Could not read caption prompt file {path}: {exc}") from exc
    return str(prompt_text or "").strip()


def _absolute_cli_path(path_text: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path_text).expanduser())))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index photo album images with cast people matching, YOLO objects, OCR, and XMP sidecars.",
    )
    parser.add_argument(
        "--photos-root",
        default=str(PHOTO_ALBUMS_DIR),
        help="Photo Albums root directory.",
    )
    parser.add_argument("--cast-store", default=str(DEFAULT_CAST_STORE), help="Cast store directory.")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument(
        "--object-threshold",
        type=float,
        default=0.30,
        help="Object detection confidence.",
    )
    parser.add_argument(
        "--people-threshold",
        type=float,
        default=0.72,
        help="Face similarity threshold.",
    )
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size in pixels.")
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "local", "lmstudio"],
        default="none",
        help="OCR backend.",
    )
    parser.add_argument(
        "--ocr-model",
        default=default_ocr_model(),
        help="Optional model id/path used by the selected OCR engine.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language.")
    parser.add_argument(
        "--caption-engine",
        choices=["none", "lmstudio"],
        default="lmstudio",
        help="Caption backend for XMP description.",
    )
    parser.add_argument(
        "--caption-model",
        default="",
        help="Optional model id/path used by the selected caption engine.",
    )
    parser.add_argument(
        "--caption-prompt",
        dest="caption_prompt",
        default="",
        help="Exact prompt text for model captioning. When set, built-in prompt hints are disabled.",
    )
    parser.add_argument(
        "--caption-prompt-file",
        dest="caption_prompt_file",
        default="",
        help="Read exact model caption prompt text from a file. Overrides --caption-prompt when set.",
    )
    parser.add_argument(
        "--local-prompt",
        dest="caption_prompt",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-prompt-file",
        dest="caption_prompt_file",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-prompt",
        dest="caption_prompt",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-prompt-file",
        dest="caption_prompt_file",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default=default_lmstudio_base_url(),
        help="Base URL for the LM Studio OpenAI-compatible API.",
    )
    parser.add_argument(
        "--caption-max-tokens",
        type=int,
        default=96,
        help="Max new tokens for caption models.",
    )
    parser.add_argument(
        "--caption-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for local captioning.",
    )
    parser.add_argument(
        "--caption-max-edge",
        type=int,
        default=0,
        help="Optional long-edge cap, in pixels, applied only during caption generation.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional processing limit.")
    parser.add_argument(
        "--photo",
        default="",
        help="Process a single photo file. Bypasses discovery and implies --force.",
    )
    parser.add_argument(
        "--album",
        default="",
        help="Filter to photos whose parent directory name contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--photo-offset",
        type=int,
        default=0,
        help="Skip first N discovered images. Use with --max-images to process a range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore manifest and process all files. Equivalent to --reprocess-mode=all.",
    )
    parser.add_argument(
        "--reprocess-mode",
        default="unprocessed",
        choices=["unprocessed", "new_only", "errors_only", "outdated", "cast_changed", "gps", "all"],
        help=(
            "Controls which images are processed. "
            "'unprocessed' (default): images with missing or stale sidecar. "
            "'new_only': only images with no manifest entry (never indexed). "
            "'errors_only': only images whose sidecar contains a processing error. "
            "'outdated': only images where the sidecar is older than the image file. "
            "'cast_changed': only images needing people re-detection when the cast store changes. "
            "'gps': re-run only the GPS location estimate step for already-indexed images. "
            "'all': force reprocess everything (same as --force)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write sidecar/manifest.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated caption text to stdout only. Implies --dry-run and forced reprocessing.",
    )
    parser.add_argument(
        "--include-view",
        action="store_true",
        help="Include files in rendered *_Pages and *_Photos folders.",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include files in *_Archive folders.",
    )
    parser.add_argument("--disable-people", action="store_true", help="Disable cast people matching.")
    parser.add_argument("--disable-objects", action="store_true", help="Disable object detection.")
    parser.add_argument(
        "--ignore-render-settings",
        action="store_true",
        help="Ignore per-archive render_settings.json overrides.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--stitch-scans",
        action="store_true",
        help=(
            "Deprecated. Multi-scan archive page OCR now uses a temporary stitched composite during normal processing."
        ),
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help="Comma-separated file extensions to include.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split discovered files into N deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process when --shard-count is greater than 1.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help=(
            "Comma-separated list of step names to force re-run unconditionally "
            "(e.g. 'caption', 'ocr,people'). Downstream steps are also marked stale."
        ),
    )
    return parser.parse_args(argv)


# Engine cache (previously ai_index_engine_cache.py + ai_objects.py)

# Propagate-to-crops (previously ai_index_propagate.py)


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
    return str(region_state.get("caption") or region_state.get("caption_hint") or "")


def _resolve_crop_metadata(
    region_state: dict,
    locations_shown: list,
    page_location: dict[str, Any],
    names_from_region: list[str],
    existing_person_names: list[str],
    geocoder: Any = None,
) -> tuple[dict, list, list[str], str | None]:
    region_override = dict(region_state.get("location_override") or {})
    region_assigned = dict(region_state.get("location_payload") or {})
    caption = _region_caption(region_state)
    photo_location = region_state.get("photo_location")  # str | None
    crop_location = resolve_crop_location(
        region_location_override=region_override,
        region_location_assigned=region_assigned,
        caption=caption,
        locations_shown=locations_shown,
        page_location=page_location,
        photo_location=photo_location,
    )
    crop_location = _materialize_crop_location(crop_location, geocoder=geocoder)
    crop_locations_shown = resolve_crop_locations_shown(
        region_location_override=region_override,
        region_location_assigned=region_assigned,
        caption=caption,
        locations_shown=locations_shown,
    )
    if crop_location and (
        not crop_locations_shown
        or not all(str(location.get("gps_latitude") or "").strip() for location in crop_locations_shown)
    ):
        materialized_location_shown = location_shown_from_payload(crop_location)
        crop_locations_shown = [materialized_location_shown] if materialized_location_shown else crop_locations_shown
    new_person_names = resolve_person_in_image(
        _dedupe(names_from_region + existing_person_names),
        locations_shown=locations_shown,
        location_payload=crop_location,
    )
    photo_est_date = region_state.get("photo_est_date")  # str | None
    return crop_location, crop_locations_shown, new_person_names, photo_est_date


def _materialize_crop_location(payload: dict[str, Any] | None, *, geocoder: Any = None) -> dict[str, Any]:
    return materialize_location_payload(payload, geocoder=geocoder)


def _build_detections_payload(existing_state: dict, crop_location: dict, step_timestamp: str) -> dict:
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


_PAGE_TEXT_PREFIX = "Captions from the original album page this photo was cropped from (may not be specific to this photo):"


def _write_propagated_crop(
    crop_xmp: Path,
    existing_state: dict,
    *,
    crop_location: dict,
    crop_locations_shown: list,
    new_person_names: list[str],
    step_timestamp: str,
    region_caption: str = "",
    page_dc_date_values: list[str] | None = None,
    photo_est_date: str | None = None,
    page_text: str = "",
) -> None:
    detections_payload = _build_detections_payload(existing_state, crop_location, step_timestamp)
    if photo_est_date:
        dc_date = [photo_est_date]
    else:
        dc_date = list(existing_state.get("dc_date_values") or page_dc_date_values or [])
    description = _str_field(existing_state, "description") or str(region_caption or "").strip()
    parent_ocr_text = _str_field(existing_state, "parent_ocr_text")
    if page_text and not parent_ocr_text:
        parent_ocr_text = f"{_PAGE_TEXT_PREFIX}\n{page_text}"
    write_xmp_sidecar(
        crop_xmp,
        person_names=new_person_names,
        subjects=list(existing_state.get("subjects") or []),
        title=_str_field(existing_state, "title"),
        title_source=_str_field(existing_state, "title_source"),
        description=description,
        ocr_text=_str_field(existing_state, "ocr_text"),
        parent_ocr_text=parent_ocr_text,
        ocr_lang=_str_field(existing_state, "ocr_lang"),
        author_text=_str_field(existing_state, "author_text"),
        scene_text=_str_field(existing_state, "scene_text"),
        album_title=_str_field(existing_state, "album_title"),
        gps_latitude=_str_field(crop_location, "gps_latitude").strip(),
        gps_longitude=_str_field(crop_location, "gps_longitude").strip(),
        location_address=_str_field(crop_location, "address").strip(),
        location_city=_str_field(crop_location, "city").strip(),
        location_state=_str_field(crop_location, "state").strip(),
        location_country=_str_field(crop_location, "country").strip(),
        location_sublocation=_str_field(crop_location, "sublocation").strip(),
        locations_shown=crop_locations_shown,
        source_text=_str_field(existing_state, "source_text"),
        detections_payload=detections_payload,
        create_date=_str_field(existing_state, "create_date"),
        dc_date=dc_date,
        date_time_original=_str_field(existing_state, "date_time_original"),
        ocr_ran=bool(existing_state.get("ocr_ran", False)),
        people_detected=bool(new_person_names),
        people_identified=bool(new_person_names),
    )


def _propagate_one_crop(
    crop_xmp: Path,
    region_state: dict,
    *,
    names_from_region: list[str],
    locations_shown: list,
    page_location: dict[str, Any],
    step_timestamp: str,
    page_dc_date_values: list[str] | None = None,
    page_text: str = "",
    default_location: dict[str, Any] | None = None,
    geocoder: Any = None,
) -> bool:
    if not crop_xmp.is_file():
        return False
    existing_state = read_ai_sidecar_state(crop_xmp)
    if not isinstance(existing_state, dict):
        return False
    existing_person_names = read_person_in_image(crop_xmp)
    crop_location, crop_locations_shown, new_person_names, photo_est_date = _resolve_crop_metadata(
        region_state,
        locations_shown,
        page_location,
        names_from_region,
        existing_person_names,
        geocoder=geocoder,
    )
    if not crop_location and default_location:
        crop_location = dict(default_location)
    _write_propagated_crop(
        crop_xmp,
        existing_state,
        crop_location=crop_location,
        crop_locations_shown=crop_locations_shown,
        new_person_names=new_person_names,
        step_timestamp=step_timestamp,
        region_caption=_region_caption(region_state),
        page_dc_date_values=page_dc_date_values,
        photo_est_date=photo_est_date,
        page_text=page_text,
    )
    return True


def _location_needs_geocoding(location: dict) -> bool:
    normalized = normalize_location_payload(location)
    has_address = bool(normalized.get("address") or str(location.get("name") or "").strip())
    has_gps = bool(normalized.get("gps_latitude"))
    return has_address and not has_gps


def _propagation_needs_geocoder(locations_shown: list, regions: list) -> bool:
    if any(_location_needs_geocoding(loc) for loc in locations_shown if isinstance(loc, dict)):
        return True
    for region in regions:
        if not isinstance(region, dict):
            continue
        photo_location = region.get("photo_location")
        if photo_location is None:
            # Old region without AI per-photo data: fall back to caption heuristic
            if location_payload_from_caption(_region_caption(region)):
                return True
        elif photo_location:
            return True
    return False


def _propagate_all_crops(
    crop_paths: list[Path],
    regions: list[dict],
    *,
    region_person_names: list[list[str]],
    locations_shown: list,
    location_payload: dict[str, Any],
    step_timestamp: str,
    page_dc_date_values: list[str],
    page_text: str,
    default_location: dict[str, Any] | None,
    geocoder: Any,
) -> int:
    crops_updated = 0
    for i, crop_path in enumerate(crop_paths):
        names_from_region = region_person_names[i] if i < len(region_person_names) else []
        region_state = regions[i] if i < len(regions) else {}
        if _propagate_one_crop(
            crop_path.with_suffix(".xmp"),
            region_state,
            names_from_region=names_from_region,
            locations_shown=locations_shown,
            page_location=location_payload,
            step_timestamp=step_timestamp,
            page_dc_date_values=page_dc_date_values,
            page_text=page_text,
            default_location=default_location,
            geocoder=geocoder,
        ):
            crops_updated += 1
    return crops_updated


def run_propagate_to_crops(
    image_path: Path,
    *,
    location_payload: dict[str, Any],
    people_payload: list[dict[str, Any]],
    default_location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Propagate location GPS and person names from page XMP to each crop XMP.

    Returns a dict with a 'crops_updated' count (for diagnostic use).
    Eligible to return None if engine not configured, but this step always runs.
    """
    crop_paths = _find_crop_paths_for_page(image_path)
    if not crop_paths:
        return {"crops_updated": 0}

    sidecar_path = image_path.with_suffix(".xmp")
    img_w, img_h = _get_image_dimensions_safe(image_path)
    regions = _read_regions_safe(sidecar_path, img_w, img_h)
    region_person_names: list[list[str]] = [list(r.get("person_names") or []) for r in regions]
    locations_shown = read_locations_shown(sidecar_path)
    page_state = read_ai_sidecar_state(sidecar_path)
    page_dc_date_values = list((page_state or {}).get("dc_date_values") or [])
    page_text = str((page_state or {}).get("ocr_text") or "").strip()
    step_timestamp = xmp_datetime_now()
    geocoder = None
    if _propagation_needs_geocoder(locations_shown, regions):
        from .ai_geocode import NominatimGeocoder  # pylint: disable=import-outside-toplevel

        geocoder = NominatimGeocoder()

    crops_updated = _propagate_all_crops(
        crop_paths,
        regions,
        region_person_names=region_person_names,
        locations_shown=locations_shown,
        location_payload=location_payload,
        step_timestamp=step_timestamp,
        page_dc_date_values=page_dc_date_values,
        page_text=page_text,
        default_location=default_location,
        geocoder=geocoder,
    )
    return {"crops_updated": crops_updated}


PROCESSOR_SIGNATURE = "page_split_v17_people_recovery_any_people"


# YOLO object detection (previously ai_objects.py)


def _resolve_model_reference(model_name: str) -> tuple[str, Path | None]:
    text = str(model_name or "").strip()
    if not text:
        text = "models/yolo11n.pt"

    path = Path(text).expanduser()
    # Keep explicit paths unchanged so callers can still opt into custom models.
    if path.is_absolute() or any(part not in {"", "."} for part in path.parts[:-1]):
        return str(path), None

    model_file = path.name
    if model_file.lower().endswith(".pt"):
        YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return model_file, YOLO_MODEL_DIR
    return text, None


@contextmanager
def _pushd(path: Path):
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


@dataclass
class ObjectDetection:
    label: str
    score: float


def _boxes_to_label_scores(boxes, names: dict) -> dict[str, float]:
    cls_vals = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
    conf_vals = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    labels_by_name: dict[str, float] = {}
    for idx, raw in enumerate(cls_vals):
        label = str(names.get(int(raw), int(raw)))
        score = float(conf_vals[idx]) if idx < len(conf_vals) else 0.0
        current = labels_by_name.get(label)
        if current is None or score > current:
            labels_by_name[label] = score
    return labels_by_name


class YOLOObjectDetector:
    def __init__(
        self,
        *,
        model_name: str = "models/yolo11n.pt",
        confidence: float = 0.30,
        max_detections: int = 100,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "Ultralytics is required for object detection. Run 'uv sync' to install project dependencies."
            ) from exc

        model_ref, model_dir = _resolve_model_reference(model_name)
        if model_dir is None:
            self._model = YOLO(model_ref)
        else:
            # When downloading stock YOLO weights, keep them under repo-root models/.
            with _pushd(model_dir):
                self._model = YOLO(model_ref)
        self.model_name = str(model_name or "models/yolo11n.pt")
        self.confidence = float(confidence)
        self.max_detections = int(max_detections)

    def detect_image(self, image_path: str | Path) -> list[ObjectDetection]:
        import cv2

        img = cv2.imread(str(image_path))
        if img is not None and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        source = img if img is not None else str(image_path)
        results = self._model.predict(
            source=source,
            conf=self.confidence,
            max_det=self.max_detections,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {}) or {}
        labels_by_name = _boxes_to_label_scores(boxes, names)
        out = [ObjectDetection(label=label, score=score) for label, score in labels_by_name.items()]
        out.sort(key=lambda row: row.score, reverse=True)
        return out


def _init_people_matcher(
    *,
    cast_store: Path,
    min_similarity: float,
    min_face_size: int,
) -> Any | None:
    if cast_store is None:
        return None
    from .ai_people import CastPeopleMatcher

    return CastPeopleMatcher(
        cast_store_dir=cast_store,
        min_similarity=float(min_similarity),
        min_face_size=int(min_face_size),
    )


def _init_object_detector(
    *,
    model_name: str,
    confidence: float,
) -> Any | None:
    if not str(model_name or "").strip():
        return None
    return YOLOObjectDetector(
        model_name=str(model_name),
        confidence=float(confidence),
    )


def _init_caption_engine(
    *,
    engine: str,
    model_name: str,
    caption_prompt: str,
    max_tokens: int,
    temperature: float,
    lmstudio_base_url: str,
    max_image_edge: int,
    stream: bool = False,
    thinking: bool = False,
    override_sources: dict[str, str] | None = None,
):
    kwargs = {
        "engine": str(engine),
        "model_name": str(model_name),
        "caption_prompt": str(caption_prompt),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "lmstudio_base_url": str(lmstudio_base_url),
        "max_image_edge": int(max_image_edge),
        "stream": stream,
        "thinking": thinking,
    }
    if override_sources:
        kwargs["override_sources"] = dict(override_sources)
    return CaptionEngine(**kwargs)


def _init_date_engine(
    *,
    engine: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    lmstudio_base_url: str,
):
    return DateEstimateEngine(
        engine=str(engine),
        model_name=str(model_name),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        lmstudio_base_url=str(lmstudio_base_url),
    )


def _settings_signature(settings: dict[str, Any]) -> str:
    caption_engine = str(settings.get("caption_engine", "lmstudio"))
    caption_model = resolve_caption_model(
        caption_engine,
        str(settings.get("caption_model", "")),
    )
    compact = {
        "processor_signature": PROCESSOR_SIGNATURE,
        "skip": bool(settings.get("skip", False)),
        "enable_people": bool(settings.get("enable_people", True)),
        "enable_objects": bool(settings.get("enable_objects", True)),
        "ocr_engine": str(settings.get("ocr_engine", "none")),
        "ocr_lang": str(settings.get("ocr_lang", "eng")),
        "ocr_model": str(settings.get("ocr_model", "")),
        "people_threshold": float(settings.get("people_threshold", 0.72)),
        "object_threshold": float(settings.get("object_threshold", 0.30)),
        "min_face_size": int(settings.get("min_face_size", 40)),
        "model": str(settings.get("model", "models/yolo11n.pt")),
        "caption_engine": caption_engine,
        "caption_model": caption_model,
        "caption_prompt": str(settings.get("caption_prompt", "")),
        "caption_max_tokens": int(settings.get("caption_max_tokens", 96)),
        "caption_temperature": float(settings.get("caption_temperature", 0.2)),
        "caption_max_edge": int(settings.get("caption_max_edge", 0)),
        "lmstudio_base_url": normalize_lmstudio_base_url(
            str(settings.get("lmstudio_base_url", default_lmstudio_base_url()))
        ),
    }
    return json.dumps(compact, sort_keys=True, ensure_ascii=True)


from .ai_index_scan import (
    _bounds_offset,
    _build_dc_source,
    _build_flat_page_description,
    _build_flat_payload,
    _dc_source_needs_refresh,
    _hash_text,
    _page_scan_filenames,
    _resolve_archive_scan_authoritative_ocr,
    _scan_group_paths,
    _scan_group_signature,
    _scan_number,
    _scan_page_key,
)
from .ai_index_steps import StepRunner
from .ai_location import (
    _has_legacy_ai_locations_shown_gps,
    _xmp_gps_to_decimal,
)
from .ai_metadata import MetadataEngine
from .ai_ocr import OCREngine
from .ai_page_layout import prepare_image_layout
from .metadata_resolver import (
    location_payload_from_caption,
    location_shown_from_payload,
    materialize_location_payload,
    normalize_location_payload,
    resolve_crop_location,
    resolve_crop_locations_shown,
)

# Processing locks (previously ai_processing_locks.py)

PROCESSING_LOCK_SUFFIX = ".photoalbums-ai.lock"
BATCH_LOCK_SUFFIX = ".photoalbums-ai.batch.lock"
JOB_ID_ENV = "IMAGO_JOB_ID"


def _processing_lock_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}{PROCESSING_LOCK_SUFFIX}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_processing_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _release_image_processing_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    for attempt in range(5):
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            log.debug("Lock file already removed: %s", lock_path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) != 32:
                raise
            if attempt == 4:
                return
            time.sleep(0.1)


def _release_batch_processing_lock(lock_path: Path | None) -> None:
    _release_image_processing_lock(lock_path)


def _clear_stale_processing_lock(lock_path: Path) -> bool:
    payload = _read_processing_lock(lock_path)
    pid = payload.get("pid")
    if isinstance(pid, int) and pid > 0 and not _pid_alive(pid):
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
        return True
    return False


def _cleanup_stale_processing_locks(photos_root: Path) -> list[Path]:
    cleaned: list[Path] = []
    if not photos_root.exists():
        return cleaned

    batch_lock_path = _batch_processing_lock_path(photos_root)
    if _clear_stale_processing_lock(batch_lock_path):
        cleaned.append(batch_lock_path)

    for lock_path in photos_root.rglob(f"*{PROCESSING_LOCK_SUFFIX}"):
        if _clear_stale_processing_lock(lock_path):
            cleaned.append(lock_path)
    return cleaned


def _acquire_image_processing_lock(image_path: Path) -> Path:
    lock_path = _processing_lock_path(image_path)
    payload = {
        "image_path": str(image_path.resolve()),
        "pid": os.getpid(),
        "job_id": str(os.environ.get(JOB_ID_ENV) or "").strip(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _clear_stale_processing_lock(lock_path):
                continue
            current = _read_processing_lock(lock_path)
            owner_parts = []
            job_id = str(current.get("job_id") or "").strip()
            if job_id:
                owner_parts.append(f"job {job_id}")
            pid = current.get("pid")
            if isinstance(pid, int):
                owner_parts.append(f"pid {pid}")
            owner = ", ".join(owner_parts) if owner_parts else str(lock_path)
            raise RuntimeError(f"already processing {image_path.name} ({owner})")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return lock_path
    raise RuntimeError(f"could not acquire processing lock for {image_path.name}")


def _batch_processing_lock_path(photos_root: Path) -> Path:
    return photos_root / BATCH_LOCK_SUFFIX


def _acquire_batch_processing_lock(photos_root: Path) -> Path:
    lock_path = _batch_processing_lock_path(photos_root)
    payload = {
        "photos_root": str(photos_root.resolve()),
        "pid": os.getpid(),
        "job_id": str(os.environ.get(JOB_ID_ENV) or "").strip(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _clear_stale_processing_lock(lock_path):
                continue
            current = _read_processing_lock(lock_path)
            owner_parts = []
            current_root = str(current.get("photos_root") or "").strip()
            if current_root:
                owner_parts.append(current_root)
            job_id = str(current.get("job_id") or "").strip()
            if job_id:
                owner_parts.append(f"job {job_id}")
            pid = current.get("pid")
            if isinstance(pid, int):
                owner_parts.append(f"pid {pid}")
            owner = ", ".join(owner_parts) if owner_parts else str(lock_path)
            raise RuntimeError(f"another photoalbums ai batch run is already active ({owner})")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return lock_path
    raise RuntimeError("could not acquire photoalbums ai batch lock")


from .ai_prompt_assets import load_params
from .ai_render_settings import (
    find_archive_dir_for_image,
    load_render_settings,
    resolve_effective_settings,
)
from .ai_sidecar_state import (
    _compute_xmp_title,
    _effective_sidecar_album_title,
    _effective_sidecar_location_payload,
    _effective_sidecar_ocr_text,
    _resolve_xmp_text_layers,
    _sidecar_current_for_paths,
    _xmp_timestamp_from_path,
    has_current_sidecar,
    has_valid_sidecar,
    read_embedded_create_date,
)
from .album_sets import find_archive_set_by_photos_root
from .metadata_resolver import resolve_person_in_image
from .prompt_debug import PromptDebugSession
from .xmp_review import load_ai_xmp_review
from .xmp_sidecar import (
    _dedupe,
    _resolve_date_time_original,
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    read_pipeline_state,
    read_region_list,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
    xmp_datetime_now,
)


def _write_sidecar_and_record(
    sidecar_path: Path,
    image_path: Path,
    *,
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
    replace_dc_date: bool = False,
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
    resolved_person_names = resolve_person_in_image(
        person_names,
        locations_shown=(
            list(detections_payload.get("locations_shown") or []) if isinstance(detections_payload, dict) else []
        ),
        location_payload=loc,
    )
    write_xmp_sidecar(
        sidecar_path,
        person_names=resolved_person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=description,
        album_title=album_title,
        gps_latitude=str(loc.get("gps_latitude") or ""),
        gps_longitude=str(loc.get("gps_longitude") or ""),
        location_address=str(loc.get("address") or ""),
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
        replace_dc_date=replace_dc_date,
        history_when=_xmp_timestamp_from_path(image_path),
        image_width=img_w,
        image_height=img_h,
        ocr_ran=ocr_ran,
        people_detected=bool(people_detected or resolved_person_names),
        people_identified=bool(resolved_person_names),
        locations_shown=detections_payload.get("locations_shown") if detections_payload else None,
    )
    _append_xmp_job_artifact(image_path, sidecar_path)


_CAPTION_PROMPT_OVERRIDE_FLAGS = frozenset(
    {
        "--caption-prompt",
        "--local-prompt",
        "--qwen-prompt",
        "--caption-prompt-file",
        "--local-prompt-file",
        "--qwen-prompt-file",
    }
)

_SIMPLE_CLI_OVERRIDES: tuple[tuple[str, str, str, Any], ...] = (
    ("--ocr-engine", "ocr_engine", "ocr_engine", str),
    ("--ocr-model", "ocr_model", "ocr_model", str),
    ("--caption-engine", "caption_engine", "caption_engine", str),
    ("--caption-model", "caption_model", "caption_model", str),
    ("--caption-max-tokens", "caption_max_tokens", "caption_max_tokens", int),
    ("--caption-temperature", "caption_temperature", "caption_temperature", float),
    ("--caption-max-edge", "caption_max_edge", "caption_max_edge", int),
)

_OVERRIDE_SOURCE_FLAGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("caption_prompt", _CAPTION_PROMPT_OVERRIDE_FLAGS),
    ("caption_max_tokens", frozenset({"--caption-max-tokens"})),
    ("caption_temperature", frozenset({"--caption-temperature"})),
    ("caption_max_edge", frozenset({"--caption-max-edge"})),
)


def _caption_engine_lower(effective: dict[str, Any], defaults: dict[str, Any]) -> str:
    return str(effective.get("caption_engine", defaults["caption_engine"])).strip().lower()


def _is_gps_repair_requested(state: _ProcessOneState) -> bool:
    return (
        state.existing_sidecar_current
        and state.existing_sidecar_complete
        and state.existing_sidecar_state is not None
        and (state.location_shown_missing or state.location_shown_gps_dirty)
        and not state.reprocess_required
        and not state.source_refresh_required
        and not state.date_refresh_required
    )


@dataclass
class _FullEngines:
    caption_engine: CaptionEngine
    caption_key: tuple[str, str, str, int, float, str, int, bool]
    ocr_engine: OCREngine
    ocr_key: tuple[str, str, str, str]
    object_detector: Any


@dataclass
class _FullHints:
    album_title_hint: str
    upstream_context_ocr: str
    upstream_location_hint: str


@dataclass
class _FullAnalysisTargets:
    analysis_target: Path
    people_analysis_source: Path


@dataclass
class _FullAnalysisOutcome:
    analysis: Any
    payload: dict[str, Any]
    person_names: list[str]
    subjects: list[str]
    description: str
    ocr_text: str
    resolved_album_title: str
    analysis_mode: str
    ocr_authority_hash: str
    scan_filenames: list[str]
    step_runner: StepRunner
    existing_detections: dict[str, Any]


def _resolve_full_hints(image_path: Path, printed_album_title_cache: dict[str, str]) -> _FullHints:
    album_title_hint = _resolve_album_title_hint(image_path)
    printed_hint = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
    upstream_page_state = _resolve_upstream_page_sidecar_state(image_path)
    upstream_context_ocr = str((upstream_page_state or {}).get("ocr_text") or "").strip()
    upstream_location_hint = _format_location_hint_from_state(upstream_page_state)
    upstream_album = str((upstream_page_state or {}).get("album_title") or "").strip()
    if not album_title_hint:
        album_title_hint = upstream_album
    if not printed_hint:
        printed_hint = upstream_album
    return _FullHints(
        album_title_hint=album_title_hint,
        upstream_context_ocr=upstream_context_ocr,
        upstream_location_hint=upstream_location_hint,
    )


def _full_analysis_targets(
    image_path: Path,
    layout: Any,
    scan_ocr_authority: ArchiveScanOCRAuthority | None,
) -> _FullAnalysisTargets:
    stitched_path = scan_ocr_authority.stitched_image_path if scan_ocr_authority is not None else None
    layout_target = layout.content_path if layout.page_like else image_path
    analysis_target = stitched_path or layout_target
    people_analysis_source = analysis_target if scan_ocr_authority is not None else layout_target
    return _FullAnalysisTargets(analysis_target=analysis_target, people_analysis_source=people_analysis_source)


def _full_engine_model_name(engine: Any, requested: str) -> str:
    if str(requested).strip().lower() in {"local", "lmstudio"}:
        return str(engine.effective_model_name)
    return ""


def _merge_pipeline_records(
    payload: dict[str, Any], existing_detections: dict[str, Any], step_runner: StepRunner
) -> None:
    pending = step_runner.get_pending_records()
    merged = dict(existing_detections.get("pipeline") or {})
    merged.update(dict(payload.get("pipeline") or {}))
    merged.update(pending)
    if merged:
        payload["pipeline"] = merged


def _full_resolve_location_payload(
    payload: dict[str, Any],
    step_runner: StepRunner,
    image_path: Path,
    existing_sidecar_state: dict | None,
) -> dict[str, Any]:
    location_payload = dict(payload.get("location") or {}) if isinstance(payload, dict) else {}
    if step_runner.reran.get("metadata") or step_runner.reran.get("locations"):
        return location_payload
    effective_location_payload = _effective_sidecar_location_payload(image_path, existing_sidecar_state)
    return effective_location_payload or location_payload


def _full_final_dc_date(analysis: Any, existing_sidecar_state: dict | None) -> Any:
    metadata_dc_date = str(analysis.dc_date or "").strip()
    if metadata_dc_date and not _has_dc_date(_dc_date_value(existing_sidecar_state)):
        return metadata_dc_date
    return _dc_date_value(existing_sidecar_state)


def _full_processing_payload(
    *,
    image_path: Path,
    settings_sig: str,
    current_cast_signature: str,
    effective: dict[str, Any],
    ocr_text: str,
    final_album_title: str,
    final_dc_date: Any,
    existing_sidecar_state: dict | None,
    scan_ocr_authority: ArchiveScanOCRAuthority | None,
    ocr_authority_hash: str,
    analysis_mode: str,
    date_estimation_enabled: bool,
) -> dict[str, Any]:
    stat = image_path.stat()
    if date_estimation_enabled or final_dc_date:
        date_hash: str = _date_estimate_input_hash(ocr_text, final_album_title)
    else:
        date_hash = str((existing_sidecar_state or {}).get("date_estimate_input_hash") or "")
    return {
        "processor_signature": PROCESSOR_SIGNATURE,
        "settings_signature": settings_sig,
        "cast_store_signature": (current_cast_signature if bool(effective.get("enable_people", True)) else ""),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "date_estimate_input_hash": date_hash,
        "ocr_authority_signature": (str(scan_ocr_authority.signature) if scan_ocr_authority is not None else ""),
        "ocr_authority_hash": ocr_authority_hash,
        "analysis_mode": str(analysis_mode),
    }


@dataclass
class _PeopleUpdateInputs:
    detections: dict[str, Any]
    existing_people_rows: list[dict[str, Any]]
    existing_caption_payload: dict[str, Any]
    existing_ocr_text: str
    existing_ocr_keywords: list[str]
    existing_object_labels: list[str]
    existing_location: dict[str, Any]


def _pu_inputs_from_state(image_path: Path, state: dict[str, Any]) -> _PeopleUpdateInputs:
    det = state.get("detections") or {}
    existing_people_rows = [r for r in list(det.get("people") or []) if isinstance(r, dict)]
    existing_caption_payload = dict(det.get("caption") or {})
    existing_ocr_text = _effective_sidecar_ocr_text(image_path, state)
    existing_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
    existing_object_rows = [r for r in list(det.get("objects") or []) if isinstance(r, dict)]
    existing_object_labels = [str(r.get("label") or "") for r in existing_object_rows if r.get("label")]
    existing_location = _effective_sidecar_location_payload(image_path, state)
    return _PeopleUpdateInputs(
        detections=det,
        existing_people_rows=existing_people_rows,
        existing_caption_payload=existing_caption_payload,
        existing_ocr_text=existing_ocr_text,
        existing_ocr_keywords=existing_ocr_keywords,
        existing_object_labels=existing_object_labels,
        existing_location=existing_location,
    )


def _people_matcher_faces(people_matcher: Any) -> int:
    if not people_matcher:
        return 0
    last = getattr(people_matcher, "last_faces_detected", 0)
    return last if isinstance(last, int) else 0


def _pu_match_people(
    people_matcher: Any,
    image_path: Path,
    ocr_text: str,
    *,
    person_hint_count: int = 0,
) -> tuple[list[Any], int]:
    if not people_matcher:
        return [], 0
    matches = _match_people_with_cast_store_retry(
        people_matcher=people_matcher,
        image_path=image_path,
        source_path=image_path,
        bbox_offset=(0, 0),
        hint_text=ocr_text,
        person_hint_count=person_hint_count,
    )
    return matches, _people_matcher_faces(people_matcher)


def _pu_finalize_detections(
    pu_updated_det: dict[str, Any],
    *,
    existing_location: dict[str, Any],
    cast_store_signature: str,
    ocr_text: str,
    album_title: str,
    stamp_date_hash: bool,
) -> dict[str, Any]:
    pu_proc = dict(pu_updated_det.get("processing") or {})
    pu_proc["cast_store_signature"] = cast_store_signature
    if stamp_date_hash:
        pu_proc["date_estimate_input_hash"] = _date_estimate_input_hash(ocr_text, album_title)
    if existing_location:
        pu_updated_det["location"] = existing_location
    return {**pu_updated_det, "processing": pu_proc}


def _refresh_gps_coords(refresh_location: dict[str, Any], review: dict[str, Any]) -> tuple[str, str]:
    refresh_gps_lat = str(refresh_location.get("gps_latitude") or "").strip()
    refresh_gps_lon = str(refresh_location.get("gps_longitude") or "").strip()
    if not refresh_gps_lat:
        refresh_gps_lat = _xmp_gps_to_decimal(review.get("gps_latitude"), axis="lat")
    if not refresh_gps_lon:
        refresh_gps_lon = _xmp_gps_to_decimal(review.get("gps_longitude"), axis="lon")
    return refresh_gps_lat, refresh_gps_lon


def _refresh_page_like(review: dict[str, Any], refresh_detections: dict[str, Any]) -> bool:
    return bool(review.get("subphotos")) or (
        str((refresh_detections.get("caption") or {}).get("effective_engine") or "").strip() == "page-summary"
    )


def _refresh_analysis_mode(existing_sidecar_state: dict | None, review: dict[str, Any]) -> str:
    refresh_subphotos = review.get("subphotos")
    return str(
        (existing_sidecar_state or {}).get("analysis_mode")
        or ("page_subphotos" if isinstance(refresh_subphotos, list) and refresh_subphotos else "single_image")
    )


def _refresh_processing_payload(
    *,
    image_path: Path,
    review: dict[str, Any],
    existing_sidecar_state: dict,
    settings_sig: str,
    current_cast_signature: str,
    effective: dict[str, Any],
    refresh_ocr_text: str,
    refresh_album_title: str,
) -> dict[str, Any]:
    stat = image_path.stat()
    return {
        "processor_signature": PROCESSOR_SIGNATURE,
        "settings_signature": settings_sig,
        "cast_store_signature": (current_cast_signature if bool(effective.get("enable_people", True)) else ""),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "date_estimate_input_hash": _date_estimate_input_hash(refresh_ocr_text, refresh_album_title),
        "ocr_authority_signature": str(existing_sidecar_state.get("ocr_authority_signature") or ""),
        "ocr_authority_hash": str(existing_sidecar_state.get("ocr_authority_hash") or ""),
        "analysis_mode": _refresh_analysis_mode(existing_sidecar_state, review),
    }


def _refresh_text_layers(
    image_path: Path,
    review: dict[str, Any],
    refresh_ocr_text: str,
    refresh_detections: dict[str, Any],
) -> dict[str, str]:
    return _resolve_xmp_text_layers(
        image_path=image_path,
        ocr_text=refresh_ocr_text,
        page_like=_refresh_page_like(review, refresh_detections),
        ocr_authority_source=str(review.get("ocr_authority_source") or ""),
        author_text=str(review.get("author_text") or ""),
        scene_text=str(review.get("scene_text") or ""),
    )


def _refresh_xmp_title(image_path: Path, review: dict[str, Any], text_layers: dict[str, str]) -> tuple[str, str]:
    return _compute_xmp_title(
        image_path=image_path,
        explicit_title=str(review.get("title") or ""),
        title_source=str(review.get("title_source") or ""),
        author_text=str(text_layers.get("author_text") or ""),
    )


def _refresh_album_title(image_path: Path, review: dict[str, Any], refresh_ocr_text: str) -> str:
    base_title = str(review.get("album_title") or "").strip() or _resolve_album_title_hint(image_path)
    return _require_album_title_for_title_page(
        image_path=image_path,
        album_title=_resolve_title_page_album_title(
            image_path=image_path,
            album_title=base_title,
            ocr_text=refresh_ocr_text,
        ),
        context="refresh",
    )


def _refresh_writer_kwargs(
    *,
    review: dict[str, Any],
    text_layers: dict[str, str],
    refresh_album_title: str,
    refresh_write_location: dict[str, Any],
    refresh_ocr_text: str,
    refresh_detections: dict[str, Any],
    refresh_dc_date: str,
    refresh_date_time_original: str,
    xmp_title: str,
    xmp_title_source: str,
    image_path: Path,
) -> dict[str, Any]:
    return dict(
        person_names=list(review.get("person_names") or []),
        subjects=list(review.get("subjects") or []),
        title=xmp_title,
        title_source=xmp_title_source,
        description=str(review.get("description") or ""),
        album_title=refresh_album_title,
        location_payload=refresh_write_location,
        source_text=_build_dc_source(refresh_album_title, image_path, _page_scan_filenames(image_path)),
        ocr_text=refresh_ocr_text,
        ocr_lang=str(review.get("ocr_lang") or ""),
        author_text=str(text_layers.get("author_text") or ""),
        scene_text=str(text_layers.get("scene_text") or ""),
        detections_payload=refresh_detections,
        stitch_key=str(review.get("stitch_key") or ""),
        ocr_authority_source=str(review.get("ocr_authority_source") or ""),
        create_date=(str(review.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
        dc_date=refresh_dc_date,
        date_time_original=refresh_date_time_original,
        ocr_ran=bool(review.get("ocr_ran")),
        people_detected=bool(review.get("people_detected")),
        people_identified=bool(review.get("people_identified")),
    )


def _refresh_write_location(
    refresh_detections: dict[str, Any],
    refresh_location: dict[str, Any],
    refresh_gps_lat: str,
    refresh_gps_lon: str,
) -> dict[str, Any]:
    refresh_write_location = dict((refresh_detections or {}).get("location") or {})
    if not refresh_write_location:
        refresh_write_location = dict(refresh_location or {})
    if refresh_gps_lat and not refresh_write_location.get("gps_latitude"):
        refresh_write_location["gps_latitude"] = refresh_gps_lat
    if refresh_gps_lon and not refresh_write_location.get("gps_longitude"):
        refresh_write_location["gps_longitude"] = refresh_gps_lon
    return refresh_write_location


def _sidecar_matches_stitched_authority(state: _ProcessOneState, existing_ocr_hash: str) -> bool:
    sidecar_state = state.existing_sidecar_state or {}
    sidecar_source = str(sidecar_state.get("ocr_authority_source") or "").strip()
    sidecar_signature = str(sidecar_state.get("ocr_authority_signature") or "").strip()
    sidecar_hash = str(sidecar_state.get("ocr_authority_hash") or "").strip()
    has_current_authority = (
        sidecar_source == "archive_stitched"
        and bool(existing_ocr_hash)
        and _sidecar_current_for_paths(state.sidecar_path, state.multi_scan_group_paths)
    )
    matches_authority = (
        sidecar_source == "archive_stitched"
        and sidecar_signature == state.multi_scan_group_signature
        and bool(sidecar_hash)
        and sidecar_hash == existing_ocr_hash
    )
    return matches_authority or has_current_authority


@dataclass
class _ProcessOneState:
    image_path: Path
    sidecar_path: Path
    effective: dict[str, Any] = field(default_factory=dict)
    settings_sig: str = ""
    date_estimation_enabled: bool = False
    existing_xmp_people: list[str] = field(default_factory=list)

    existing_sidecar_valid: bool = False
    existing_sidecar_current: bool = False
    existing_sidecar_state: dict | None = None
    existing_sidecar_complete: bool = False
    source_refresh_required: bool = False
    date_refresh_required: bool = False
    reprocess_required: bool = False
    reprocess_reasons: list[str] = field(default_factory=list)

    location_shown_missing: bool = False
    location_shown_backfill_needed: bool = False
    location_shown_gps_dirty: bool = False

    people_update_only: bool = False
    people_matcher: Any = None
    current_cast_signature: str = ""

    gps_repair_requested: bool = False

    archive_stitched_ocr_required: bool = False
    multi_scan_group_paths: list[Path] = field(default_factory=list)
    multi_scan_group_signature: str = ""

    needs_full: bool = False
    gps_update_only: bool = False
    extra_forced: set[str] = field(default_factory=set)


class IndexRunner:
    def __init__(self, argv: list[str] | None = None) -> None:
        self.args = parse_args(argv)
        self.explicit_flags = _explicit_cli_flags(argv)
        self.requested_caption_prompt = _resolve_caption_prompt(
            str(getattr(self.args, "caption_prompt", "")),
            str(getattr(self.args, "caption_prompt_file", "")),
        )
        self.photos_root = _absolute_cli_path(self.args.photos_root)
        self.archive_set = find_archive_set_by_photos_root(self.photos_root)
        self.title_page_location = self.archive_set.title_page_location if self.archive_set is not None else None
        self.stdout_only = bool(self.args.stdout)
        self.reprocess_mode = str(self.args.reprocess_mode)
        self.force_processing = bool(self.args.force or self.stdout_only or self.reprocess_mode == "all")
        self.dry_run = bool(self.args.dry_run or self.stdout_only)
        self.shard_count = int(self.args.shard_count or 1)
        self.shard_index = int(self.args.shard_index or 0)
        self._validate_init_args()

        self.include_archive, self.include_view = self._resolve_inclusion_flags()
        self.ext_set = self._resolve_extension_set()
        self.single_photo = str(self.args.photo or "").strip()
        self.defaults = self._build_defaults()
        self._init_caches()

        self.processed = 0
        self.skipped = 0
        self.failures = 0
        self.completed_times: list[float] = []

        self.files: list[Path] = []
        self.batch_lock_path: Path | None = None
        self.allow_concurrent_shards = False

    def _validate_init_args(self) -> None:
        if not self.photos_root.is_dir():
            raise SystemExit(f"Photo root is not a directory: {self.photos_root}")
        if self.shard_count < 1:
            raise SystemExit("--shard-count must be at least 1")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise SystemExit("--shard-index must be between 0 and --shard-count - 1")

    def _resolve_inclusion_flags(self) -> tuple[bool, bool]:
        include_archive = bool(self.args.include_archive)
        include_view = bool(self.args.include_view)
        if not include_archive and not include_view:
            return True, False
        return include_archive, include_view

    def _resolve_extension_set(self) -> set[str]:
        ext_set = {
            (item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}")
            for item in str(self.args.extensions or "").split(",")
            if item.strip()
        }
        return ext_set or set(IMAGE_EXTENSIONS)

    def _build_defaults(self) -> dict[str, Any]:
        caption_params = load_params("ai-index/metadata/params.toml").values
        is_lmstudio = str(self.args.caption_engine) == "lmstudio"
        max_tokens = int(self.args.caption_max_tokens)
        if "--caption-max-tokens" not in self.explicit_flags and is_lmstudio:
            max_tokens = int(caption_params.get("max_tokens", DEFAULT_LMSTUDIO_MAX_NEW_TOKENS))
        temperature = float(self.args.caption_temperature)
        if "--caption-temperature" not in self.explicit_flags and is_lmstudio:
            temperature = float(caption_params.get("temperature", self.args.caption_temperature))
        max_edge = int(self.args.caption_max_edge)
        if "--caption-max-edge" not in self.explicit_flags and is_lmstudio:
            max_edge = int(caption_params.get("max_image_edge", self.args.caption_max_edge))
        return {
            "skip": False,
            "enable_people": not bool(self.args.disable_people),
            "enable_objects": not bool(self.args.disable_objects),
            "ocr_engine": str(self.args.ocr_engine),
            "ocr_lang": str(self.args.ocr_lang),
            "ocr_model": str(self.args.ocr_model),
            "caption_engine": str(self.args.caption_engine),
            "caption_model": resolve_caption_model(str(self.args.caption_engine), str(self.args.caption_model)),
            "caption_prompt": str(self.requested_caption_prompt),
            "caption_max_tokens": max_tokens,
            "caption_temperature": temperature,
            "caption_max_edge": max_edge,
            "caption_thinking": bool(caption_params.get("thinking", False)),
            "lmstudio_base_url": normalize_lmstudio_base_url(str(self.args.lmstudio_base_url)),
            "people_threshold": float(self.args.people_threshold),
            "object_threshold": float(self.args.object_threshold),
            "min_face_size": int(self.args.min_face_size),
            "model": str(self.args.model),
        }

    def _init_caches(self) -> None:
        self.archive_settings_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.people_matcher_cache: dict[tuple[str, float, int], Any] = {}
        self.object_detector_cache: dict[tuple[str, float], Any] = {}
        self.ocr_engine_cache: dict[tuple[str, str, str, str], OCREngine] = {}
        self.caption_engine_cache: dict[tuple[str, str, str, int, float, str, int, bool], CaptionEngine] = {}
        self.date_engine_cache: dict[tuple[str, str, int, float, str], DateEstimateEngine] = {}
        self.metadata_engine_cache: dict[tuple[str, str, str], MetadataEngine] = {}
        self.archive_scan_ocr_cache: dict[str, ArchiveScanOCRAuthority] = {}
        self.printed_album_title_cache: dict[str, str] = {}
        self.geocoder = NominatimGeocoder()
        self.stitch_cap_td = tempfile.TemporaryDirectory(prefix="imago-stitch-cap-")
        self.stitch_cap_dir = Path(self.stitch_cap_td.name)

    def emit_info(self, message: str) -> None:
        if not self.stdout_only:
            print(message)

    def emit_error(self, message: str) -> None:
        print(message, file=sys.stderr if self.stdout_only else sys.stdout, flush=True)

    def _caption_key_from_effective(
        self, effective: dict[str, Any]
    ) -> tuple[str, str, str, int, float, str, int, bool]:
        return (
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
            str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
            int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
            float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
            str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
            int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
            bool(effective.get("caption_thinking", self.defaults["caption_thinking"])),
        )

    def _get_caption_engine_for_key(
        self,
        caption_key: tuple[str, str, str, int, float, str, int, bool],
        effective: dict[str, Any],
        *,
        stream: bool = True,
    ) -> CaptionEngine:
        caption_engine = self.caption_engine_cache.get(caption_key)
        if caption_engine is None:
            caption_engine = _init_caption_engine(
                engine=caption_key[0],
                model_name=caption_key[1],
                caption_prompt=caption_key[2],
                max_tokens=int(caption_key[3]),
                temperature=float(caption_key[4]),
                lmstudio_base_url=caption_key[5],
                max_image_edge=int(caption_key[6]),
                stream=stream,
                thinking=bool(caption_key[7]),
                override_sources=dict(effective.get("_override_sources") or {}),
            )
            self.caption_engine_cache[caption_key] = caption_engine
        caption_engine.override_sources = dict(effective.get("_override_sources") or {})
        return caption_engine

    def _get_date_engine(self, effective: dict[str, Any]) -> DateEstimateEngine:
        date_key = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
            int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
            0.0,
            str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
        )
        date_engine = self.date_engine_cache.get(date_key)
        if date_engine is None:
            date_engine = _init_date_engine(
                engine=date_key[0],
                model_name=date_key[1],
                max_tokens=int(date_key[2]),
                temperature=0.0,
                lmstudio_base_url=date_key[4],
            )
            self.date_engine_cache[date_key] = date_engine
        return date_engine

    def _get_metadata_engine(self, effective: dict[str, Any]) -> MetadataEngine:
        engine = str(effective.get("caption_engine", self.defaults["caption_engine"]))
        model = str(effective.get("caption_model", self.defaults["caption_model"]))
        base_url = str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"]))
        metadata_key = (engine, model, base_url)
        cached = self.metadata_engine_cache.get(metadata_key)
        if cached is None:
            cached = MetadataEngine(
                engine=engine,
                model_name=model,
                lmstudio_base_url=base_url,
            )
            self.metadata_engine_cache[metadata_key] = cached
        return cached

    def _record_failure(self, idx: int, image_path: Path, exc: Exception) -> None:
        self.failures += 1
        self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def run(self) -> int:
        setup_result = self._setup()
        if setup_result is not None:
            return setup_result
        for idx, image_path in enumerate(self.files, 1):
            self._process_one(idx, image_path)
        return self._summarize()

    def _discover_multi_photo_files(self) -> None:
        self.files = discover_images(
            self.photos_root,
            include_archive=self.include_archive,
            include_view=self.include_view,
            extensions=self.ext_set,
        )
        album_filter = str(self.args.album or "").strip()
        if album_filter:
            album_lower = album_filter.casefold()
            self.files = [f for f in self.files if album_lower in f.parent.name.casefold()]
        self.files = _coalesce_archive_processing_files(self.files)
        photo_offset = int(self.args.photo_offset or 0)
        if photo_offset > 0:
            self.files = self.files[photo_offset:]
        if self.args.max_images and self.args.max_images > 0:
            self.files = self.files[: int(self.args.max_images)]
        self.files = _apply_shard(self.files, self.shard_count, self.shard_index)

    def _setup(self) -> int | None:
        if self.single_photo:
            if self.shard_count > 1:
                raise SystemExit("--shard-count > 1 is only supported for multi-photo discovery runs")
            photo_path = _absolute_cli_path(self.single_photo)
            if not photo_path.is_file():
                raise SystemExit(f"Photo not found: {photo_path}")
            self.files = [photo_path]
            self.force_processing = True
        else:
            self._discover_multi_photo_files()

        original_file_count = len(self.files)
        self.files = _expand_album_title_dependencies(self.files, self.ext_set)
        if not self.single_photo:
            self.files = _filter_files_by_tree(
                self.files,
                include_archive=self.include_archive,
                include_view=self.include_view,
            )

        self.emit_info(f"Discovered {len(self.files)} image files")
        if len(self.files) > original_file_count:
            self.emit_info(f"Added {len(self.files) - original_file_count} title-page dependency files")
        if not self.files:
            return 0

        self.allow_concurrent_shards = not self.single_photo and self.shard_count > 1
        if not self.single_photo and not self.allow_concurrent_shards:
            try:
                self.batch_lock_path = _acquire_batch_processing_lock(self.photos_root)
            except RuntimeError as exc:
                self.emit_error(str(exc))
                return 1

        return None

    def _summarize(self) -> int:
        stitch_failures = 0
        if bool(getattr(self.args, "stitch_scans", False)):
            self.emit_info("Scan stitch pass skipped: archive scan OCR stitching now happens during normal processing.")

        if not self.stdout_only:
            print("\nSummary")
            print(f"- Processed: {self.processed}")
            print(f"- Skipped:   {self.skipped}")
            print(f"- Failed:    {self.failures + stitch_failures}")
        _release_batch_processing_lock(self.batch_lock_path)
        self.stitch_cap_td.cleanup()
        return 1 if (self.failures or stitch_failures) else 0

    def _resolve_effective_settings(self, image_path: Path) -> tuple[dict[str, Any], str, bool]:
        settings_file, loaded_settings = self._load_archive_settings(image_path)
        effective = resolve_effective_settings(
            image_path,
            defaults=self.defaults,
            loaded=loaded_settings,
        )
        self._apply_cli_overrides(effective)
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
        )
        effective["_override_sources"] = self._build_override_sources(effective, loaded_settings, settings_file)
        settings_sig = _settings_signature(effective)
        date_estimation_enabled = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])).strip().lower() == "lmstudio"
        )
        return effective, settings_sig, date_estimation_enabled

    def _load_archive_settings(self, image_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
        archive_dir = find_archive_dir_for_image(image_path)
        if archive_dir is None or self.args.ignore_render_settings:
            return None, None
        key = str(archive_dir.resolve())
        cached = self.archive_settings_cache.get(key)
        if cached is None:
            path, payload = load_render_settings(
                archive_dir,
                defaults=self.defaults,
                create=False,
            )
            cached = (path, payload)
            self.archive_settings_cache[key] = cached
        return cached

    def _apply_cli_overrides(self, effective: dict[str, Any]) -> None:
        if self.args.disable_people:
            effective["enable_people"] = False
        if self.args.disable_objects:
            effective["enable_objects"] = False
        for flag, key, attr, converter in _SIMPLE_CLI_OVERRIDES:
            if flag in self.explicit_flags:
                effective[key] = converter(getattr(self.args, attr))
        if _CAPTION_PROMPT_OVERRIDE_FLAGS & self.explicit_flags:
            effective["caption_prompt"] = str(self.requested_caption_prompt)
        if "--lmstudio-base-url" in self.explicit_flags:
            effective["lmstudio_base_url"] = normalize_lmstudio_base_url(str(self.args.lmstudio_base_url))

    def _build_override_sources(
        self,
        effective: dict[str, Any],
        loaded_settings: dict[str, Any] | None,
        settings_file: Path | None,
    ) -> dict[str, str]:
        override_sources: dict[str, str] = {}
        for key, flags in _OVERRIDE_SOURCE_FLAGS:
            if flags & self.explicit_flags:
                override_sources[key] = "cli"
            elif loaded_settings is not None and effective.get(key) != self.defaults.get(key):
                override_sources[key] = f"render_settings:{settings_file}"
        return override_sources

    def _get_people_matcher_and_signature(self, effective: dict[str, Any]) -> tuple[Any, str]:
        if not bool(effective.get("enable_people", True)):
            return None, ""
        people_key = (
            str(Path(self.args.cast_store).resolve()),
            float(effective.get("people_threshold", self.defaults["people_threshold"])),
            int(effective.get("min_face_size", self.defaults["min_face_size"])),
        )
        people_matcher = self.people_matcher_cache.get(people_key)
        if people_matcher is None:
            people_matcher = _init_people_matcher(
                cast_store=Path(self.args.cast_store),
                min_similarity=float(people_key[1]),
                min_face_size=int(people_key[2]),
            )
            self.people_matcher_cache[people_key] = people_matcher
        return people_matcher, self._people_invalidation_signature(people_matcher)

    def _people_invalidation_signature(self, people_matcher: Any) -> str:
        reviewed_signature_fn = getattr(people_matcher, "reviewed_identity_signature", None)
        if callable(reviewed_signature_fn):
            reviewed_signature = reviewed_signature_fn()
            if isinstance(reviewed_signature, str) and reviewed_signature.strip():
                return reviewed_signature
        return str(people_matcher.store_signature())

    # Per-image dispatch

    def _process_one(self, idx: int, image_path: Path) -> None:
        sidecar_path = image_path.with_suffix(".xmp")
        state = _ProcessOneState(
            image_path=image_path,
            sidecar_path=sidecar_path,
            existing_xmp_people=read_person_in_image(sidecar_path),
        )
        state.effective, state.settings_sig, state.date_estimation_enabled = self._resolve_effective_settings(
            image_path
        )

        self._evaluate_existing_sidecar(state)
        self._evaluate_locations_shown(state)
        self._evaluate_people_update(state)
        state.gps_repair_requested = _is_gps_repair_requested(state)

        if self._can_skip_current(state):
            self._emit_skip(idx, image_path, "current xmp")
            return

        if state.people_matcher is None:
            state.people_matcher, state.current_cast_signature = self._get_people_matcher_and_signature(state.effective)

        self._evaluate_multi_scan(state)
        self._evaluate_extra_reprocess_reasons(state)
        self._decide_processing_mode(state)

        if self._should_skip_after_decision(state):
            self._emit_skip(idx, image_path, "")
            return
        if bool(state.effective.get("skip", False)):
            self._emit_skip(idx, image_path, "render_settings skip=true")
            return
        if not self._matches_reprocess_mode(state):
            self._emit_skip(idx, image_path, f"reprocess_mode={self.reprocess_mode}")
            return

        self._emit_reprocess_status(idx, state)
        self._dispatch_with_lock(idx, state)

    def _emit_skip(self, idx: int, image_path: Path, reason: str) -> None:
        self.skipped += 1
        if self.args.verbose and not self.stdout_only:
            suffix = f" ({reason})" if reason else ""
            print(f"[{idx}/{len(self.files)}] skip  {image_path.name}{suffix}")

    def _evaluate_existing_sidecar(self, state: _ProcessOneState) -> None:
        image_path = state.image_path
        sidecar_path = state.sidecar_path
        effective = state.effective
        state.existing_sidecar_valid = has_valid_sidecar(image_path)
        state.existing_sidecar_current = has_current_sidecar(image_path) if state.existing_sidecar_valid else False
        if state.existing_sidecar_valid:
            state.existing_sidecar_state = read_ai_sidecar_state(sidecar_path)
        if state.existing_sidecar_valid and not state.existing_sidecar_current:
            state.reprocess_reasons.append("sidecar_older_than_image")
        if _sidecar_has_lmstudio_caption_error(state.existing_sidecar_state):
            state.reprocess_required = True
            state.reprocess_reasons.append("lmstudio_caption_error")
        if state.existing_sidecar_valid:
            state.existing_sidecar_complete = sidecar_has_expected_ai_fields(
                sidecar_path,
                enable_people=bool(effective.get("enable_people", True)),
                enable_objects=bool(effective.get("enable_objects", True)),
                ocr_engine=str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
                caption_engine=str(effective.get("caption_engine", self.defaults["caption_engine"])),
            )
            state.source_refresh_required = _dc_source_needs_refresh(image_path, state.existing_sidecar_state)
            if state.source_refresh_required:
                state.reprocess_reasons.append("dc_source_stale")
            state.date_refresh_required = _dc_date_needs_refresh(
                image_path,
                state.existing_sidecar_state,
                enabled=state.date_estimation_enabled,
            )
            if state.date_refresh_required:
                state.reprocess_reasons.append("timeline_date_missing")

    def _evaluate_locations_shown(self, state: _ProcessOneState) -> None:
        if not (
            state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
            and _caption_engine_lower(state.effective, self.defaults) == "lmstudio"
        ):
            return
        det = state.existing_sidecar_state.get("detections") or {}
        detected_locations = list(det.get("locations_shown") or []) if isinstance(det, dict) else []
        written_locations = read_locations_shown(state.sidecar_path)
        location_shown_ran = isinstance(det, dict) and det.get("location_shown_ran") is True
        state.location_shown_missing = bool(written_locations) is False and (
            location_shown_ran or bool(detected_locations)
        )
        state.location_shown_backfill_needed = (
            not location_shown_ran
            and not detected_locations
            and not written_locations
            and isinstance(det, dict)
            and bool(det.get("location"))
        )
        state.location_shown_gps_dirty = _has_legacy_ai_locations_shown_gps(state.existing_sidecar_state)

    def _evaluate_people_update(self, state: _ProcessOneState) -> None:
        if state.existing_sidecar_state is None or not bool(state.effective.get("enable_people", True)):
            return
        old_cast_signature = str(state.existing_sidecar_state.get("cast_store_signature") or "")
        if not (old_cast_signature and _sidecar_has_people_to_refresh(state.existing_sidecar_state)):
            return
        state.people_matcher, state.current_cast_signature = self._get_people_matcher_and_signature(state.effective)
        if old_cast_signature != state.current_cast_signature:
            state.people_update_only = True
            state.reprocess_reasons.append("cast_store_signature_changed")

    def _can_skip_current(self, state: _ProcessOneState) -> bool:
        return (
            state.existing_sidecar_current
            and state.existing_sidecar_complete
            and not state.reprocess_required
            and not state.source_refresh_required
            and not state.date_refresh_required
            and not self.force_processing
            and self.reprocess_mode != "gps"
            and not state.people_update_only
            and not state.gps_repair_requested
        )

    def _evaluate_multi_scan(self, state: _ProcessOneState) -> None:
        state.multi_scan_group_paths = _scan_group_paths(state.image_path)
        state.archive_stitched_ocr_required = (
            str(state.effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower() != "none"
            and len(state.multi_scan_group_paths) > 1
        )
        state.multi_scan_group_signature = (
            _scan_group_signature(state.multi_scan_group_paths) if state.archive_stitched_ocr_required else ""
        )

    @staticmethod
    def _check_sidecar_incomplete(state: _ProcessOneState) -> None:
        if state.existing_sidecar_valid and not state.existing_sidecar_complete:
            state.reprocess_required = True
            state.reprocess_reasons.append("sidecar_incomplete")

    @staticmethod
    def _check_album_title_missing(state: _ProcessOneState) -> None:
        existing_album_title = str((state.existing_sidecar_state or {}).get("album_title") or "").strip()
        if not existing_album_title and (
            _is_album_title_source_candidate(state.image_path) or _resolve_album_title_from_sidecars(state.image_path)
        ):
            state.reprocess_required = True
            state.reprocess_reasons.append("missing_album_title")

    @staticmethod
    def _check_stitched_authority_mismatch(state: _ProcessOneState) -> None:
        ocr_text = str((state.existing_sidecar_state or {}).get("ocr_text") or "")
        if state.archive_stitched_ocr_required and not _sidecar_matches_stitched_authority(state, _hash_text(ocr_text)):
            state.reprocess_required = True
            state.reprocess_reasons.append("missing_stitched_authority")

    @staticmethod
    def _check_settings_sig_mismatch(state: _ProcessOneState) -> None:
        if state.existing_sidecar_state is None:
            return
        old_sig = str(state.existing_sidecar_state.get("settings_signature") or "")
        if old_sig != state.settings_sig and not (state.existing_sidecar_current and state.existing_sidecar_complete):
            state.reprocess_required = True
            state.reprocess_reasons.append("settings_signature_mismatch")

    def _evaluate_extra_reprocess_reasons(self, state: _ProcessOneState) -> None:
        self._check_sidecar_incomplete(state)
        self._check_album_title_missing(state)
        self._check_stitched_authority_mismatch(state)
        self._check_settings_sig_mismatch(state)

    def _decide_processing_mode(self, state: _ProcessOneState) -> None:
        state.needs_full = needs_processing(
            state.image_path,
            state.existing_sidecar_state,
            self.force_processing and not state.gps_repair_requested,
            reprocess_required=state.reprocess_required,
        )
        if state.gps_repair_requested:
            state.gps_update_only = True
            self._record_location_shown_reasons(state)
        if not state.gps_update_only and self._gps_mode_eligible(state):
            state.gps_update_only = True
        if not state.gps_update_only and self._lmstudio_location_repair_eligible(state):
            if state.location_shown_missing:
                state.gps_update_only = True
            if state.location_shown_gps_dirty:
                state.gps_update_only = True
            self._record_location_shown_reasons(state)
        if not state.gps_update_only and state.people_update_only and state.location_shown_backfill_needed:
            state.gps_update_only = True
            state.reprocess_reasons.append("missing_location_shown")

    def _gps_mode_eligible(self, state: _ProcessOneState) -> bool:
        return (
            self.reprocess_mode == "gps"
            and not state.needs_full
            and state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
        )

    def _lmstudio_location_repair_eligible(self, state: _ProcessOneState) -> bool:
        return (
            not state.needs_full
            and not state.source_refresh_required
            and not state.date_refresh_required
            and state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
            and _caption_engine_lower(state.effective, self.defaults) == "lmstudio"
        )

    @staticmethod
    def _record_location_shown_reasons(state: _ProcessOneState) -> None:
        if state.location_shown_missing:
            state.reprocess_reasons.append("missing_location_shown")
        if state.location_shown_gps_dirty:
            state.reprocess_reasons.append("location_shown_ai_gps_stale")

    def _should_skip_after_decision(self, state: _ProcessOneState) -> bool:
        return (
            not state.needs_full
            and not state.people_update_only
            and not state.gps_update_only
            and not isinstance(state.existing_sidecar_state, dict)
        )

    def _matches_reprocess_mode(self, state: _ProcessOneState) -> bool:
        mode = self.reprocess_mode
        if mode in ("unprocessed", "all"):
            return True
        reasons_set = set(state.reprocess_reasons)
        if mode == "new_only":
            return state.existing_sidecar_state is None
        if mode == "errors_only":
            return bool(reasons_set & {"lmstudio_caption_error", "sidecar_incomplete"})
        if mode == "outdated":
            return "sidecar_older_than_image" in reasons_set
        if mode == "cast_changed":
            return "cast_store_signature_changed" in reasons_set
        if mode == "gps":
            return state.gps_update_only
        return True

    def _emit_reprocess_status(self, idx: int, state: _ProcessOneState) -> None:
        if not state.existing_sidecar_valid or self.stdout_only:
            return
        reason_text = _format_reprocess_reasons(state.reprocess_reasons)
        if not reason_text:
            return
        prefix = f"  [{idx}/{len(self.files)}]  {state.image_path.name}"
        if state.needs_full:
            print(f"{prefix}  [reprocess: {reason_text}]", flush=True)
        elif state.people_update_only:
            print(f"{prefix}  [update: {reason_text}]", flush=True)
        elif state.source_refresh_required or state.date_refresh_required:
            print(f"{prefix}  [refresh: {reason_text}]", flush=True)

    def _dispatch_with_lock(self, idx: int, state: _ProcessOneState) -> None:
        try:
            lock_path = _acquire_image_processing_lock(state.image_path)
        except RuntimeError as exc:
            if self.allow_concurrent_shards and "already processing" in str(exc):
                self._emit_skip(idx, state.image_path, str(exc))
            else:
                self._record_failure(idx, state.image_path, exc)
            return
        try:
            self._dispatch_processing(idx, state)
        finally:
            _release_image_processing_lock(lock_path)

    def _dispatch_processing(self, idx: int, state: _ProcessOneState) -> None:
        if not state.needs_full and not state.people_update_only and not state.gps_update_only:
            if isinstance(state.existing_sidecar_state, dict):
                self._process_refresh(
                    idx,
                    image_path=state.image_path,
                    sidecar_path=state.sidecar_path,
                    effective=state.effective,
                    settings_sig=state.settings_sig,
                    date_estimation_enabled=state.date_estimation_enabled,
                    existing_sidecar_state=state.existing_sidecar_state,
                    current_cast_signature=state.current_cast_signature,
                )
            return

        if not state.needs_full and state.people_update_only:
            if isinstance(state.existing_sidecar_state, dict):
                state.extra_forced.add("people")
            state.needs_full = True
        if not state.needs_full and state.gps_update_only:
            if not isinstance(state.existing_sidecar_state, dict):
                return
            state.extra_forced.add("metadata")
            state.needs_full = True

        self._process_full(
            idx,
            image_path=state.image_path,
            sidecar_path=state.sidecar_path,
            effective=state.effective,
            settings_sig=state.settings_sig,
            date_estimation_enabled=state.date_estimation_enabled,
            existing_sidecar_state=state.existing_sidecar_state,
            existing_xmp_people=state.existing_xmp_people,
            people_matcher=state.people_matcher,
            current_cast_signature=state.current_cast_signature,
            archive_stitched_ocr_required=state.archive_stitched_ocr_required,
            multi_scan_group_paths=state.multi_scan_group_paths,
            multi_scan_group_signature=state.multi_scan_group_signature,
            extra_forced_steps=state.extra_forced or None,
        )

    # Refresh fast-path

    def _process_refresh(
        self,
        idx: int,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        current_cast_signature: str,
    ) -> None:
        if not isinstance(existing_sidecar_state, dict):
            return
        file_start = time.monotonic()
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
        try:
            review = load_ai_xmp_review(sidecar_path)
            review_dict = review if isinstance(review, dict) else None
            refresh_ocr_text = _effective_sidecar_ocr_text(image_path, review_dict)
            refresh_location = _effective_sidecar_location_payload(image_path, review_dict)
            refresh_detections = (
                dict(review.get("detections") or {}) if isinstance(review.get("detections"), dict) else {}
            )
            if refresh_location:
                refresh_detections["location"] = refresh_location
            refresh_location, refresh_detections = _apply_title_page_location_config(
                image_path=image_path,
                location_payload=refresh_location,
                detections_payload=refresh_detections,
                title_page_location=self.title_page_location,
            )
            if not self.dry_run:
                self._write_refresh_payload(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    review=review,
                    effective=effective,
                    settings_sig=settings_sig,
                    date_estimation_enabled=date_estimation_enabled,
                    existing_sidecar_state=existing_sidecar_state,
                    current_cast_signature=current_cast_signature,
                    refresh_ocr_text=refresh_ocr_text,
                    refresh_location=refresh_location,
                    refresh_detections=refresh_detections,
                    prompt_debug=prompt_debug,
                )
                _append_xmp_job_artifact(image_path, sidecar_path)
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            if not self.stdout_only:
                eta_str = _format_eta(self.completed_times, len(self.files) - idx)
                eta_part = f"  {eta_str}" if eta_str else ""
                print(
                    f"[{idx}/{len(self.files)}]{eta_part}  ok    {image_path.name}  [refresh]",
                    flush=True,
                )
        except Exception as exc:
            self.failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _write_refresh_payload(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        review: dict,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict,
        current_cast_signature: str,
        refresh_ocr_text: str,
        refresh_location: dict[str, Any],
        refresh_detections: dict[str, Any],
        prompt_debug: PromptDebugSession,
    ) -> None:
        refresh_gps_lat, refresh_gps_lon = _refresh_gps_coords(refresh_location, review)
        text_layers = _refresh_text_layers(image_path, review, refresh_ocr_text, refresh_detections)
        xmp_title, xmp_title_source = _refresh_xmp_title(image_path, review, text_layers)
        refresh_album_title = _refresh_album_title(image_path, review, refresh_ocr_text)
        date_engine = self._refresh_date_engine(effective, date_estimation_enabled, review)
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
        refresh_detections["processing"] = _refresh_processing_payload(
            image_path=image_path,
            review=review,
            existing_sidecar_state=existing_sidecar_state,
            settings_sig=settings_sig,
            current_cast_signature=current_cast_signature,
            effective=effective,
            refresh_ocr_text=refresh_ocr_text,
            refresh_album_title=refresh_album_title,
        )
        refresh_write_location = _refresh_write_location(
            refresh_detections, refresh_location, refresh_gps_lat, refresh_gps_lon
        )
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            **_refresh_writer_kwargs(
                review=review,
                text_layers=text_layers,
                refresh_album_title=refresh_album_title,
                refresh_write_location=refresh_write_location,
                refresh_ocr_text=refresh_ocr_text,
                refresh_detections=refresh_detections,
                refresh_dc_date=refresh_dc_date,
                refresh_date_time_original=refresh_date_time_original,
                xmp_title=xmp_title,
                xmp_title_source=xmp_title_source,
                image_path=image_path,
            ),
            title_page_location=self.title_page_location,
        )

    def _refresh_date_engine(
        self,
        effective: dict[str, Any],
        date_estimation_enabled: bool,
        existing: dict[str, Any],
    ) -> DateEstimateEngine | None:
        if date_estimation_enabled and not _has_dc_date(_dc_date_value(existing)):
            return self._get_date_engine(effective)
        return None

    # People-update fast-path (deleted, routed via _process_full + StepRunner)

    def _process_people_update(
        self,
        idx: int,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        chain_gps: bool,
        preserve_existing_xmp_people: bool = True,
        raise_on_error: bool = False,
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return

        file_start = time.monotonic()
        pu_inputs = _pu_inputs_from_state(image_path, state)
        prefix = self._format_progress_prefix(idx, image_path)
        print(prefix, flush=True)
        _pu_stop, _pu_step = _progress_ticker(prefix)

        try:
            _pu_step("people")
            pu_people_matches, pu_faces_detected = _pu_match_people(
                people_matcher,
                image_path,
                pu_inputs.existing_ocr_text,
                person_hint_count=len(existing_xmp_people),
            )
            pu_people_match_names = _dedupe([r.name for r in pu_people_matches])
            _pu_step(_format_people_step_label("people", pu_people_match_names))
            pu_person_names = (
                _dedupe(pu_people_match_names + existing_xmp_people)
                if preserve_existing_xmp_people
                else pu_people_match_names
            )
            pu_album_title = _resolve_album_title_hint(image_path) or _effective_sidecar_album_title(image_path, state)
            pu_printed_title = _resolve_album_printed_title_hint(image_path, self.printed_album_title_cache)
            pu_people_payload = _serialize_people_matches(pu_people_matches)
            people_names_changed = pu_person_names != existing_xmp_people

            if people_names_changed:
                pu_updated_det, pu_faces_detected, pu_prompt_debug = self._pu_recompute_caption(
                    image_path=image_path,
                    effective=effective,
                    people_matcher=people_matcher,
                    pu_inputs=pu_inputs,
                    pu_people_matches=pu_people_matches,
                    pu_person_names=pu_person_names,
                    pu_album_title=pu_album_title,
                    pu_printed_title=pu_printed_title,
                    pu_people_payload=pu_people_payload,
                    step_fn=_pu_step,
                )
            else:
                pu_updated_det = {
                    **pu_inputs.detections,
                    "people": pu_people_payload or pu_inputs.existing_people_rows,
                    "caption": pu_inputs.existing_caption_payload,
                }
                pu_prompt_debug = None

            pu_subjects = _dedupe(
                pu_inputs.existing_object_labels
                + pu_inputs.existing_ocr_keywords
                + ([pu_album_title] if pu_album_title else [])
            )
            pu_people_detected = pu_faces_detected > 0 or len(pu_person_names) > 0
            pu_people_identified = len(pu_person_names) > 0

            if not self.dry_run:
                self._write_pu_payload(
                    sidecar_path=sidecar_path,
                    image_path=image_path,
                    state=state,
                    effective=effective,
                    date_estimation_enabled=date_estimation_enabled,
                    people_matcher=people_matcher,
                    pu_inputs=pu_inputs,
                    pu_album_title=pu_album_title,
                    pu_person_names=pu_person_names,
                    pu_subjects=pu_subjects,
                    pu_updated_det=pu_updated_det,
                    pu_people_detected=pu_people_detected,
                    pu_people_identified=pu_people_identified,
                    pu_prompt_debug=pu_prompt_debug,
                )

            _pu_stop()
            if not chain_gps:
                self.processed += 1
                self.completed_times.append(time.monotonic() - file_start)
                self._emit_ok(idx, image_path)
        except Exception as exc:
            self.failures += 1
            _pu_stop()
            if raise_on_error:
                raise
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _pu_recompute_caption(
        self,
        *,
        image_path: Path,
        effective: dict[str, Any],
        people_matcher: Any,
        pu_inputs: _PeopleUpdateInputs,
        pu_people_matches: list[Any],
        pu_person_names: list[str],
        pu_album_title: str,
        pu_printed_title: str,
        pu_people_payload: list[Any],
        step_fn: Any,
    ) -> tuple[dict[str, Any], int, PromptDebugSession]:
        caption_key = self._caption_key_from_effective(effective)
        pu_caption_engine = self._get_caption_engine_for_key(caption_key, effective)
        pu_people_positions = _compute_people_positions(pu_people_matches, image_path)
        step_fn("caption")
        pu_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
        with _prepare_ai_model_image(image_path) as pu_model_path:
            pu_caption_out = pu_caption_engine.generate(
                image_path=pu_model_path,
                people=pu_person_names,
                objects=pu_inputs.existing_object_labels,
                ocr_text=pu_inputs.existing_ocr_text,
                source_path=image_path,
                album_title=pu_album_title,
                printed_album_title=pu_printed_title,
                people_positions=pu_people_positions,
                debug_recorder=pu_prompt_debug.record,
                debug_step="caption_refresh",
            )
            pu_faces_detected = _people_matcher_faces(people_matcher)
            pu_local_people_present, pu_local_estimated_people_count = _estimate_people_from_detections(
                people_matches=pu_people_matches,
                people_names=pu_person_names,
                object_labels=pu_inputs.existing_object_labels,
                faces_detected=pu_faces_detected,
            )
            pu_people_present, pu_estimated_people_count = _resolve_people_count_metadata(
                requested_caption_engine=str(caption_key[0]),
                caption_engine=pu_caption_engine,
                model_image_path=pu_model_path,
                people=pu_person_names,
                objects=pu_inputs.existing_object_labels,
                ocr_text=pu_inputs.existing_ocr_text,
                source_path=image_path,
                album_title=pu_album_title,
                printed_album_title=pu_printed_title,
                people_positions=pu_people_positions,
                local_people_present=pu_local_people_present,
                local_estimated_people_count=pu_local_estimated_people_count,
                prompt_debug=pu_prompt_debug,
                debug_step="people_count_refresh",
            )
        _emit_prompt_debug_artifact(pu_prompt_debug, dry_run=self.dry_run)
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
        pu_ocr_model = self._pu_resolved_ocr_model(pu_inputs.detections, effective)
        pu_caption_model = (
            str(pu_caption_engine.effective_model_name)
            if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
            else ""
        )
        pu_updated_det = _refresh_detection_model_metadata(
            {
                **pu_inputs.detections,
                "people": pu_people_payload,
                "caption": pu_caption_payload,
            },
            ocr_model=pu_ocr_model,
            caption_model=pu_caption_model,
        )
        return pu_updated_det, pu_faces_detected, pu_prompt_debug

    def _pu_resolved_ocr_model(self, det: dict[str, Any], effective: dict[str, Any]) -> str:
        existing = dict(det.get("ocr") or {}).get("model")
        if existing:
            return str(existing)
        ocr_engine = str(effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower()
        if ocr_engine in {"local", "lmstudio"}:
            return str(effective.get("ocr_model", self.defaults["ocr_model"]))
        return ""

    def _pu_resolve_dates(
        self,
        *,
        image_path: Path,
        state: dict,
        effective: dict[str, Any],
        date_estimation_enabled: bool,
        pu_album_title: str,
        pu_inputs: _PeopleUpdateInputs,
        pu_prompt_debug: PromptDebugSession | None,
    ) -> tuple[str, str, str]:
        date_engine = self._refresh_date_engine(effective, date_estimation_enabled, state)
        pu_dc_date = _resolve_dc_date(
            existing_dc_date=_dc_date_value(state),
            ocr_text=pu_inputs.existing_ocr_text,
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
        return pu_dc_date, pu_date_time_original, pu_source_text

    def _pu_resolve_text_and_title(
        self,
        *,
        image_path: Path,
        state: dict,
        pu_updated_det: dict[str, Any],
        pu_inputs: _PeopleUpdateInputs,
    ) -> tuple[dict[str, Any], str, str]:
        pu_page_like = (
            str((pu_updated_det.get("caption") or {}).get("effective_engine") or "").strip() == "page-summary"
        )
        text_layers = _resolve_xmp_text_layers(
            image_path=image_path,
            ocr_text=pu_inputs.existing_ocr_text,
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
        return text_layers, xmp_title, xmp_title_source

    def _write_pu_payload(
        self,
        *,
        sidecar_path: Path,
        image_path: Path,
        state: dict,
        effective: dict[str, Any],
        date_estimation_enabled: bool,
        people_matcher: Any,
        pu_inputs: _PeopleUpdateInputs,
        pu_album_title: str,
        pu_person_names: list[str],
        pu_subjects: list[str],
        pu_updated_det: dict[str, Any],
        pu_people_detected: bool,
        pu_people_identified: bool,
        pu_prompt_debug: PromptDebugSession | None,
    ) -> None:
        pu_album_title = _require_album_title_for_title_page(
            image_path=image_path,
            album_title=_resolve_title_page_album_title(
                image_path=image_path,
                album_title=pu_album_title,
                ocr_text=pu_inputs.existing_ocr_text,
            ),
            context="people update",
        )
        pu_dc_date, pu_date_time_original, pu_source_text = self._pu_resolve_dates(
            image_path=image_path,
            state=state,
            effective=effective,
            date_estimation_enabled=date_estimation_enabled,
            pu_album_title=pu_album_title,
            pu_inputs=pu_inputs,
            pu_prompt_debug=pu_prompt_debug,
        )
        text_layers, xmp_title, xmp_title_source = self._pu_resolve_text_and_title(
            image_path=image_path,
            state=state,
            pu_updated_det=pu_updated_det,
            pu_inputs=pu_inputs,
        )
        current_cast_signature = self._people_invalidation_signature(people_matcher)
        pu_updated_det = _pu_finalize_detections(
            pu_updated_det,
            existing_location=pu_inputs.existing_location,
            cast_store_signature=current_cast_signature,
            ocr_text=pu_inputs.existing_ocr_text,
            album_title=pu_album_title,
            stamp_date_hash=date_estimation_enabled or bool(pu_dc_date),
        )
        create_date = str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            person_names=pu_person_names,
            subjects=pu_subjects,
            title=xmp_title,
            title_source=xmp_title_source,
            description=str(state.get("description") or ""),
            album_title=pu_album_title,
            location_payload=pu_inputs.existing_location,
            source_text=pu_source_text,
            ocr_text=pu_inputs.existing_ocr_text,
            author_text=str(text_layers.get("author_text") or ""),
            scene_text=str(text_layers.get("scene_text") or ""),
            detections_payload=pu_updated_det,
            stitch_key=str(state.get("stitch_key") or ""),
            ocr_authority_source=str(state.get("ocr_authority_source") or ""),
            create_date=create_date,
            dc_date=pu_dc_date,
            date_time_original=pu_date_time_original,
            ocr_ran=bool(state.get("ocr_ran") or True),
            people_detected=pu_people_detected,
            people_identified=pu_people_identified,
            title_page_location=self.title_page_location,
        )

    # GPS-update path

    def _format_progress_prefix(self, idx: int, image_path: Path) -> str:
        eta_str = _format_eta(self.completed_times, len(self.files) - idx + 1)
        eta_part = f"  {eta_str}" if eta_str else ""
        return f"[{idx}/{len(self.files)}]{eta_part}  {_display_work_label(image_path)}"

    def _emit_ok(self, idx: int, image_path: Path, suffix: str = "") -> None:
        if self.stdout_only:
            return
        eta_str = _format_eta(self.completed_times, len(self.files) - idx)
        eta_part = f"  {eta_str}" if eta_str else ""
        suffix_text = f"  {suffix}" if suffix else ""
        print(
            f"[{idx}/{len(self.files)}]{eta_part}  ok    {image_path.name}{suffix_text}",
            flush=True,
        )

    # Full processing path

    def _process_full(
        self,
        idx: int,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        archive_stitched_ocr_required: bool,
        multi_scan_group_paths: list[Path],
        multi_scan_group_signature: str,
        extra_forced_steps: set[str] | None = None,
    ) -> None:
        file_start = time.monotonic()
        stop_ticker, set_step = self._begin_full_progress(idx, image_path)
        hints = _resolve_full_hints(image_path, self.printed_album_title_cache)
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))

        try:
            engines = self._init_full_engines(effective)
            scan_ocr_authority = self._resolve_full_scan_ocr_authority(
                archive_stitched_ocr_required=archive_stitched_ocr_required,
                image_path=image_path,
                multi_scan_group_paths=multi_scan_group_paths,
                multi_scan_group_signature=multi_scan_group_signature,
                ocr_engine=engines.ocr_engine,
                set_step=set_step,
                prompt_debug=prompt_debug,
            )
            with prepare_image_layout(image_path, split_mode="off") as layout:
                outcome = self._run_full_analysis(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    effective=effective,
                    hints=hints,
                    engines=engines,
                    layout=layout,
                    scan_ocr_authority=scan_ocr_authority,
                    people_matcher=people_matcher,
                    existing_xmp_people=existing_xmp_people,
                    existing_sidecar_state=existing_sidecar_state,
                    current_cast_signature=current_cast_signature,
                    multi_scan_group_signature=multi_scan_group_signature,
                    set_step=set_step,
                    prompt_debug=prompt_debug,
                    extra_forced_steps=extra_forced_steps,
                )
                if not self.dry_run:
                    self._write_full_payload(
                        sidecar_path=sidecar_path,
                        image_path=image_path,
                        effective=effective,
                        settings_sig=settings_sig,
                        date_estimation_enabled=date_estimation_enabled,
                        existing_sidecar_state=existing_sidecar_state,
                        people_matcher=people_matcher,
                        current_cast_signature=current_cast_signature,
                        layout=layout,
                        scan_ocr_authority=scan_ocr_authority,
                        outcome=outcome,
                    )
                    self._run_propagate_to_crops(image_path, outcome)

            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            if stop_ticker is not None:
                stop_ticker()
            self._emit_full_completion(idx, image_path, outcome)
            _mirror_page_sidecars(image_path)
        except Exception as exc:
            self.failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            if stop_ticker is not None:
                stop_ticker()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _begin_full_progress(self, idx: int, image_path: Path) -> tuple[Any, Any]:
        if self.stdout_only:
            return None, None
        prefix = self._format_progress_prefix(idx, image_path)
        print(prefix, flush=True)
        return _progress_ticker(prefix)

    def _init_full_engines(self, effective: dict[str, Any]) -> _FullEngines:
        object_detector = self._get_object_detector(effective) if bool(effective.get("enable_objects", True)) else None
        caption_key = self._caption_key_from_effective(effective)
        caption_engine = self._get_caption_engine_for_key(caption_key, effective, stream=not self.stdout_only)
        ocr_engine, ocr_key = self._get_ocr_engine(effective)
        return _FullEngines(
            caption_engine=caption_engine,
            caption_key=caption_key,
            ocr_engine=ocr_engine,
            ocr_key=ocr_key,
            object_detector=object_detector,
        )

    def _get_object_detector(self, effective: dict[str, Any]) -> Any:
        object_key = (
            str(effective.get("model", self.defaults["model"])),
            float(effective.get("object_threshold", self.defaults["object_threshold"])),
        )
        detector = self.object_detector_cache.get(object_key)
        if detector is None:
            detector = _init_object_detector(
                model_name=str(object_key[0]),
                confidence=float(object_key[1]),
            )
            self.object_detector_cache[object_key] = detector
        return detector

    def _get_ocr_engine(self, effective: dict[str, Any]) -> tuple[OCREngine, tuple[str, str, str, str]]:
        ocr_key = (
            str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
            str(effective.get("ocr_lang", self.defaults["ocr_lang"])),
            str(effective.get("ocr_model", self.defaults["ocr_model"])),
            normalize_lmstudio_base_url(str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"]))),
        )
        engine = self.ocr_engine_cache.get(ocr_key)
        if engine is None:
            engine = OCREngine(
                engine=ocr_key[0],
                language=ocr_key[1],
                model_name=ocr_key[2],
                base_url=ocr_key[3],
            )
            self.ocr_engine_cache[ocr_key] = engine
        return engine, ocr_key

    def _resolve_full_scan_ocr_authority(
        self,
        *,
        archive_stitched_ocr_required: bool,
        image_path: Path,
        multi_scan_group_paths: list[Path],
        multi_scan_group_signature: str,
        ocr_engine: OCREngine,
        set_step: Any,
        prompt_debug: PromptDebugSession | None,
    ) -> ArchiveScanOCRAuthority | None:
        if not archive_stitched_ocr_required:
            return None
        return _resolve_archive_scan_authoritative_ocr(
            image_path=image_path,
            group_paths=multi_scan_group_paths,
            group_signature=multi_scan_group_signature,
            cache=self.archive_scan_ocr_cache,
            ocr_engine=ocr_engine,
            step_fn=set_step,
            stitched_image_dir=self.stitch_cap_dir,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
        )

    def _build_full_step_runner(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        engines: _FullEngines,
        current_cast_signature: str,
        multi_scan_group_signature: str,
        existing_sidecar_state: dict | None,
        extra_forced_steps: set[str] | None,
    ) -> tuple[StepRunner, dict[str, Any]]:
        # propagate functions now inlined at module level

        crop_paths = _find_crop_paths_for_page(image_path)
        step_settings = {
            "ocr_engine": str(engines.ocr_key[0]),
            "ocr_model": str(engines.ocr_key[2]),
            "ocr_lang": str(engines.ocr_key[1]),
            "scan_group_signature": multi_scan_group_signature,
            "cast_store_signature": (current_cast_signature if bool(effective.get("enable_people", True)) else ""),
            "caption_engine": str(engines.caption_key[0]),
            "caption_model": str(engines.caption_key[1]),
            "nominatim_base_url": str(getattr(self.geocoder, "base_url", "") or "") if self.geocoder else "",
            "model": str(effective.get("model", self.defaults.get("model", ""))),
            "enable_objects": bool(effective.get("enable_objects", True)),
            "crop_paths_signature": _crop_paths_signature(crop_paths),
        }
        existing_pipeline_state = read_pipeline_state(sidecar_path)
        existing_detections = dict((existing_sidecar_state or {}).get("detections") or {})
        steps_arg = str(getattr(self.args, "steps", "") or "").strip()
        forced_steps = {s.strip() for s in steps_arg.split(",") if s.strip()} if steps_arg else set()
        if extra_forced_steps:
            forced_steps = forced_steps | extra_forced_steps
        runner = StepRunner(
            settings=step_settings,
            existing_pipeline_state=existing_pipeline_state,
            existing_detections=existing_detections,
            forced_steps=forced_steps,
        )
        return runner, existing_detections

    def _run_full_analysis(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        hints: _FullHints,
        engines: _FullEngines,
        layout: Any,
        scan_ocr_authority: ArchiveScanOCRAuthority | None,
        people_matcher: Any,
        existing_xmp_people: list[str],
        existing_sidecar_state: dict | None,
        current_cast_signature: str,
        multi_scan_group_signature: str,
        set_step: Any,
        prompt_debug: PromptDebugSession,
        extra_forced_steps: set[str] | None,
    ) -> _FullAnalysisOutcome:
        scan_filenames = _page_scan_filenames(image_path)
        if not scan_filenames and scan_ocr_authority is not None:
            scan_filenames = [path.name for path in scan_ocr_authority.group_paths]
        targets = _full_analysis_targets(image_path, layout, scan_ocr_authority)
        derived_ocr_override = _effective_sidecar_ocr_text(image_path, existing_sidecar_state)
        step_runner, existing_detections = self._build_full_step_runner(
            image_path=image_path,
            sidecar_path=sidecar_path,
            effective=effective,
            engines=engines,
            current_cast_signature=current_cast_signature,
            multi_scan_group_signature=multi_scan_group_signature,
            existing_sidecar_state=existing_sidecar_state,
            extra_forced_steps=extra_forced_steps,
        )
        analysis = _run_image_analysis(
            image_path=targets.analysis_target,
            people_image_path=targets.people_analysis_source,
            people_matcher=people_matcher,
            object_detector=engines.object_detector,
            ocr_engine=engines.ocr_engine,
            caption_engine=engines.caption_engine,
            requested_caption_engine=str(engines.caption_key[0]),
            ocr_engine_name=engines.ocr_key[0],
            ocr_language=engines.ocr_key[1],
            people_source_path=targets.people_analysis_source,
            people_bbox_offset=(_bounds_offset(layout.content_bounds) if layout.page_like else (0, 0)),
            caption_source_path=(image_path if layout.page_like else targets.analysis_target),
            album_title=hints.album_title_hint,
            printed_album_title=hints.album_title_hint,
            geocoder=self.geocoder,
            step_fn=set_step,
            extra_people_names=existing_xmp_people,
            is_page_scan=layout.page_like,
            ocr_text_override=(
                scan_ocr_authority.ocr_text if scan_ocr_authority is not None else (derived_ocr_override or None)
            ),
            context_ocr_text=hints.upstream_context_ocr,
            context_location_hint=hints.upstream_location_hint,
            prompt_debug=prompt_debug,
            title_page_location=self.title_page_location,
            step_runner=step_runner,
            existing_sidecar_state=existing_sidecar_state,
            metadata_engine=self._get_metadata_engine(effective),
        )
        resolved_album_title = analysis.album_title or hints.album_title_hint
        _store_album_printed_title_hint(
            image_path,
            self.printed_album_title_cache,
            resolved_album_title,
        )
        person_names = _dedupe(analysis.people_names + existing_xmp_people)
        subjects = _dedupe(analysis.subjects + ([resolved_album_title] if resolved_album_title else []))
        description = _build_flat_page_description(analysis=analysis) if layout.page_like else analysis.description
        payload = _build_flat_payload(layout, analysis)
        analysis_mode = "page_flat" if layout.page_like else "single_image"
        ocr_authority_hash = str(scan_ocr_authority.ocr_hash) if scan_ocr_authority is not None else ""
        payload = _refresh_detection_model_metadata(
            payload,
            ocr_model=_full_engine_model_name(engines.ocr_engine, engines.ocr_key[0]),
            caption_model=_full_engine_model_name(engines.caption_engine, engines.caption_key[0]),
        )
        return _FullAnalysisOutcome(
            analysis=analysis,
            payload=payload,
            person_names=person_names,
            subjects=subjects,
            description=description,
            ocr_text=analysis.ocr_text,
            resolved_album_title=resolved_album_title,
            analysis_mode=analysis_mode,
            ocr_authority_hash=ocr_authority_hash,
            scan_filenames=scan_filenames,
            step_runner=step_runner,
            existing_detections=existing_detections,
        )

    @staticmethod
    def _full_resolve_album_and_dates(
        image_path: Path,
        outcome: _FullAnalysisOutcome,
        existing_sidecar_state: dict | None,
    ) -> tuple[str, str, str]:
        final_album_title = _require_album_title_for_title_page(
            image_path=image_path,
            album_title=_resolve_title_page_album_title(
                image_path=image_path,
                album_title=(outcome.resolved_album_title or _resolve_album_title_hint(image_path)),
                ocr_text=outcome.ocr_text,
            ),
            context="write",
        )
        final_dc_date = _full_final_dc_date(outcome.analysis, existing_sidecar_state)
        existing_dto = str((existing_sidecar_state or {}).get("date_time_original") or "")
        final_date_time_original = _resolve_date_time_original(
            dc_date=final_dc_date,
            date_time_original=existing_dto,
        )
        return final_album_title, final_dc_date, final_date_time_original

    @staticmethod
    def _full_resolve_text_layers(
        image_path: Path,
        outcome: _FullAnalysisOutcome,
        layout: Any,
        scan_ocr_authority: ArchiveScanOCRAuthority | None,
    ) -> tuple[dict[str, Any], str, str]:
        ocr_authority_source = "archive_stitched" if scan_ocr_authority is not None else ""
        text_layers = _resolve_xmp_text_layers(
            image_path=image_path,
            ocr_text=outcome.ocr_text,
            page_like=bool(layout.page_like),
            ocr_authority_source=ocr_authority_source,
            author_text=str(outcome.analysis.author_text or ""),
            scene_text=str(outcome.analysis.scene_text or ""),
        )
        xmp_title, xmp_title_source = _compute_xmp_title(
            image_path=image_path,
            explicit_title=str(outcome.analysis.title or ""),
            author_text=str(text_layers.get("author_text") or ""),
        )
        return text_layers, xmp_title, xmp_title_source

    def _write_full_payload(
        self,
        *,
        sidecar_path: Path,
        image_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        people_matcher: Any,
        current_cast_signature: str,
        layout: Any,
        scan_ocr_authority: ArchiveScanOCRAuthority | None,
        outcome: _FullAnalysisOutcome,
    ) -> None:
        payload = outcome.payload
        analysis = outcome.analysis
        _merge_pipeline_records(payload, outcome.existing_detections, outcome.step_runner)
        location_payload = _full_resolve_location_payload(
            payload, outcome.step_runner, image_path, existing_sidecar_state
        )
        if location_payload:
            payload["location"] = location_payload
        final_album_title, final_dc_date, final_date_time_original = self._full_resolve_album_and_dates(
            image_path, outcome, existing_sidecar_state
        )
        text_layers, xmp_title, xmp_title_source = self._full_resolve_text_layers(
            image_path, outcome, layout, scan_ocr_authority
        )
        if people_matcher is not None:
            current_cast_signature = self._people_invalidation_signature(people_matcher)
        ocr_authority_source = "archive_stitched" if scan_ocr_authority is not None else ""
        payload["processing"] = _full_processing_payload(
            image_path=image_path,
            settings_sig=settings_sig,
            current_cast_signature=current_cast_signature,
            effective=effective,
            ocr_text=outcome.ocr_text,
            final_album_title=final_album_title,
            final_dc_date=final_dc_date,
            existing_sidecar_state=existing_sidecar_state,
            scan_ocr_authority=scan_ocr_authority,
            ocr_authority_hash=outcome.ocr_authority_hash,
            analysis_mode=outcome.analysis_mode,
            date_estimation_enabled=date_estimation_enabled,
        )
        ocr_engine_name = str(effective.get("ocr_engine", self.defaults["ocr_engine"])).lower()
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            person_names=outcome.person_names,
            subjects=outcome.subjects,
            title=xmp_title,
            title_source=xmp_title_source,
            description=outcome.description,
            album_title=final_album_title,
            location_payload=location_payload,
            source_text=_build_dc_source(final_album_title, image_path, outcome.scan_filenames),
            ocr_text=outcome.ocr_text,
            ocr_lang=str(analysis.ocr_lang or ""),
            author_text=str(text_layers.get("author_text") or ""),
            scene_text=str(text_layers.get("scene_text") or ""),
            detections_payload=payload,
            subphotos=None,
            ocr_authority_source=ocr_authority_source,
            create_date=read_embedded_create_date(image_path),
            dc_date=final_dc_date,
            date_time_original=final_date_time_original,
            ocr_ran=ocr_engine_name != "none",
            people_detected=analysis.faces_detected > 0 or len(outcome.person_names) > 0,
            people_identified=len(outcome.person_names) > 0,
            title_page_location=self.title_page_location,
        )

    def _run_propagate_to_crops(self, image_path: Path, outcome: _FullAnalysisOutcome) -> None:

        locations_out = dict(outcome.payload.get("location") or {})
        people_out = list(outcome.payload.get("people") or [])

        def _do_propagate() -> dict:
            return run_propagate_to_crops(
                image_path,
                location_payload=locations_out,
                people_payload=people_out,
                default_location=self.title_page_location or None,
            )

        outcome.step_runner.run("propagate-to-crops", _do_propagate)

    def _emit_full_completion(self, idx: int, image_path: Path, outcome: _FullAnalysisOutcome) -> None:
        if self.stdout_only:
            payload = outcome.payload
            caption_meta = dict(payload.get("caption") or {}) if isinstance(payload, dict) else {}
            fallback_error = str(caption_meta.get("error") or "").strip()
            if bool(caption_meta.get("fallback")) and fallback_error:
                self.emit_error(
                    f"[{idx}/{len(self.files)}] warn  {image_path.name}: caption fallback: {fallback_error}"
                )
            print(f"{image_path.name}: {outcome.description}" if outcome.description else image_path.name)
            return
        self._emit_ok(idx, image_path)


def refresh_rendered_view_people_metadata(
    image_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
) -> None:
    rendered_image_path = Path(image_path)
    rendered_sidecar_path = Path(sidecar_path) if sidecar_path is not None else rendered_image_path.with_suffix(".xmp")
    if not has_valid_sidecar(rendered_image_path):
        raise RuntimeError(f"Rendered sidecar missing or invalid for people refresh: {rendered_sidecar_path}")
    existing_sidecar_state = read_ai_sidecar_state(rendered_sidecar_path)
    if not isinstance(existing_sidecar_state, dict):
        raise RuntimeError(f"Rendered sidecar could not be parsed for people refresh: {rendered_sidecar_path}")

    runner = IndexRunner(["--photo", str(rendered_image_path), "--include-view"])
    runner.files = [rendered_image_path]
    effective, settings_sig, date_estimation_enabled = runner._resolve_effective_settings(rendered_image_path)
    people_matcher, current_cast_signature = runner._get_people_matcher_and_signature(effective)
    if people_matcher is None:
        return

    runner._process_people_update(
        1,
        image_path=rendered_image_path,
        sidecar_path=rendered_sidecar_path,
        effective=effective,
        settings_sig=settings_sig,
        date_estimation_enabled=date_estimation_enabled,
        existing_sidecar_state=existing_sidecar_state,
        existing_xmp_people=read_person_in_image(rendered_sidecar_path),
        people_matcher=people_matcher,
        current_cast_signature=current_cast_signature,
        chain_gps=False,
        preserve_existing_xmp_people=False,
        raise_on_error=True,
    )
