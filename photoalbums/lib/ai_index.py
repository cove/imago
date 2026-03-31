from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .ai_page_layout import PreparedImageLayout, prepare_image_layout
from .ai_geocode import NominatimGeocoder
from .ai_render_settings import (
    find_archive_dir_for_image,
    load_render_settings,
    resolve_effective_settings,
)
from .image_limits import allow_large_pillow_images
from .prompt_debug import PromptDebugSession
from ..common import PHOTO_ALBUMS_DIR
from ..exiftool_utils import read_tag
from ..naming import (
    BASE_PAGE_NAME_RE,
    DERIVED_NAME_RE,
    DERIVED_VIEW_RE,
    SCAN_TIFF_RE,
    parse_album_filename,
    SCAN_NAME_RE,
)
from .xmp_sidecar import (
    _dedupe,
    _normalize_xmp_datetime,
    _resolve_date_time_original,
    read_ai_sidecar_state,
    read_person_in_image,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
)
from .xmp_review import load_ai_xmp_review


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


def _format_reprocess_reasons(reasons: list[str]) -> str:
    clean = _dedupe([str(reason or "").strip() for reason in reasons])
    return ", ".join(clean)


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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MIN_EXISTING_SIDECAR_BYTES = 100
AI_MODEL_MAX_SOURCE_BYTES = 30 * 1024 * 1024
DEFAULT_CREATOR_TOOL = "https://github.com/cove/imago"
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"
PROCESSOR_SIGNATURE = "page_split_v17_people_recovery_any_people"
JOB_ARTIFACTS_ENV = "IMAGO_JOB_ARTIFACTS"
JOB_ID_ENV = "IMAGO_JOB_ID"
PROCESSING_LOCK_SUFFIX = ".photoalbums-ai.lock"
BATCH_LOCK_SUFFIX = ".photoalbums-ai.batch.lock"
CAST_STORE_RETRY_ATTEMPTS = 6
CAST_STORE_RETRY_DELAY_SECONDS = 0.5


@dataclass
class ImageAnalysis:
    image_path: Path
    people_names: list[str]
    object_labels: list[str]
    ocr_text: str
    ocr_keywords: list[str]
    subjects: list[str]
    description: str
    payload: dict[str, Any]
    author_text: str = ""
    scene_text: str = ""
    faces_detected: int = 0
    image_regions: list[dict] = None
    album_title: str = ""
    title: str = ""
    ocr_lang: str = ""

    def __post_init__(self):
        if self.image_regions is None:
            self.image_regions = []


@dataclass(frozen=True)
class ArchiveScanOCRAuthority:
    page_key: str
    group_paths: tuple[Path, ...]
    signature: str
    ocr_text: str
    ocr_keywords: tuple[str, ...]
    ocr_hash: str
    stitched_image_path: Path | None = None


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


def _album_identity_key(image_path: Path) -> str:
    collection, year, book, _page = parse_album_filename(image_path.name)
    if collection != "Unknown":
        return f"{collection}_{year}_B{book}".casefold()
    parent_name = str(image_path.parent.name or "")
    base_name = parent_name.removesuffix("_Archive").removesuffix("_View")
    return str((image_path.parent.parent / base_name).resolve()).casefold()


def _album_directory_candidates(image_path: Path) -> list[Path]:
    out: list[Path] = [image_path.parent]
    parent_name = str(image_path.parent.name or "")
    base_name = parent_name.removesuffix("_Archive").removesuffix("_View")
    root = image_path.parent.parent
    for suffix in ("_Archive", "_View"):
        candidate = root / f"{base_name}{suffix}"
        if candidate in out or not candidate.is_dir():
            continue
        out.append(candidate)
    return out


def _iter_album_cover_sidecars(image_path: Path):
    collection, year, book, _page = parse_album_filename(image_path.name)
    target_prefix = ""
    if collection != "Unknown":
        target_prefix = f"{collection}_{year}_B{book}_".casefold()
    seen: set[str] = set()
    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for folder in _album_directory_candidates(image_path):
        for sidecar_path in sorted(folder.glob("*.xmp")):
            match = _cover_sidecar_match(sidecar_path)
            if match is None:
                continue
            if target_prefix and not Path(sidecar_path).stem.casefold().startswith(target_prefix):
                continue
            sidecar_key = str(sidecar_path.resolve()).casefold()
            if sidecar_key in seen:
                continue
            seen.add(sidecar_key)
            page_rank = int(match.group("page"))
            scan_match = _scan_name_match(sidecar_path)
            kind_rank = 1 if scan_match is not None else 0
            candidates.append(((page_rank, kind_rank, sidecar_path.name.casefold()), sidecar_path))
    for _sort_key, sidecar_path in sorted(candidates, key=lambda item: item[0]):
        yield sidecar_path


def _iter_album_p01_sidecars(image_path: Path):
    for sidecar_path in _iter_album_cover_sidecars(image_path):
        match = _cover_sidecar_match(sidecar_path)
        if match is None:
            continue
        if int(match.group("page")) == 1:
            yield sidecar_path


def _scan_name_match(path: str | Path):
    return SCAN_NAME_RE.fullmatch(Path(path).stem)


def _derived_name_match(path: str | Path):
    return DERIVED_NAME_RE.fullmatch(Path(path).stem)


def _base_page_name_match(path: str | Path):
    return BASE_PAGE_NAME_RE.fullmatch(Path(path).stem)


def _title_page_scan_match(path: str | Path):
    match = _scan_name_match(path)
    if match is None:
        return None
    try:
        page_number = int(match.group("page"))
        scan_number = int(match.group("scan"))
    except (ValueError, IndexError):
        return None
    if page_number == 1 and scan_number == 1:
        return match
    return None


def _title_page_base_match(path: str | Path):
    match = _base_page_name_match(path)
    if match is None:
        return None
    try:
        page_number = int(match.group("page"))
    except (ValueError, IndexError):
        return None
    if page_number == 1:
        return match
    return None


def _title_page_match(path: str | Path):
    return _title_page_scan_match(path) or _title_page_base_match(path)


def _cover_sidecar_match(path: str | Path):
    return _title_page_match(path)


def _title_page_dependency_sort_key(path: Path) -> tuple[int, int, int, int, str]:
    _, _, _, page = parse_album_filename(path.name)
    try:
        page_number = int(str(page or "").strip())
    except ValueError:
        page_number = 999
    scan_match = _scan_name_match(path)
    derived_match = _derived_name_match(path)
    if scan_match is not None:
        kind_rank = 0
        item_number = int(scan_match.group("scan"))
    elif derived_match is None:
        kind_rank = 1
        item_number = 0
    else:
        kind_rank = 2
        item_number = int(derived_match.group("derived"))
    return (
        page_number,
        kind_rank,
        item_number,
        len(path.name),
        path.name.casefold(),
    )


def _is_album_title_source_candidate(image_path: Path) -> bool:
    return _title_page_match(image_path) is not None


def _iter_album_title_page_images(image_path: Path, extensions: set[str]):
    seen: set[str] = set()
    album_key = _album_identity_key(image_path)
    candidates: list[Path] = []
    for folder in _album_directory_candidates(image_path):
        try:
            rows = list(folder.iterdir())
        except FileNotFoundError:
            continue
        for candidate in rows:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in extensions:
                continue
            if _album_identity_key(candidate) != album_key:
                continue
            if _title_page_match(candidate) is None:
                continue
            key = str(candidate.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    for candidate in sorted(candidates, key=_title_page_dependency_sort_key):
        yield candidate


def _resolve_album_title_dependencies(image_path: Path, extensions: set[str]) -> list[Path]:
    if _is_album_title_source_candidate(image_path):
        return []
    if _album_title_valid_in_sidecars(image_path):
        return []
    current_key = str(image_path.resolve()).casefold()
    candidates = [
        path
        for path in _iter_album_title_page_images(image_path, extensions)
        if str(path.resolve()).casefold() != current_key
    ]
    source_candidates = [path for path in candidates if _is_album_title_source_candidate(path)]
    return source_candidates or candidates


def _expand_album_title_dependencies(files: list[Path], extensions: set[str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    for image_path in files:
        for dependency in _resolve_album_title_dependencies(image_path, extensions):
            dep_key = str(dependency.resolve()).casefold()
            if dep_key in seen:
                continue
            seen.add(dep_key)
            expanded.append(dependency)
        image_key = str(image_path.resolve()).casefold()
        if image_key in seen:
            continue
        seen.add(image_key)
        expanded.append(image_path)
    return expanded


def _read_album_title_from_sidecar_iter(sidecar_iter) -> str:
    for sidecar_path in sidecar_iter:
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        album_title = str(state.get("album_title") or "").strip()
        if album_title:
            return album_title
    return ""


def _resolve_album_title_from_sidecars(image_path: Path) -> str:
    """Read album title from the P01 XMP sidecar. Returns '' if not yet processed."""
    return _read_album_title_from_sidecar_iter(_iter_album_p01_sidecars(image_path))


def _album_title_valid_in_sidecars(image_path: Path) -> bool:
    """Return True only if the P01 sidecar exists and its album_title matches its ocr_text.

    A mismatch means the title was set by the AI caption (not OCR) and needs to be
    reprocessed so the OCR-authoritative value is written.
    """
    for sidecar_path in _iter_album_p01_sidecars(image_path):
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        album_title = str(state.get("album_title") or "").strip()
        ocr_text = str(state.get("ocr_text") or "").strip()
        if album_title and ocr_text and album_title == ocr_text:
            return True
    return False


def _resolve_album_title_hint(image_path: Path) -> str:
    return _resolve_album_title_from_sidecars(image_path)


def _resolve_album_printed_title_from_sidecars(image_path: Path) -> str:
    return _read_album_title_from_sidecar_iter(_iter_album_cover_sidecars(image_path))


def _resolve_album_printed_title_hint(image_path: Path, printed_title_cache: dict[str, str]) -> str:
    key = _album_identity_key(image_path)
    cached = str(printed_title_cache.get(key) or "").strip()
    if cached:
        return cached
    title = _resolve_album_printed_title_from_sidecars(image_path)
    if title:
        printed_title_cache[key] = title
    return title


def _store_album_printed_title_hint(image_path: Path, printed_title_cache: dict[str, str], title: str) -> str:
    value = str(title or "").strip()
    if value:
        printed_title_cache[_album_identity_key(image_path)] = value
    return value


def _looks_like_album_title_page(image_path: Path) -> bool:
    return _title_page_match(image_path) is not None


def _require_album_title_for_title_page(
    *,
    image_path: Path,
    album_title: str,
    context: str,
) -> str:
    value = clean_text(str(album_title or ""))
    if value:
        return value
    if _is_album_title_source_candidate(image_path):
        raise RuntimeError(f"Missing album title for title page during {context}: {image_path}")
    return ""


def _resolve_title_page_album_title(
    *,
    image_path: Path,
    album_title: str,
    ocr_text: str,
) -> str:
    value = clean_text(str(album_title or ""))
    if value:
        return value
    if _is_album_title_source_candidate(image_path):
        return clean_text(str(ocr_text or ""))
    return ""


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
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


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


def _page_scan_filenames(image_path: Path) -> list[str]:
    """Return sorted list of scan TIF basenames associated with image_path's page.

    For scan TIFs (_S##): returns all sibling TIFs sharing the same page key.
    For derived or stitched/base page images: finds archive TIF scans for the same page.
    Returns [] if no scans are found.
    """
    if _scan_name_match(image_path):
        return [p.name for p in sorted(_scan_group_paths(image_path))]
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is None or not archive_dir.is_dir():
        return []
    _, _, _, page_str = parse_album_filename(image_path.name)
    # parse_album_filename returns "00" when no page token is found (its default).
    # Page 00 is not a valid archive page, so treat it the same as unparseable.
    if not page_str.isdigit() or int(page_str) == 0:
        return []
    page_int = int(page_str)
    scans: list[Path] = sorted(
        p
        for p in archive_dir.iterdir()
        for sm in (SCAN_TIFF_RE.match(p.name),)
        if sm and int(sm.group("page")) == page_int
    )
    return [p.name for p in scans]


def _build_dc_source(album_title: str, image_path: Path, scan_filenames: list[str]) -> str:
    """Build a human-readable dc:source string followed by source scan filenames.

    e.g. "Mainland China 1986 Book 11 Page 02 Scan(s) S01 S02; China_1986_B02_P17_S01.tif; ..."
    """
    _, _, _, _page_str = parse_album_filename(image_path.name)
    page_number = int(_page_str) if _page_str.isdigit() else 0
    scan_match = _scan_name_match(image_path)
    if scan_match:
        source_filenames = [image_path.name]
        scan_nums = [int(scan_match.group("scan"))]
    else:
        source_filenames = _dedupe([str(fn or "").strip() for fn in scan_filenames if str(fn or "").strip()])
        scan_nums = sorted(int(sm.group("scan")) for fn in source_filenames if (sm := _scan_name_match(fn)))
    parts: list[str] = [p for p in [str(album_title or "").strip()] if p]
    if page_number > 0:
        parts.append(f"Page {page_number:02d}")
    if scan_nums:
        parts.append("Scan(s) " + " ".join(f"S{n:02d}" for n in scan_nums))
    label = " ".join(parts)
    return "; ".join(p for p in [label] + source_filenames if p)


def _dc_source_needs_refresh(image_path: Path, sidecar_state: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar_state, dict):
        return False
    source_text = str(sidecar_state.get("source_text") or "").strip()
    album_title = str(sidecar_state.get("album_title") or "").strip()
    if not album_title and " Page " in source_text:
        album_title = source_text.split(" Page ", 1)[0].strip()
    expected_source = _build_dc_source(album_title, image_path, _page_scan_filenames(image_path))
    return source_text != expected_source


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


def _xmp_gps_to_decimal(value: object, *, axis: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "," not in text or len(text) < 2:
        return _normalize_gps_value(text, axis=axis)
    hemisphere = text[-1:].upper()
    body = text[:-1]
    if axis == "lat" and hemisphere not in {"N", "S"}:
        return _normalize_gps_value(text, axis=axis)
    if axis == "lon" and hemisphere not in {"E", "W"}:
        return _normalize_gps_value(text, axis=axis)
    degrees_text, minutes_text = body.split(",", 1)
    try:
        degrees = int(degrees_text.strip())
        minutes = float(minutes_text.strip())
    except ValueError:
        return _normalize_gps_value(text, axis=axis)
    decimal = float(degrees) + (minutes / 60.0)
    if hemisphere in {"S", "W"}:
        decimal = -decimal
    return f"{decimal:.8f}".rstrip("0").rstrip(".")


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
    parser.add_argument("--creator-tool", default=DEFAULT_CREATOR_TOOL, help="XMP CreatorTool value.")
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
    parser.add_argument("--include-view", action="store_true", help="Include files in *_View folders.")
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
    return parser.parse_args(argv)


def _init_people_matcher(
    *,
    cast_store: Path,
    min_similarity: float,
    min_face_size: int,
):
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
):
    if not str(model_name or "").strip():
        return None
    from .ai_objects import YOLOObjectDetector

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
):
    return CaptionEngine(
        engine=str(engine),
        model_name=str(model_name),
        caption_prompt=str(caption_prompt),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        lmstudio_base_url=str(lmstudio_base_url),
        max_image_edge=int(max_image_edge),
        stream=stream,
    )


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


def _dc_date_needs_refresh(
    image_path: Path,
    sidecar_state: dict[str, Any] | None,
    *,
    enabled: bool,
) -> bool:
    if not isinstance(sidecar_state, dict):
        return False
    current_dc_date = str(sidecar_state.get("dc_date") or "").strip()
    current_date_time_original = str(sidecar_state.get("date_time_original") or "").strip()
    if current_dc_date:
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
    existing_dc_date: str,
    ocr_text: str,
    album_title: str,
    image_path: Path,
    date_engine: DateEstimateEngine | None,
    prompt_debug: PromptDebugSession | None,
) -> str:
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
        "creator_tool": str(settings.get("creator_tool", DEFAULT_CREATOR_TOOL)),
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


def _build_caption_metadata(
    *,
    requested_engine: str,
    effective_engine: str,
    fallback: bool,
    error: str,
    engine_error: str,
    model: str,
    people_present: bool = False,
    estimated_people_count: int = 0,
) -> dict[str, Any]:
    return {
        "requested_engine": str(requested_engine),
        "effective_engine": str(effective_engine),
        "fallback": bool(fallback),
        "error": str(error or "")[:500],
        "engine_error": str(engine_error or ""),
        "model": str(model or ""),
        "people_present": bool(people_present),
        "estimated_people_count": max(0, int(estimated_people_count)),
    }


def _refresh_detection_model_metadata(
    detections: dict[str, Any] | None,
    *,
    ocr_model: str,
    caption_model: str,
) -> dict[str, Any]:
    updated = dict(detections or {})
    ocr_payload = dict(updated.get("ocr") or {})
    ocr_payload["model"] = str(ocr_model or "")
    updated["ocr"] = ocr_payload
    caption_payload = dict(updated.get("caption") or {})
    caption_payload["model"] = str(caption_model or "")
    updated["caption"] = caption_payload
    return updated


_COORDINATE_LABEL_RE = re.compile(
    r"\b(?P<label>lat(?:itude)?|lon(?:gitude)?|long)\b\s*[:=]?\s*"
    r"(?P<value>.+?)(?=(?:\b(?:lat(?:itude)?|lon(?:gitude)?|long)\b)|[\n\r;]|$)",
    flags=re.IGNORECASE,
)
_COORDINATE_HEMISPHERE_RE = re.compile(
    r"(?:\d{1,3}(?:\.\d+)?\s*[NSEW])"
    r"|(?:\d{1,3}\s*[°º]\s*\d{1,2}\s*[′']\s*\d{1,2}(?:\.\d+)?\s*[″\"]?\s*[NSEW])",
    flags=re.IGNORECASE,
)


def _estimate_people_from_detections(
    *,
    people_matches: list | None = None,
    people_names: list[str] | None = None,
    object_labels: list[str] | None = None,
    faces_detected: int = 0,
) -> tuple[bool, int]:
    object_person_count = sum(1 for label in list(object_labels or []) if str(label).strip().casefold() == "person")
    estimated_people_count = max(
        0,
        int(faces_detected or 0),
        len(list(people_matches or [])),
        len(list(people_names or [])),
        int(object_person_count),
    )
    return estimated_people_count > 0, estimated_people_count


def _caption_people_name_score(text: str, people_names: list[str] | None = None) -> int:
    caption_text = str(text or "").casefold()
    score = 0
    for name in _dedupe([str(item or "").strip() for item in list(people_names or [])]):
        if not name:
            continue
        normalized_name = name.casefold()
        if normalized_name in caption_text:
            score += 2
            continue
        first_token = normalized_name.split()[0]
        if len(first_token) >= 4 and re.search(rf"\b{re.escape(first_token)}\b", caption_text):
            score += 1
    return score


def _merge_people_estimates(
    *,
    local_people_present: bool,
    local_estimated_people_count: int,
    model_people_present: bool,
    model_estimated_people_count: int,
) -> tuple[bool, int]:
    estimated_people_count = max(
        0,
        int(local_estimated_people_count),
        int(model_estimated_people_count),
    )
    people_present = bool(local_people_present or model_people_present or estimated_people_count > 0)
    return people_present, estimated_people_count


def _merge_location_estimates(
    *,
    local_gps_latitude: str,
    local_gps_longitude: str,
    model_gps_latitude: str,
    model_gps_longitude: str,
    model_location_name: str,
) -> tuple[str, str, str]:
    lat_text = str(local_gps_latitude or "").strip()
    lon_text = str(local_gps_longitude or "").strip()
    if lat_text and lon_text:
        return lat_text, lon_text, str(model_location_name or "").strip()
    model_lat = str(model_gps_latitude or "").strip()
    model_lon = str(model_gps_longitude or "").strip()
    return model_lat, model_lon, str(model_location_name or "").strip()


def _resolve_people_count_metadata(
    *,
    requested_caption_engine: str,
    caption_engine: Any,
    model_image_path: Path,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: Path,
    album_title: str,
    printed_album_title: str,
    people_positions: dict[str, str],
    local_people_present: bool,
    local_estimated_people_count: int,
    prompt_debug: PromptDebugSession | None = None,
    debug_step: str = "people_count",
) -> tuple[bool, int]:
    if str(requested_caption_engine or "").strip().lower() != "lmstudio":
        return local_people_present, local_estimated_people_count
    estimate_people = getattr(caption_engine, "estimate_people", None)
    if not callable(estimate_people):
        return local_people_present, local_estimated_people_count
    try:
        result = estimate_people(
            image_path=model_image_path,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step=debug_step,
        )
    except Exception:
        return local_people_present, local_estimated_people_count
    fallback = getattr(result, "fallback", False)
    if not isinstance(fallback, bool) or fallback:
        return local_people_present, local_estimated_people_count
    model_people_present = getattr(result, "people_present", False)
    model_estimated_people_count = getattr(result, "estimated_people_count", 0)
    if not isinstance(model_people_present, bool):
        model_people_present = False
    if isinstance(model_estimated_people_count, bool):
        model_estimated_people_count = 0
    try:
        model_estimated_people_count = max(0, int(model_estimated_people_count or 0))
    except Exception:
        model_estimated_people_count = 0
    return _merge_people_estimates(
        local_people_present=local_people_present,
        local_estimated_people_count=local_estimated_people_count,
        model_people_present=model_people_present,
        model_estimated_people_count=model_estimated_people_count,
    )


def _resolve_location_metadata(
    *,
    requested_caption_engine: str,
    caption_engine: Any,
    model_image_path: Path,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: Path,
    album_title: str,
    printed_album_title: str,
    people_positions: dict[str, str],
    fallback_location_name: str,
    prompt_debug: PromptDebugSession | None = None,
    debug_step: str = "location",
) -> tuple[str, str, str]:
    local_gps_latitude, local_gps_longitude = _extract_explicit_gps_from_text(ocr_text)
    if str(requested_caption_engine or "").strip().lower() != "lmstudio":
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    estimate_location = getattr(caption_engine, "estimate_location", None)
    if not callable(estimate_location):
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    try:
        result = estimate_location(
            image_path=model_image_path,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step=debug_step,
        )
    except Exception:
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    fallback = getattr(result, "fallback", False)
    if not isinstance(fallback, bool) or fallback:
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    return _merge_location_estimates(
        local_gps_latitude=local_gps_latitude,
        local_gps_longitude=local_gps_longitude,
        model_gps_latitude=str(getattr(result, "gps_latitude", "") or "").strip(),
        model_gps_longitude=str(getattr(result, "gps_longitude", "") or "").strip(),
        model_location_name=(
            str(getattr(result, "location_name", "") or "").strip() or str(fallback_location_name or "").strip()
        ),
    )


def _extract_explicit_gps_from_text(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""

    lat_text = ""
    lon_text = ""
    for match in _COORDINATE_LABEL_RE.finditer(raw):
        label = str(match.group("label") or "").casefold()
        axis = "lat" if label.startswith("lat") else "lon"
        value = _normalize_gps_value(str(match.group("value") or ""), axis=axis)
        if not value:
            continue
        if axis == "lat" and not lat_text:
            lat_text = value
        if axis == "lon" and not lon_text:
            lon_text = value
        if lat_text and lon_text:
            return lat_text, lon_text

    for match in _COORDINATE_HEMISPHERE_RE.finditer(raw):
        value = str(match.group(0) or "").strip()
        if not value:
            continue
        upper_value = value.upper()
        if any(marker in upper_value for marker in ("N", "S")) and not lat_text:
            lat_text = _normalize_gps_value(value, axis="lat")
        if any(marker in upper_value for marker in ("E", "W")) and not lon_text:
            lon_text = _normalize_gps_value(value, axis="lon")
        if lat_text and lon_text:
            return lat_text, lon_text

    return ("", "") if not (lat_text and lon_text) else (lat_text, lon_text)


def _serialize_people_matches(people_matches: list) -> list[dict[str, Any]]:
    return [
        {
            "name": row.name,
            "score": round(row.score, 5),
            "certainty": round(float(getattr(row, "certainty", row.score)), 5),
            "reviewed_by_human": bool(getattr(row, "reviewed_by_human", False)),
            "face_id": str(getattr(row, "face_id", "") or ""),
            **({"bbox": [int(v) for v in row.bbox[:4]]} if getattr(row, "bbox", None) else {}),
        }
        for row in people_matches
    ]


def _merge_people_matches(*match_groups: list) -> list:
    merged: dict[str, Any] = {}
    for group in match_groups:
        for row in list(group or []):
            name = str(getattr(row, "name", "") or "").strip()
            if not name:
                continue
            current = merged.get(name)
            if current is None:
                merged[name] = row
                continue
            row_certainty = float(getattr(row, "certainty", getattr(row, "score", 0.0)) or 0.0)
            current_certainty = float(getattr(current, "certainty", getattr(current, "score", 0.0)) or 0.0)
            row_score = float(getattr(row, "score", 0.0) or 0.0)
            current_score = float(getattr(current, "score", 0.0) or 0.0)
            if row_certainty > current_certainty or (row_certainty == current_certainty and row_score > current_score):
                merged[name] = row
    out = list(merged.values())
    out.sort(
        key=lambda row: (
            -float(getattr(row, "certainty", getattr(row, "score", 0.0)) or 0.0),
            -float(getattr(row, "score", 0.0) or 0.0),
            str(getattr(row, "name", "") or "").casefold(),
        )
    )
    return out


def _resolve_location_payload(
    *,
    geocoder: NominatimGeocoder | None,
    gps_latitude: str,
    gps_longitude: str,
    location_name: str,
) -> dict[str, Any]:
    lat_text = str(gps_latitude or "").strip()
    lon_text = str(gps_longitude or "").strip()
    query = str(location_name or "").strip()
    # Reject generic place-type descriptions (e.g. "a beach", "a park") — they
    # are not named places and produce spurious Nominatim results.
    if re.match(r"^(?:a|an)\s+\S", query, re.IGNORECASE):
        query = ""
    if lat_text and lon_text:
        payload: dict[str, Any] = {
            "gps_latitude": float(lat_text),
            "gps_longitude": float(lon_text),
            "map_datum": "WGS-84",
            "source": "caption",
        }
        if query:
            payload["query"] = query
        return payload
    geocode_error = ""
    if query and geocoder is not None:
        try:
            result = geocoder.geocode(query)
        except Exception as exc:
            result = None
            geocode_error = str(exc or "").strip()
        if result is not None:
            loc: dict[str, Any] = {
                "query": result.query,
                "display_name": result.display_name,
                "gps_latitude": float(result.latitude),
                "gps_longitude": float(result.longitude),
                "map_datum": "WGS-84",
                "source": result.source,
            }
            if str(getattr(result, "city", "") or "").strip():
                loc["city"] = str(result.city).strip()
            if str(getattr(result, "state", "") or "").strip():
                loc["state"] = str(result.state).strip()
            if str(getattr(result, "country", "") or "").strip():
                loc["country"] = str(result.country).strip()
            return loc
    if query and geocode_error:
        return {
            "query": query,
            "error": geocode_error,
            "source": "nominatim",
        }
    return {}


@contextlib.contextmanager
def _prepare_ai_model_image(image_path: Path):
    path = Path(image_path)
    try:
        source_size = int(path.stat().st_size)
    except FileNotFoundError:
        yield path
        return
    if source_size <= AI_MODEL_MAX_SOURCE_BYTES:
        yield path
        return

    try:
        from PIL import Image, ImageOps  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(Image)
    except Exception:
        yield path
        return

    temp_dir = tempfile.TemporaryDirectory(prefix="imago-ai-")
    try:
        out_path = Path(temp_dir.name) / f"{path.stem}_ai.jpg"
        with Image.open(str(path)) as image:
            working = ImageOps.exif_transpose(image)
            if working.mode not in {"RGB", "L"}:
                working = working.convert("RGB")
            width, height = working.size
            scale = min(
                0.95,
                max(
                    0.2,
                    ((AI_MODEL_MAX_SOURCE_BYTES / float(max(1, source_size))) ** 0.5) * 0.92,
                ),
            )
            quality = 90
            candidate = working
            created_candidate = False
            while True:
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                if new_size != candidate.size:
                    if created_candidate:
                        candidate.close()
                    resampling = getattr(getattr(working, "Resampling", None), "LANCZOS", None)
                    if resampling is None:
                        resampling = 1
                    candidate = working.resize(new_size, resampling)
                    created_candidate = True
                save_image = candidate.convert("RGB") if candidate.mode != "RGB" else candidate
                save_image.save(out_path, format="JPEG", quality=quality, optimize=True)
                if save_image is not candidate:
                    save_image.close()
                if int(out_path.stat().st_size) <= AI_MODEL_MAX_SOURCE_BYTES or scale <= 0.25:
                    break
                scale = max(0.25, scale * 0.85)
                quality = max(72, quality - 5)
            if created_candidate:
                candidate.close()
        yield out_path
    finally:
        temp_dir.cleanup()


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image as _PIL_Image  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(_PIL_Image)
        with _PIL_Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _run_image_analysis(
    *,
    image_path: Path,
    people_image_path: Path | None = None,
    people_matcher: Any,
    object_detector: Any,
    ocr_engine: OCREngine,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    ocr_engine_name: str,
    ocr_language: str,
    people_hint_text: str = "",
    people_source_path: Path | None = None,
    people_bbox_offset: tuple[int, int] = (0, 0),
    caption_source_path: Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    step_fn=None,
    extra_people_names: list[str] | None = None,
    is_page_scan: bool = False,
    ocr_text_override: str | None = None,
    prompt_debug: PromptDebugSession | None = None,
) -> ImageAnalysis:
    del ocr_engine_name
    page_photo_count = 0 if is_page_scan else 1
    people_input_path = people_image_path or image_path
    people_coordinate_path = people_source_path or people_input_path

    with _prepare_ai_model_image(image_path) as model_image_path:
        object_labels: list[str] = []
        ocr_text = str(ocr_text_override or "").strip()
        if ocr_text_override is None and ocr_engine.engine != "none":
            if step_fn:
                step_fn("ocr")
            ocr_text = ocr_engine.read_text(
                model_image_path,
                debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
                debug_step="ocr",
            )
        combined_hint_text = " ".join(part for part in [str(people_hint_text or "").strip(), ocr_text] if part).strip()
        people_matches = (
            _match_people_with_cast_store_retry(
                people_matcher=people_matcher,
                image_path=people_input_path,
                source_path=people_coordinate_path,
                bbox_offset=people_bbox_offset,
                hint_text=combined_hint_text,
            )
            if people_matcher
            else []
        )
        people_match_names = _dedupe([row.name for row in people_matches])
        if step_fn:
            step_fn(_format_people_step_label("people", people_match_names))
        if step_fn:
            step_fn("objects")
        object_matches = object_detector.detect_image(model_image_path) if object_detector else []
        people_names = _dedupe(people_match_names + list(extra_people_names or []))
        object_labels = [row.label for row in object_matches]
        people_positions = _compute_people_positions(people_matches, people_coordinate_path)
        if step_fn:
            step_fn("caption")
        caption_output = caption_engine.generate(
            image_path=model_image_path,
            people=people_names,
            objects=object_labels,
            ocr_text=ocr_text,
            source_path=caption_source_path or people_source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=page_photo_count,
            people_positions=people_positions,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step="caption",
        )
        _faces_detected = (
            (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
            if people_matcher
            else 0
        )

    if ocr_text_override is None and not ocr_text:
        ocr_text = str(getattr(caption_output, "ocr_text", "") or "").strip()
    ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
    subjects = _dedupe(object_labels + ocr_keywords)
    (
        local_people_present,
        local_estimated_people_count,
    ) = _estimate_people_from_detections(
        people_matches=people_matches,
        people_names=people_names,
        object_labels=object_labels,
        faces_detected=_faces_detected,
    )
    people_present, estimated_people_count = _resolve_people_count_metadata(
        requested_caption_engine=requested_caption_engine,
        caption_engine=caption_engine,
        model_image_path=model_image_path,
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
        source_path=caption_source_path or people_source_path or image_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        people_positions=people_positions,
        local_people_present=local_people_present,
        local_estimated_people_count=local_estimated_people_count,
        prompt_debug=prompt_debug,
        debug_step="people_count",
    )

    payload = {
        "people": _serialize_people_matches(people_matches),
        "objects": [{"label": row.label, "score": round(row.score, 5)} for row in object_matches],
        "ocr": {
            "engine": str(caption_output.engine),
            "model": str(caption_engine.effective_model_name),
            "language": str(getattr(caption_output, "ocr_lang", "") or ocr_language),
            "keywords": ocr_keywords,
            "chars": len(ocr_text),
        },
        "caption": _build_caption_metadata(
            requested_engine=requested_caption_engine,
            effective_engine=str(caption_output.engine),
            fallback=bool(caption_output.fallback),
            error=str(caption_output.error or ""),
            engine_error=str(getattr(caption_output, "engine_error", "") or ""),
            model=str(caption_engine.effective_model_name),
            people_present=people_present,
            estimated_people_count=estimated_people_count,
        ),
    }
    if object_detector is not None:
        payload["object_model"] = str(object_detector.model_name)
    gps_latitude, gps_longitude, location_name = _resolve_location_metadata(
        requested_caption_engine=requested_caption_engine,
        caption_engine=caption_engine,
        model_image_path=model_image_path,
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
        source_path=caption_source_path or people_source_path or image_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        people_positions=people_positions,
        fallback_location_name=str(getattr(caption_output, "location_name", "") or "").strip(),
        prompt_debug=prompt_debug,
        debug_step="location",
    )
    location_payload = _resolve_location_payload(
        geocoder=geocoder,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_name=location_name,
    )
    if location_payload:
        payload["location"] = location_payload
    description = caption_output.text
    author_text = str(getattr(caption_output, "author_text", "") or "")
    scene_text = str(getattr(caption_output, "scene_text", "") or "")
    resolved_album_title = _resolve_title_page_album_title(
        image_path=image_path,
        album_title=str(getattr(caption_output, "album_title", "") or ""),
        ocr_text=ocr_text,
    )
    resolved_album_title = _require_album_title_for_title_page(
        image_path=image_path,
        album_title=(resolved_album_title or album_title),
        context="analysis",
    )
    return ImageAnalysis(
        image_path=image_path,
        people_names=people_names,
        object_labels=object_labels,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        subjects=subjects,
        description=description,
        author_text=author_text,
        scene_text=scene_text,
        payload=payload,
        faces_detected=_faces_detected,
        image_regions=list(getattr(caption_output, "image_regions", None) or []),
        album_title=resolved_album_title,
        title=str(getattr(caption_output, "title", "") or ""),
        ocr_lang=str(getattr(caption_output, "ocr_lang", "") or ""),
    )


def _aggregate_best_rows(results: list[ImageAnalysis], section: str, key_name: str) -> list[dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for result in results:
        for row in list(result.payload.get(section) or []):
            name = str(row.get(key_name) or "").strip()
            if not name:
                continue
            score = float(row.get("score") or 0.0)
            current = best_rows.get(name)
            if current is None or score > float(current.get("score") or 0.0):
                best_rows[name] = dict(row)
    out = list(best_rows.values())
    out.sort(
        key=lambda row: (
            -float(row.get("score") or 0.0),
            str(row.get(key_name) or "").casefold(),
        )
    )
    return out


def _layout_payload(layout: PreparedImageLayout) -> dict[str, Any]:
    return {
        "kind": str(layout.kind),
        "page_like": bool(layout.page_like),
        "split_mode": str(layout.split_mode),
        "content_bounds": layout.content_bounds.as_dict(),
        "split_applied": bool(layout.split_applied),
        "fallback_used": bool(layout.fallback_used),
    }


def _bounds_offset(bounds: Any) -> tuple[int, int]:
    if hasattr(bounds, "x") and hasattr(bounds, "y"):
        return int(getattr(bounds, "x")), int(getattr(bounds, "y"))
    if hasattr(bounds, "as_dict"):
        try:
            payload = dict(bounds.as_dict())
        except Exception:
            payload = {}
        return int(payload.get("x", 0) or 0), int(payload.get("y", 0) or 0)
    if isinstance(bounds, dict):
        return int(bounds.get("x", 0) or 0), int(bounds.get("y", 0) or 0)
    return 0, 0


def _build_flat_payload(layout: PreparedImageLayout, analysis: ImageAnalysis) -> dict[str, Any]:
    payload = dict(analysis.payload)
    payload["layout"] = _layout_payload(layout)
    payload["subphotos"] = []
    return payload


def _build_flat_page_description(*, analysis: ImageAnalysis) -> str:
    return analysis.description


def _hash_text(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()


def _scan_page_key(image_path: Path) -> str | None:
    """Return a page-level grouping key for _S# scan files (same P##, different S##).

    Returns None for files that don't match the scan naming pattern.
    """
    match = _scan_name_match(image_path)
    if not match:
        return None
    return (
        f"{match.group('collection')}_{match.group('year')}_B{match.group('book')}_P{match.group('page')}"
    ).casefold()


def _scan_number(image_path: Path) -> int:
    """Return the S## scan number for ordering within a page group."""
    match = _scan_name_match(image_path)
    if not match:
        return 0
    try:
        return int(match.group("scan"))
    except (ValueError, IndexError):
        return 0


def _scan_group_paths(image_path: Path) -> list[Path]:
    page_key = _scan_page_key(image_path)
    if page_key is None:
        return [image_path]
    group_paths = [
        path
        for path in image_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and _scan_page_key(path) == page_key
    ]
    group_paths.sort(key=_scan_number)
    return group_paths or [image_path]


def _scan_group_signature(group_paths: list[Path]) -> str:
    parts: list[str] = []
    for path in group_paths:
        stat = path.stat()
        parts.append(f"{path.name}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}")
    return _hash_text("|".join(parts))


def _resolve_archive_scan_authoritative_ocr(
    *,
    image_path: Path,
    group_paths: list[Path],
    group_signature: str,
    cache: dict[str, ArchiveScanOCRAuthority],
    ocr_engine: OCREngine | None = None,
    step_fn=None,
    stitched_image_dir: Path | None = None,
    debug_recorder=None,
    debug_step: str = "ocr_authority",
) -> ArchiveScanOCRAuthority:
    page_key = _scan_page_key(image_path)
    if page_key is None or len(group_paths) < 2:
        raise RuntimeError(f"Authoritative stitched OCR requires a multi-scan archive page: {image_path}")
    cached = cache.get(page_key)
    if cached is not None and cached.signature == group_signature and (ocr_engine is None or bool(cached.ocr_hash)):
        return cached

    from ..stitch_oversized_pages import (  # pylint: disable=import-outside-toplevel
        build_stitched_image,
        get_view_dirname,
    )

    collection, year, book, page = parse_album_filename(image_path.name)
    view_jpg: Path | None = None
    if collection != "Unknown":
        view_dir = Path(get_view_dirname(image_path.parent))
        candidate = view_dir / f"{collection}_{year}_B{book}_P{int(page):02d}_V.jpg"
        if candidate.is_file():
            view_jpg = candidate

    def _run_authoritative_ocr(source_path: Path) -> tuple[str, tuple[str, ...], str]:
        if ocr_engine is None or ocr_engine.engine == "none":
            return "", (), ""
        if step_fn:
            step_fn("ocr")
        with _prepare_ai_model_image(source_path) as model_image_path:
            ocr_text = ocr_engine.read_text(
                model_image_path,
                debug_recorder=debug_recorder,
                debug_step=debug_step,
            )
        return ocr_text, tuple(extract_keywords(ocr_text, max_keywords=15)), _hash_text(ocr_text)

    stitched_cap_path: Path | None = view_jpg
    ocr_text = ""
    ocr_keywords: tuple[str, ...] = ()
    ocr_hash = ""

    if view_jpg is not None:
        ocr_text, ocr_keywords, ocr_hash = _run_authoritative_ocr(view_jpg)

    if view_jpg is None:
        if step_fn:
            step_fn("stitch")

        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - dependency optional in tests
            raise RuntimeError("opencv-python is required for stitched archive OCR.") from exc

        with tempfile.TemporaryDirectory(prefix="imago-archive-ocr-") as tmp_dir_name:
            stitched = build_stitched_image([str(path) for path in group_paths])
            tmp_path = Path(tmp_dir_name) / f"{group_paths[0].stem}_ocr_stitched.jpg"
            wrote_temp_image = False
            if hasattr(cv2, "imwrite"):
                wrote_temp_image = bool(cv2.imwrite(str(tmp_path), stitched))
            else:
                try:
                    from PIL import Image  # pylint: disable=import-outside-toplevel

                    rgb_image = stitched[:, :, ::-1] if len(stitched.shape) == 3 else stitched
                    Image.fromarray(rgb_image).save(tmp_path, format="JPEG", quality=95)
                    wrote_temp_image = True
                except Exception:
                    wrote_temp_image = False
            if not wrote_temp_image:
                raise RuntimeError(f"Could not write temporary stitched OCR image: {tmp_path}")
            cap_wrote = False
            ocr_source_path = tmp_path
            if stitched_image_dir is not None:
                cap_path = stitched_image_dir / f"{group_paths[0].stem}_stitched.jpg"
                if hasattr(cv2, "imwrite"):
                    cap_wrote = bool(cv2.imwrite(str(cap_path), stitched))
                else:
                    try:
                        from PIL import Image  # pylint: disable=import-outside-toplevel

                        rgb_image = stitched[:, :, ::-1] if len(stitched.shape) == 3 else stitched
                        Image.fromarray(rgb_image).save(cap_path, format="JPEG", quality=95)
                        cap_wrote = True
                    except Exception:
                        pass
            if cap_wrote:
                ocr_source_path = cap_path
            ocr_text, ocr_keywords, ocr_hash = _run_authoritative_ocr(ocr_source_path)
            if cap_wrote:
                stitched_cap_path = cap_path

    result = ArchiveScanOCRAuthority(
        page_key=page_key,
        group_paths=tuple(group_paths),
        signature=group_signature,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        ocr_hash=ocr_hash,
        stitched_image_path=stitched_cap_path,
    )
    cache[page_key] = result
    return result


def _run_scan_stitch_pass(
    files: list[Path],
    *,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    creator_tool: str,
    dry_run: bool,
    stdout_only: bool,
    printed_album_title_cache: dict[str, str],
    geocoder: NominatimGeocoder | None,
) -> int:
    """Group _S# scan files by page, combine OCR text, re-run caption, update XMPs.

    Only files whose names match the _S## scan pattern are considered. Groups with a
    single scan are skipped. OCR text from all scans is joined in scan-number order so
    that text cut off at the right edge of S01 and continued on S02 is reconstructed
    before the caption model sees it.
    """
    # Build page groups from all candidate files
    groups: dict[str, list[Path]] = {}
    for path in files:
        key = _scan_page_key(path)
        if key is not None:
            groups.setdefault(key, []).append(path)

    failures = 0
    for key in sorted(groups):
        group_paths = sorted(groups[key], key=_scan_number)
        if len(group_paths) < 2:
            continue

        # Read XMP state for every scan in the group
        states: list[dict] = []
        for path in group_paths:
            state = read_ai_sidecar_state(path.with_suffix(".xmp"))
            states.append(state if isinstance(state, dict) else {})

        # Skip if every scan already carries the stitch-applied flag
        if all(str(s.get("stitch_key") or "").strip() == "true" for s in states):
            if not stdout_only:
                names_str = " + ".join(p.name for p in group_paths)
                print(f"  stitch skip  {names_str} (already stitched)")
            continue

        # Combine OCR text in scan order
        ocr_parts = [str(s.get("ocr_text") or "").strip() for s in states]
        combined_ocr = " ".join(p for p in ocr_parts if p).strip()

        # Aggregate people and objects across all scans (union, preserving order)
        all_people: list[str] = []
        all_objects: list[str] = []
        for s in states:
            det = s.get("detections") or {}
            if isinstance(det, dict):
                all_people += [
                    str(d.get("name") or "")
                    for d in list(det.get("people") or [])
                    if isinstance(d, dict) and d.get("name")
                ]
                all_objects += [
                    str(d.get("label") or "")
                    for d in list(det.get("objects") or [])
                    if isinstance(d, dict) and d.get("label")
                ]
        person_names = _dedupe(all_people)
        object_labels = _dedupe(all_objects)

        primary_path = group_paths[0]
        primary_state = states[0]
        album_title = str(primary_state.get("album_title") or "").strip() or _resolve_album_title_hint(primary_path)
        printed_album_title = _resolve_album_printed_title_hint(primary_path, printed_album_title_cache)

        names_str = " + ".join(p.name for p in group_paths)
        if not stdout_only:
            print(f"  stitch  {names_str}", end="", flush=True)

        try:
            # Re-run caption with the combined OCR text against the stitched page image when available.
            if requested_caption_engine in {"local", "lmstudio"}:
                caption_source_path = scan_ocr_authority.stitched_image_path or primary_path
                with _prepare_ai_model_image(caption_source_path) as model_image_path:
                    stitch_prompt_debug = PromptDebugSession(caption_source_path)
                    caption_output = caption_engine.generate(
                        image_path=model_image_path,
                        people=person_names,
                        objects=object_labels,
                        ocr_text=combined_ocr,
                        source_path=caption_source_path,
                        album_title=album_title,
                        printed_album_title=printed_album_title,
                        debug_recorder=stitch_prompt_debug.record,
                        debug_step="caption_stitch",
                    )
                    gps_latitude, gps_longitude, location_name = _resolve_location_metadata(
                        requested_caption_engine=requested_caption_engine,
                        caption_engine=caption_engine,
                        model_image_path=model_image_path,
                        people=person_names,
                        objects=object_labels,
                        ocr_text=combined_ocr,
                        source_path=caption_source_path,
                        album_title=album_title,
                        printed_album_title=printed_album_title,
                        people_positions={},
                        fallback_location_name=str(getattr(caption_output, "location_name", "") or "").strip(),
                        prompt_debug=stitch_prompt_debug,
                        debug_step="location_stitch",
                    )
                _emit_prompt_debug_artifact(stitch_prompt_debug, dry_run=dry_run)
                combined_description = caption_output.text
            else:
                combined_description = ""
                gps_latitude = ""
                gps_longitude = ""
                location_name = ""

            location_payload = _resolve_location_payload(
                geocoder=geocoder,
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
                location_name=location_name,
            )
            combined_ocr_keywords = extract_keywords(combined_ocr, max_keywords=15)

            # Write combined caption + combined OCR to every scan's XMP,
            # marking each with stitch_key="true" to record that the pass has run.
            for path, state in zip(group_paths, states):
                sidecar_path = path.with_suffix(".xmp")
                source_text = _build_dc_source(album_title, path, [p.name for p in group_paths])
                det = dict(state.get("detections") or {})
                # Refresh OCR metadata in the detections payload
                if isinstance(det.get("ocr"), dict):
                    det["ocr"] = dict(det["ocr"])
                    det["ocr"]["chars"] = len(combined_ocr)
                    det["ocr"]["keywords"] = combined_ocr_keywords
                subjects = _dedupe(
                    object_labels
                    + [str(k) for k in combined_ocr_keywords if k]
                    + ([album_title] if album_title else [])
                )
                final_gps_lat = str((location_payload or {}).get("gps_latitude") or state.get("gps_latitude") or "")
                final_gps_lon = str((location_payload or {}).get("gps_longitude") or state.get("gps_longitude") or "")
                if stdout_only:
                    print(f"{path.name}: {combined_description}")
                elif not dry_run:
                    stitch_img_w, stitch_img_h = _get_image_dimensions(path)
                    text_layers = _resolve_xmp_text_layers(
                        image_path=path,
                        ocr_text=combined_ocr,
                        page_like=True,
                        author_text=str(getattr(caption_output, "author_text", "") or ""),
                        scene_text=str(getattr(caption_output, "scene_text", "") or ""),
                    )
                    xmp_title, xmp_title_source = _compute_xmp_title(
                        image_path=path,
                        explicit_title="",
                        author_text=str(text_layers.get("author_text") or ""),
                    )
                    write_xmp_sidecar(
                        sidecar_path,
                        creator_tool=str(state.get("creator_tool") or creator_tool),
                        person_names=person_names,
                        subjects=subjects,
                        title=xmp_title,
                        title_source=xmp_title_source,
                        description=combined_description,
                        album_title=album_title,
                        gps_latitude=final_gps_lat,
                        gps_longitude=final_gps_lon,
                        location_city=str((location_payload or {}).get("city") or ""),
                        location_state=str((location_payload or {}).get("state") or ""),
                        location_country=str((location_payload or {}).get("country") or ""),
                        source_text=source_text,
                        ocr_text=combined_ocr,
                        author_text=str(text_layers.get("author_text") or ""),
                        scene_text=str(text_layers.get("scene_text") or ""),
                        detections_payload=det or None,
                        stitch_key="true",
                        create_date=read_embedded_create_date(path),
                        history_when=_xmp_timestamp_from_path(path),
                        image_width=stitch_img_w,
                        image_height=stitch_img_h,
                    )

        except Exception as exc:
            failures += 1
            if not stdout_only:
                print()
            msg = f"  stitch fail  {names_str}: {exc}"
            print(msg, file=sys.stderr if stdout_only else sys.stdout, flush=True)
            continue

        if not stdout_only:
            print(f"\r  stitch ok    {names_str}", flush=True)

    return failures


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
    dc_date: str = "",
    date_time_original: str = "",
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
) -> None:
    """Write XMP sidecar and record the artifact.  Derives history_when and image
    dimensions from image_path; unpacks GPS fields from location_payload."""
    img_w, img_h = _get_image_dimensions(image_path)
    loc = location_payload
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
    )
    append_job_artifact(
        {
            "kind": "photoalbums_xmp",
            "image_path": str(image_path),
            "sidecar_path": str(sidecar_path),
            "label": image_path.name,
        }
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_flags = _explicit_cli_flags(argv)
    requested_caption_prompt = _resolve_caption_prompt(
        str(getattr(args, "caption_prompt", "")),
        str(getattr(args, "caption_prompt_file", "")),
    )
    photos_root = _absolute_cli_path(args.photos_root)
    stdout_only = bool(args.stdout)
    reprocess_mode = str(args.reprocess_mode)
    force_processing = bool(args.force or stdout_only or reprocess_mode == "all")
    dry_run = bool(args.dry_run or stdout_only)

    def emit_info(message: str) -> None:
        if not stdout_only:
            print(message)

    def emit_error(message: str) -> None:
        print(message, file=sys.stderr if stdout_only else sys.stdout, flush=True)

    if not photos_root.is_dir():
        raise SystemExit(f"Photo root is not a directory: {photos_root}")

    include_archive = bool(args.include_archive)
    include_view = bool(args.include_view)
    if not include_archive and not include_view:
        include_archive = True
        include_view = True

    ext_set = {
        (item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}")
        for item in str(args.extensions or "").split(",")
        if item.strip()
    }
    if not ext_set:
        ext_set = set(IMAGE_EXTENSIONS)

    single_photo = str(args.photo or "").strip()
    if single_photo:
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
        photo_offset = int(args.photo_offset or 0)
        if photo_offset > 0:
            files = files[photo_offset:]
        if args.max_images and args.max_images > 0:
            files = files[: int(args.max_images)]

    original_file_count = len(files)
    files = _expand_album_title_dependencies(files, ext_set)

    emit_info(f"Discovered {len(files)} image files")
    if len(files) > original_file_count:
        emit_info(f"Added {len(files) - original_file_count} title-page dependency files")
    if not files:
        return 0

    batch_lock_path: Path | None = None
    if not single_photo:
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

        if (
            existing_sidecar_current
            and existing_sidecar_complete
            and not reprocess_required
            and not source_refresh_required
            and not date_refresh_required
            and not force_processing
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
            force_processing,
            reprocess_required=reprocess_required,
        )

        gps_update_only = False
        if (
            reprocess_mode == "gps"
            and not needs_full
            and existing_sidecar_complete
            and existing_sidecar_state is not None
        ):
            gps_update_only = True

        if not needs_full and not people_update_only and not gps_update_only and not isinstance(existing_sidecar_state, dict):
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
            failures += 1
            emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")
            continue

        if not needs_full and not people_update_only and not gps_update_only:
            state = existing_sidecar_state
            if isinstance(state, dict):
                file_start = time.monotonic()
                prompt_debug = PromptDebugSession(image_path)
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
                            if date_estimation_enabled and not str(review.get("dc_date") or "").strip()
                            else None
                        )
                        refresh_dc_date = _resolve_dc_date(
                            existing_dc_date=str(review.get("dc_date") or ""),
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
                        )

                    if not dry_run:
                        append_job_artifact(
                            {
                                "kind": "photoalbums_xmp",
                                "image_path": str(image_path),
                                "sidecar_path": str(sidecar_path),
                                "label": image_path.name,
                            }
                        )
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
                prefix = f"[{idx}/{len(files)}]{eta_part}  {image_path.name}"
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
                        pu_prompt_debug = PromptDebugSession(image_path)
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
                            if date_estimation_enabled and not str(state.get("dc_date") or "").strip()
                            else None
                        )
                        pu_dc_date = _resolve_dc_date(
                            existing_dc_date=str(state.get("dc_date") or ""),
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
                        )

                    processed += 1
                    completed_times.append(time.monotonic() - file_start)
                    _pu_stop()
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

                if not needs_full:
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
                [str(r.get("name") or "") for r in list(det.get("people") or []) if isinstance(r, dict) and r.get("name")]
            )
            gps_object_labels = [
                str(r.get("label") or "") for r in list(det.get("objects") or []) if isinstance(r, dict) and r.get("label")
            ]
            gps_album_title = str(state.get("album_title") or "").strip()
            gps_printed_title = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
            gps_existing_location_name = str((dict(det.get("location") or {})).get("query") or "").strip()

            eta_str = _format_eta(completed_times, len(files) - idx + 1)
            eta_part = f"  {eta_str}" if eta_str else ""
            prefix = f"[{idx}/{len(files)}]{eta_part}  {image_path.name}"
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

                gps_prompt_debug = PromptDebugSession(image_path)
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
                gps_location_payload = _resolve_location_payload(
                    geocoder=geocoder,
                    gps_latitude=gps_latitude,
                    gps_longitude=gps_longitude,
                    location_name=location_name,
                )
                _emit_prompt_debug_artifact(gps_prompt_debug, dry_run=dry_run)

                if not dry_run:
                    gps_updated_det = {**det}
                    if gps_location_payload:
                        gps_updated_det["location"] = gps_location_payload
                    elif "location" in gps_updated_det:
                        del gps_updated_det["location"]
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
                        create_date=(str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
                        dc_date=str(state.get("dc_date") or ""),
                        date_time_original=str(state.get("date_time_original") or ""),
                        ocr_ran=bool(state.get("ocr_ran")),
                        people_detected=bool(state.get("people_detected")),
                        people_identified=bool(state.get("people_identified")),
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
            prefix = f"[{idx}/{len(files)}]{eta_part}  {image_path.name}"
            print(prefix, flush=True)
            stop_ticker, set_step = _progress_ticker(prefix)
        album_title_hint = _resolve_album_title_hint(image_path)
        printed_album_title_hint = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
        prompt_debug = PromptDebugSession(image_path)

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
                analysis = _run_image_analysis(
                    image_path=analysis_target,
                    people_image_path=(layout.content_path if layout.page_like else image_path),
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine=str(caption_key[0]),
                    ocr_engine_name=ocr_key[0],
                    ocr_language=ocr_key[1],
                    people_source_path=image_path,
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
                    prompt_debug=prompt_debug,
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
                        if date_estimation_enabled
                        and not str((existing_sidecar_state or {}).get("dc_date") or "").strip()
                        else None
                    )
                    final_dc_date = _resolve_dc_date(
                        existing_dc_date=str((existing_sidecar_state or {}).get("dc_date") or ""),
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
