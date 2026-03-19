from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
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
    infer_printed_album_title,
    infer_album_title,
    looks_like_album_cover,
    _normalize_gps_value,
    normalize_lmstudio_base_url,
    normalize_qwen_attn_implementation,
    resolve_caption_model,
)
from .ai_model_settings import default_ocr_model
from .ai_ocr import OCREngine, extract_keywords
from .ai_page_layout import PreparedImageLayout, prepare_image_layout
from .ai_geocode import NominatimGeocoder
from .ai_render_settings import (
    find_archive_dir_for_image,
    load_render_settings,
    resolve_effective_settings,
)
from ..common import PHOTO_ALBUMS_DIR
from ..exiftool_utils import read_tag
from ..naming import DERIVED_NAME_RE, SCAN_TIFF_RE, parse_album_filename, SCAN_NAME_RE
from .xmp_sidecar import (
    _dedupe,
    read_ai_sidecar_state,
    read_person_in_image,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MIN_EXISTING_SIDECAR_BYTES = 100
AI_MODEL_MAX_SOURCE_BYTES = 30 * 1024 * 1024
DEFAULT_CREATOR_TOOL = "imago-photoalbums-ai-index"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "ai_index_manifest.jsonl"
)
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"
PROCESSOR_SIGNATURE = "page_split_v16_archive_stitched_ocr"


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
    faces_detected: int = 0


@dataclass(frozen=True)
class ArchiveScanOCRAuthority:
    page_key: str
    group_paths: tuple[Path, ...]
    signature: str
    ocr_text: str
    ocr_keywords: tuple[str, ...]
    ocr_hash: str


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
    if collection != "Unknown":
        patterns = [
            f"{collection}_{year}_B{book}_P00*.xmp",
            f"{collection}_{year}_B{book}_P01*.xmp",
        ]
    else:
        patterns = ["*_P00*.xmp", "*_P01*.xmp"]
    seen: set[str] = set()
    for folder in _album_directory_candidates(image_path):
        for pattern in patterns:
            for sidecar_path in sorted(folder.glob(pattern)):
                sidecar_key = str(sidecar_path.resolve()).casefold()
                if sidecar_key in seen:
                    continue
                seen.add(sidecar_key)
                yield sidecar_path


def _resolve_album_title_from_sidecars(image_path: Path) -> str:
    for sidecar_path in _iter_album_cover_sidecars(image_path):
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        album_title = str(state.get("album_title") or "").strip()
        if album_title:
            return album_title
        inferred_title = infer_album_title(
            image_path=sidecar_path,
            ocr_text=str(state.get("ocr_text") or ""),
        )
        if inferred_title:
            return inferred_title
    return ""


def _resolve_album_printed_title_from_sidecars(image_path: Path) -> str:
    for sidecar_path in _iter_album_cover_sidecars(image_path):
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        printed_title = infer_printed_album_title(
            ocr_text=str(state.get("ocr_text") or ""),
            fallback_title=str(state.get("album_title") or ""),
        )
        if printed_title:
            return printed_title
    return ""


def _resolve_album_title_hint(
    image_path: Path, album_title_cache: dict[str, str]
) -> str:
    key = _album_identity_key(image_path)
    cached = str(album_title_cache.get(key) or "").strip()
    if cached:
        return cached
    title = _resolve_album_title_from_sidecars(image_path) or infer_album_title(
        image_path=image_path
    )
    if title:
        album_title_cache[key] = title
    return title


def _resolve_album_printed_title_hint(
    image_path: Path, printed_title_cache: dict[str, str]
) -> str:
    key = _album_identity_key(image_path)
    cached = str(printed_title_cache.get(key) or "").strip()
    if cached:
        return cached
    title = _resolve_album_printed_title_from_sidecars(image_path)
    if title:
        printed_title_cache[key] = title
    return title


def _store_album_title_hint(
    image_path: Path, album_title_cache: dict[str, str], title: str
) -> str:
    value = str(title or "").strip()
    if value:
        album_title_cache[_album_identity_key(image_path)] = value
    return value


def _store_album_printed_title_hint(
    image_path: Path, printed_title_cache: dict[str, str], title: str
) -> str:
    value = str(title or "").strip()
    if value:
        printed_title_cache[_album_identity_key(image_path)] = value
    return value


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line or "").strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        image_path = str(row.get("image_path") or "").strip()
        if not image_path:
            continue
        rows[image_path] = row
    return rows


def save_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key in sorted(rows):
        lines.append(json.dumps(rows[key], ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def has_valid_sidecar(path: Path) -> bool:
    sidecar_path = path.with_suffix(".xmp")
    try:
        return (
            sidecar_path.is_file()
            and int(sidecar_path.stat().st_size) > MIN_EXISTING_SIDECAR_BYTES
        )
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


def read_embedded_source_text(path: Path) -> str:
    return str(read_tag(path, "XMP-dc:Source") or "").strip()


def _derived_source_text(image_path: Path) -> str:
    """Return a semicolon-separated dc:source string for a D## derived image.

    Finds the sibling _Archive directory and lists all S## TIF files for the
    same page number, e.g. "China_1986_B02_P17_S01.tif; China_1986_B02_P17_S02.tif".
    Returns "" if the image is not a derived file or no archive scans are found.
    """
    m = DERIVED_NAME_RE.search(image_path.name)
    if not m:
        return ""
    page = str(m.group("page"))
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is None or not archive_dir.is_dir():
        return ""
    page_int = int(page)
    scans: list[Path] = sorted(
        p
        for p in archive_dir.iterdir()
        for sm in (SCAN_TIFF_RE.match(p.name),)
        if sm and int(sm.group("page")) == page_int
    )
    if not scans:
        return ""
    return "; ".join(p.name for p in scans)


def needs_processing(
    path: Path,
    manifest_row: dict[str, Any] | None,
    force: bool,
    *,
    reprocess_required: bool = False,
) -> bool:
    if force:
        return True
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if reprocess_required:
        return True
    sidecar_path = path.with_suffix(".xmp")
    if manifest_row is not None:
        recorded_sidecar = str(manifest_row.get("sidecar_path") or "").strip()
        if recorded_sidecar and recorded_sidecar != str(sidecar_path):
            return True
        if str(manifest_row.get("processor_signature") or "") != PROCESSOR_SIGNATURE:
            return True
        recorded_size = int(manifest_row.get("size", -1))
        recorded_mtime = int(manifest_row.get("mtime_ns", -1))
        if (
            int(stat.st_size) != recorded_size
            or int(stat.st_mtime_ns) != recorded_mtime
        ):
            return True
        return not has_current_sidecar(path)
    if has_current_sidecar(path):
        return False
    if not has_valid_sidecar(path):
        return True
    return int(sidecar_path.stat().st_mtime_ns) < int(stat.st_mtime_ns)


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
            raise SystemExit(
                f"Could not read caption prompt file {path}: {exc}"
            ) from exc
    return str(prompt_text or "").strip()


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index photo album images with cast people matching, YOLO objects, OCR, and XMP sidecars.",
    )
    parser.add_argument(
        "--photos-root",
        default=str(PHOTO_ALBUMS_DIR),
        help="Photo Albums root directory.",
    )
    parser.add_argument(
        "--cast-store", default=str(DEFAULT_CAST_STORE), help="Cast store directory."
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST_PATH), help="JSONL state file path."
    )
    parser.add_argument(
        "--creator-tool", default=DEFAULT_CREATOR_TOOL, help="XMP CreatorTool value."
    )
    parser.add_argument(
        "--model", default="models/yolo11n.pt", help="Ultralytics model path/name."
    )
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
    parser.add_argument(
        "--people-recovery-mode",
        choices=["off", "auto", "always"],
        default="auto",
        help="Optional second people pass after caption using rembg when people may be missing.",
    )
    parser.add_argument(
        "--min-face-size", type=int, default=40, help="Minimum face size in pixels."
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "qwen", "lmstudio"],
        default="lmstudio",
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
        choices=["none", "qwen", "lmstudio"],
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
        default="http://192.168.4.72:1234/v1",
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
        help="Sampling temperature for qwen.",
    )
    parser.add_argument(
        "--caption-max-edge",
        type=int,
        default=0,
        help="Optional long-edge cap, in pixels, applied only during caption generation.",
    )
    parser.add_argument(
        "--qwen-attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention implementation for Qwen captioning. flash_attention_2 is only useful on compatible GPUs.",
    )
    parser.add_argument(
        "--qwen-min-pixels",
        type=int,
        default=0,
        help="Optional Qwen processor min_pixels value. Use 0 to keep the model default.",
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=0,
        help="Optional Qwen processor max_pixels value. Use 0 to keep the model default.",
    )
    parser.add_argument(
        "--max-images", type=int, default=0, help="Optional processing limit."
    )
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
        "--force", action="store_true", help="Ignore manifest and process all files."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write sidecar/manifest."
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated caption text to stdout only. Implies --dry-run and forced reprocessing.",
    )
    parser.add_argument(
        "--include-view", action="store_true", help="Include files in *_View folders."
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include files in *_Archive folders.",
    )
    parser.add_argument(
        "--disable-people", action="store_true", help="Disable cast people matching."
    )
    parser.add_argument(
        "--disable-objects", action="store_true", help="Disable object detection."
    )
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
            "Deprecated. Multi-scan archive page OCR now uses a temporary stitched composite "
            "during normal processing."
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
    qwen_attn_implementation: str,
    qwen_min_pixels: int,
    qwen_max_pixels: int,
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
        qwen_attn_implementation=str(qwen_attn_implementation),
        qwen_min_pixels=int(qwen_min_pixels),
        qwen_max_pixels=int(qwen_max_pixels),
        lmstudio_base_url=str(lmstudio_base_url),
        max_image_edge=int(max_image_edge),
        stream=stream,
    )


def _settings_signature(settings: dict[str, Any]) -> str:
    caption_engine = str(settings.get("caption_engine", "qwen"))
    caption_model = resolve_caption_model(
        caption_engine,
        str(settings.get("caption_model", "")),
    )
    compact = {
        "processor_signature": PROCESSOR_SIGNATURE,
        "skip": bool(settings.get("skip", False)),
        "enable_people": bool(settings.get("enable_people", True)),
        "enable_objects": bool(settings.get("enable_objects", True)),
        "people_recovery_mode": str(settings.get("people_recovery_mode", "off")),
        "ocr_engine": str(settings.get("ocr_engine", "none")),
        "ocr_lang": str(settings.get("ocr_lang", "eng")),
        "ocr_model": str(settings.get("ocr_model", "")),
        "page_split_mode": str(settings.get("page_split_mode", "off")),
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
            str(settings.get("lmstudio_base_url", "http://192.168.4.72:1234/v1"))
        ),
        "qwen_attn_implementation": normalize_qwen_attn_implementation(
            str(settings.get("qwen_attn_implementation", "auto"))
        ),
        "qwen_min_pixels": int(settings.get("qwen_min_pixels", 0)),
        "qwen_max_pixels": int(settings.get("qwen_max_pixels", 0)),
    }
    return json.dumps(compact, sort_keys=True, ensure_ascii=True)


def _build_caption_metadata(
    *,
    requested_engine: str,
    effective_engine: str,
    fallback: bool,
    error: str,
    model: str,
    people_present: bool = False,
    estimated_people_count: int = 0,
) -> dict[str, Any]:
    return {
        "requested_engine": str(requested_engine),
        "effective_engine": str(effective_engine),
        "fallback": bool(fallback),
        "error": str(error or "")[:500],
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
    object_person_count = sum(
        1
        for label in list(object_labels or [])
        if str(label).strip().casefold() == "person"
    )
    estimated_people_count = max(
        0,
        int(faces_detected or 0),
        len(list(people_matches or [])),
        len(list(people_names or [])),
        int(object_person_count),
    )
    return estimated_people_count > 0, estimated_people_count


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
    people_present = bool(
        local_people_present or model_people_present or estimated_people_count > 0
    )
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
    is_cover_page: bool,
    people_positions: dict[str, str],
    fallback_location_name: str,
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
            is_cover_page=is_cover_page,
            people_positions=people_positions,
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
            str(getattr(result, "location_name", "") or "").strip()
            or str(fallback_location_name or "").strip()
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
            **(
                {"bbox": [int(v) for v in row.bbox[:4]]}
                if getattr(row, "bbox", None)
                else {}
            ),
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
            row_certainty = float(
                getattr(row, "certainty", getattr(row, "score", 0.0)) or 0.0
            )
            current_certainty = float(
                getattr(current, "certainty", getattr(current, "score", 0.0)) or 0.0
            )
            row_score = float(getattr(row, "score", 0.0) or 0.0)
            current_score = float(getattr(current, "score", 0.0) or 0.0)
            if row_certainty > current_certainty or (
                row_certainty == current_certainty and row_score > current_score
            ):
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


def _should_run_people_recovery(
    *,
    people_recovery_mode: str,
    faces_detected: int,
    people_matches: list,
    people_names: list[str],
    object_labels: list[str],
) -> bool:
    mode = str(people_recovery_mode or "off").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    if mode != "auto":
        return False
    people_present, estimated_people_count = _estimate_people_from_detections(
        people_matches=people_matches,
        people_names=people_names,
        object_labels=object_labels,
        faces_detected=faces_detected,
    )
    if people_present and int(faces_detected) <= 0:
        return True
    return estimated_people_count > int(faces_detected)


def _maybe_run_people_recovery(
    *,
    people_matcher: Any,
    people_recovery_mode: str,
    image_path: Path,
    people_source_path: Path,
    people_bbox_offset: tuple[int, int],
    people_hint_text: str,
    extra_people_names: list[str],
    people_matches: list,
    people_names: list[str],
    object_labels: list[str],
    ocr_text: str,
    caption_output: CaptionOutput,
    caption_engine: CaptionEngine,
    model_image_path: Path,
    caption_source_path: Path,
    album_title: str,
    printed_album_title: str,
    photo_count: int,
    is_cover_page: bool,
    step_fn=None,
) -> tuple[list, list[str], dict[str, str], CaptionOutput, int]:
    faces_detected = (
        (
            _v
            if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int)
            else 0
        )
        if people_matcher
        else 0
    )
    people_positions = _compute_people_positions(people_matches, image_path)
    if not people_matcher or not _should_run_people_recovery(
        people_recovery_mode=people_recovery_mode,
        faces_detected=faces_detected,
        people_matches=people_matches,
        people_names=people_names,
        object_labels=object_labels,
    ):
        return (
            people_matches,
            people_names,
            people_positions,
            caption_output,
            faces_detected,
        )

    recovery_hint_text = " ".join(
        part
        for part in [str(people_hint_text or "").strip(), str(ocr_text or "").strip()]
        if part
    ).strip()
    if step_fn:
        step_fn("people-recovery")
    recovered_matches = people_matcher.match_image_recovery(
        image_path,
        source_path=people_source_path,
        bbox_offset=people_bbox_offset,
        hint_text=recovery_hint_text,
    )
    recovered_faces_detected = (
        (
            _v
            if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int)
            else 0
        )
        if people_matcher
        else 0
    )
    merged_matches = _merge_people_matches(people_matches, recovered_matches)
    recovered_names = _dedupe(
        [row.name for row in merged_matches] + list(extra_people_names or [])
    )
    recovered_positions = _compute_people_positions(merged_matches, image_path)
    if list(recovered_names) == list(people_names):
        return (
            merged_matches,
            recovered_names,
            recovered_positions,
            caption_output,
            recovered_faces_detected,
        )
    if step_fn:
        step_fn("caption")
    recovered_caption = caption_engine.generate(
        image_path=model_image_path,
        people=recovered_names,
        objects=object_labels,
        ocr_text=ocr_text,
        source_path=caption_source_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        photo_count=photo_count,
        is_cover_page=is_cover_page,
        people_positions=recovered_positions,
    )
    return (
        merged_matches,
        recovered_names,
        recovered_positions,
        recovered_caption,
        recovered_faces_detected,
    )


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
            return {
                "query": result.query,
                "display_name": result.display_name,
                "gps_latitude": float(result.latitude),
                "gps_longitude": float(result.longitude),
                "map_datum": "WGS-84",
                "source": result.source,
            }
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
                    ((AI_MODEL_MAX_SOURCE_BYTES / float(max(1, source_size))) ** 0.5)
                    * 0.92,
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
                    resampling = getattr(
                        getattr(working, "Resampling", None), "LANCZOS", None
                    )
                    if resampling is None:
                        resampling = 1
                    candidate = working.resize(new_size, resampling)
                    created_candidate = True
                save_image = (
                    candidate.convert("RGB") if candidate.mode != "RGB" else candidate
                )
                save_image.save(out_path, format="JPEG", quality=quality, optimize=True)
                if save_image is not candidate:
                    save_image.close()
                if (
                    int(out_path.stat().st_size) <= AI_MODEL_MAX_SOURCE_BYTES
                    or scale <= 0.25
                ):
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

        with _PIL_Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _run_image_analysis(
    *,
    image_path: Path,
    people_matcher: Any,
    object_detector: Any,
    ocr_engine: OCREngine,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    requested_caption_model: str,
    ocr_engine_name: str,
    ocr_language: str,
    people_hint_text: str = "",
    people_source_path: Path | None = None,
    people_bbox_offset: tuple[int, int] = (0, 0),
    people_recovery_mode: str = "off",
    caption_source_path: Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    step_fn=None,
    extra_people_names: list[str] | None = None,
    is_page_scan: bool = False,
    ocr_text_override: str | None = None,
) -> ImageAnalysis:
    use_combined = (
        ocr_text_override is None
        and ocr_engine.engine == "qwen"
        and caption_engine.engine == "qwen"
    )
    page_photo_count = 0 if is_page_scan else 1

    with _prepare_ai_model_image(image_path) as model_image_path:
        object_labels: list[str] = []
        is_cover_page = False
        if use_combined:
            if step_fn:
                step_fn("people")
            people_matches = (
                people_matcher.match_image(
                    image_path,
                    source_path=people_source_path or image_path,
                    bbox_offset=people_bbox_offset,
                    hint_text=str(people_hint_text or "").strip(),
                )
                if people_matcher
                else []
            )
            if step_fn:
                step_fn("objects")
            object_matches = (
                object_detector.detect_image(model_image_path)
                if object_detector
                else []
            )
            people_names = _dedupe(
                [row.name for row in people_matches] + list(extra_people_names or [])
            )
            object_labels = [row.label for row in object_matches]
            people_positions = _compute_people_positions(people_matches, image_path)
            if step_fn:
                step_fn("ocr+caption")
            caption_output, ocr_text = caption_engine.generate_combined(
                image_path=model_image_path,
                people=people_names,
                objects=object_labels,
                source_path=caption_source_path or people_source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=page_photo_count,
                people_positions=people_positions,
            )
            ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
            (
                people_matches,
                people_names,
                people_positions,
                caption_output,
                _faces_detected,
            ) = _maybe_run_people_recovery(
                people_matcher=people_matcher,
                people_recovery_mode=people_recovery_mode,
                image_path=image_path,
                people_source_path=people_source_path or image_path,
                people_bbox_offset=people_bbox_offset,
                people_hint_text=str(people_hint_text or "").strip(),
                extra_people_names=list(extra_people_names or []),
                people_matches=people_matches,
                people_names=people_names,
                object_labels=object_labels,
                ocr_text=ocr_text,
                caption_output=caption_output,
                caption_engine=caption_engine,
                model_image_path=model_image_path,
                caption_source_path=caption_source_path
                or people_source_path
                or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=page_photo_count,
                is_cover_page=False,
                step_fn=step_fn,
            )
        else:
            if step_fn and ocr_text_override is None:
                step_fn("ocr")
            if ocr_text_override is None:
                ocr_text = ocr_engine.read_text(model_image_path)
            else:
                ocr_text = str(ocr_text_override or "").strip()
            ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
            combined_hint_text = " ".join(
                part for part in [str(people_hint_text or "").strip(), ocr_text] if part
            ).strip()
            if step_fn:
                step_fn("people")
            people_matches = (
                people_matcher.match_image(
                    image_path,
                    source_path=people_source_path or image_path,
                    bbox_offset=people_bbox_offset,
                    hint_text=combined_hint_text,
                )
                if people_matcher
                else []
            )
            if step_fn:
                step_fn("objects")
            object_matches = (
                object_detector.detect_image(model_image_path)
                if object_detector
                else []
            )
            people_names = _dedupe(
                [row.name for row in people_matches] + list(extra_people_names or [])
            )
            object_labels = [row.label for row in object_matches]
            people_positions = _compute_people_positions(people_matches, image_path)
            if step_fn:
                step_fn("caption")
            is_cover_page = is_page_scan and looks_like_album_cover(
                image_path, ocr_text=ocr_text
            )
            caption_output = caption_engine.generate(
                image_path=model_image_path,
                people=people_names,
                objects=object_labels,
                ocr_text=ocr_text,
                source_path=caption_source_path or people_source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=page_photo_count,
                is_cover_page=is_cover_page,
                people_positions=people_positions,
            )
            (
                people_matches,
                people_names,
                people_positions,
                caption_output,
                _faces_detected,
            ) = _maybe_run_people_recovery(
                people_matcher=people_matcher,
                people_recovery_mode=people_recovery_mode,
                image_path=image_path,
                people_source_path=people_source_path or image_path,
                people_bbox_offset=people_bbox_offset,
                people_hint_text=str(people_hint_text or "").strip(),
                extra_people_names=list(extra_people_names or []),
                people_matches=people_matches,
                people_names=people_names,
                object_labels=object_labels,
                ocr_text=ocr_text,
                caption_output=caption_output,
                caption_engine=caption_engine,
                model_image_path=model_image_path,
                caption_source_path=caption_source_path
                or people_source_path
                or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=page_photo_count,
                is_cover_page=is_cover_page,
                step_fn=step_fn,
            )

    if "_faces_detected" not in locals():
        _faces_detected = (
            (
                _v
                if isinstance(
                    _v := getattr(people_matcher, "last_faces_detected", 0), int
                )
                else 0
            )
            if people_matcher
            else 0
        )

    subjects = _dedupe(object_labels + ocr_keywords)
    description = caption_output.text
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
    )

    payload = {
        "people": _serialize_people_matches(people_matches),
        "objects": [
            {"label": row.label, "score": round(row.score, 5)} for row in object_matches
        ],
        "ocr": {
            "engine": str(ocr_engine_name),
            "model": str(ocr_engine.effective_model_name),
            "language": str(ocr_language),
            "keywords": ocr_keywords,
            "chars": len(ocr_text),
        },
        "caption": _build_caption_metadata(
            requested_engine=requested_caption_engine,
            effective_engine=str(caption_output.engine),
            fallback=bool(caption_output.fallback),
            error=str(caption_output.error or ""),
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
        is_cover_page=is_cover_page,
        people_positions=people_positions,
        fallback_location_name=str(
            getattr(caption_output, "location_name", "") or ""
        ).strip(),
    )
    location_payload = _resolve_location_payload(
        geocoder=geocoder,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_name=location_name,
    )
    if location_payload:
        payload["location"] = location_payload
    return ImageAnalysis(
        image_path=image_path,
        people_names=people_names,
        object_labels=object_labels,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        subjects=subjects,
        description=description,
        payload=payload,
        faces_detected=_faces_detected,
    )


def _aggregate_best_rows(
    results: list[ImageAnalysis], section: str, key_name: str
) -> list[dict[str, Any]]:
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


def _build_flat_payload(
    layout: PreparedImageLayout, analysis: ImageAnalysis
) -> dict[str, Any]:
    payload = dict(analysis.payload)
    payload["layout"] = _layout_payload(layout)
    payload["subphotos"] = []
    return payload


def _build_page_payload(
    *,
    layout: PreparedImageLayout,
    sub_results: list[ImageAnalysis],
    page_ocr_text: str,
    page_ocr_keywords: list[str],
    requested_caption_engine: str,
    album_title: str = "",
    printed_album_title: str = "",
) -> tuple[list[str], list[str], list[str], str, dict[str, Any], list[dict[str, Any]]]:
    aggregate_people = _aggregate_best_rows(sub_results, "people", "name")
    aggregate_objects = _aggregate_best_rows(sub_results, "objects", "label")
    people_names = [str(row["name"]) for row in aggregate_people]
    object_labels = [str(row["label"]) for row in aggregate_objects]
    subphoto_rows: list[dict[str, Any]] = []
    page_subjects: list[str] = list(page_ocr_keywords)
    for prepared, result in zip(layout.subphotos, sub_results):
        page_subjects.extend(result.subjects)
        subphoto_rows.append(
            {
                "index": int(prepared.index),
                "bounds": prepared.bounds.as_dict(),
                "description": result.description,
                "ocr_text": result.ocr_text,
                "people": result.people_names,
                "subjects": result.subjects,
                "detections": result.payload,
            }
        )

    subjects = _dedupe(page_subjects)
    if len(sub_results) == 1:
        description = sub_results[0].description
    else:
        description = ""
    page_people_present = any(
        bool(dict(result.payload.get("caption") or {}).get("people_present"))
        for result in sub_results
    )
    page_estimated_people_count = max(
        len(people_names),
        sum(
            max(
                0,
                int(
                    dict(result.payload.get("caption") or {}).get(
                        "estimated_people_count",
                        0,
                    )
                    or 0
                ),
            )
            for result in sub_results
        ),
    )
    payload = {
        "layout": _layout_payload(layout),
        "people": aggregate_people,
        "objects": aggregate_objects,
        "ocr": {
            "engine": (
                str(sub_results[0].payload["ocr"]["engine"]) if sub_results else ""
            ),
            "language": (
                str(sub_results[0].payload["ocr"]["language"]) if sub_results else ""
            ),
            "keywords": page_ocr_keywords,
            "chars": len(page_ocr_text),
        },
        "caption": _build_caption_metadata(
            requested_engine=requested_caption_engine,
            effective_engine="page-summary",
            fallback=False,
            error="",
            model="",
            people_present=page_people_present,
            estimated_people_count=page_estimated_people_count,
        ),
        "subphotos": subphoto_rows,
    }
    return people_names, object_labels, subjects, description, payload, subphoto_rows


def _build_flat_page_description(*, analysis: ImageAnalysis) -> str:
    return analysis.description


def _hash_text(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()


def _scan_page_key(image_path: Path) -> str | None:
    """Return a page-level grouping key for _S# scan files (same P##, different S##).

    Returns None for files that don't match the scan naming pattern.
    """
    match = SCAN_NAME_RE.search(image_path.name)
    if not match:
        return None
    return (
        f"{match.group('collection')}_{match.group('year')}"
        f"_B{match.group('book')}_P{match.group('page')}"
    ).casefold()


def _scan_number(image_path: Path) -> int:
    """Return the S## scan number for ordering within a page group."""
    match = SCAN_NAME_RE.search(image_path.name)
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
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and _scan_page_key(path) == page_key
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
    ocr_engine: OCREngine,
    cache: dict[str, ArchiveScanOCRAuthority],
    step_fn=None,
) -> ArchiveScanOCRAuthority:
    page_key = _scan_page_key(image_path)
    if page_key is None or len(group_paths) < 2:
        raise RuntimeError(
            f"Authoritative stitched OCR requires a multi-scan archive page: {image_path}"
        )
    cached = cache.get(page_key)
    if cached is not None and cached.signature == group_signature:
        return cached

    if step_fn:
        step_fn("stitch")
    from ..stitch_oversized_pages import (  # pylint: disable=import-outside-toplevel
        build_stitched_image,
    )

    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - dependency optional in tests
        raise RuntimeError(
            "opencv-python is required for stitched archive OCR."
        ) from exc

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
                Image.fromarray(rgb_image).save(
                    tmp_path, format="JPEG", quality=95
                )
                wrote_temp_image = True
            except Exception:
                wrote_temp_image = False
        if not wrote_temp_image:
            raise RuntimeError(
                f"Could not write temporary stitched OCR image: {tmp_path}"
            )
        if step_fn:
            step_fn("ocr")
        with _prepare_ai_model_image(tmp_path) as model_image_path:
            ocr_text = ocr_engine.read_text(model_image_path)

    result = ArchiveScanOCRAuthority(
        page_key=page_key,
        group_paths=tuple(group_paths),
        signature=group_signature,
        ocr_text=ocr_text,
        ocr_keywords=tuple(extract_keywords(ocr_text, max_keywords=15)),
        ocr_hash=_hash_text(ocr_text),
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
    album_title_cache: dict[str, str],
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
        album_title = str(
            primary_state.get("album_title") or ""
        ).strip() or _resolve_album_title_hint(primary_path, album_title_cache)
        printed_album_title = _resolve_album_printed_title_hint(
            primary_path, printed_album_title_cache
        )

        names_str = " + ".join(p.name for p in group_paths)
        if not stdout_only:
            print(f"  stitch  {names_str}", end="", flush=True)

        try:
            # Re-run caption with the combined OCR text; use S01 image as representative
            if requested_caption_engine in {"qwen", "lmstudio"}:
                with _prepare_ai_model_image(primary_path) as model_image_path:
                    caption_output = caption_engine.generate(
                        image_path=model_image_path,
                        people=person_names,
                        objects=object_labels,
                        ocr_text=combined_ocr,
                        source_path=primary_path,
                        album_title=album_title,
                        printed_album_title=printed_album_title,
                    )
                    gps_latitude, gps_longitude, location_name = (
                        _resolve_location_metadata(
                            requested_caption_engine=requested_caption_engine,
                            caption_engine=caption_engine,
                            model_image_path=model_image_path,
                            people=person_names,
                            objects=object_labels,
                            ocr_text=combined_ocr,
                            source_path=primary_path,
                            album_title=album_title,
                            printed_album_title=printed_album_title,
                            is_cover_page=False,
                            people_positions={},
                            fallback_location_name=str(
                                getattr(caption_output, "location_name", "") or ""
                            ).strip(),
                        )
                    )
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
                source_text = read_embedded_source_text(path)
                det = dict(state.get("detections") or {})
                # Refresh OCR metadata in the detections payload
                if isinstance(det.get("ocr"), dict):
                    det["ocr"] = dict(det["ocr"])
                    det["ocr"]["chars"] = len(combined_ocr)
                    det["ocr"]["keywords"] = combined_ocr_keywords
                subjects = _dedupe(
                    object_labels + [str(k) for k in combined_ocr_keywords if k]
                )
                final_gps_lat = str(
                    (location_payload or {}).get("gps_latitude")
                    or state.get("gps_latitude")
                    or ""
                )
                final_gps_lon = str(
                    (location_payload or {}).get("gps_longitude")
                    or state.get("gps_longitude")
                    or ""
                )
                if stdout_only:
                    print(f"{path.name}: {combined_description}")
                elif not dry_run:
                    stitch_img_w, stitch_img_h = _get_image_dimensions(path)
                    write_xmp_sidecar(
                        sidecar_path,
                        creator_tool=str(state.get("creator_tool") or creator_tool),
                        person_names=person_names,
                        subjects=subjects,
                        description=combined_description,
                        album_title=album_title,
                        gps_latitude=final_gps_lat,
                        gps_longitude=final_gps_lon,
                        source_text=source_text,
                        ocr_text=combined_ocr,
                        detections_payload=det or None,
                        stitch_key="true",
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


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_flags = _explicit_cli_flags(argv)
    requested_caption_prompt = _resolve_caption_prompt(
        str(getattr(args, "caption_prompt", "")),
        str(getattr(args, "caption_prompt_file", "")),
    )
    photos_root = Path(args.photos_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    stdout_only = bool(args.stdout)
    force_processing = bool(args.force or stdout_only)
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
        (
            item.strip().lower()
            if item.strip().startswith(".")
            else f".{item.strip().lower()}"
        )
        for item in str(args.extensions or "").split(",")
        if item.strip()
    }
    if not ext_set:
        ext_set = set(IMAGE_EXTENSIONS)

    single_photo = str(args.photo or "").strip()
    if single_photo:
        photo_path = Path(single_photo).expanduser().resolve()
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

    emit_info(f"Discovered {len(files)} image files")
    if not files:
        return 0

    default_caption_max_tokens = int(args.caption_max_tokens)
    if (
        "--caption-max-tokens" not in explicit_flags
        and str(args.caption_engine) == "lmstudio"
    ):
        default_caption_max_tokens = max(
            default_caption_max_tokens, int(DEFAULT_LMSTUDIO_MAX_NEW_TOKENS)
        )

    defaults = {
        "skip": False,
        "enable_people": not bool(args.disable_people),
        "enable_objects": not bool(args.disable_objects),
        "people_recovery_mode": str(args.people_recovery_mode),
        "ocr_engine": str(args.ocr_engine),
        "ocr_lang": str(args.ocr_lang),
        "ocr_model": str(args.ocr_model),
        "page_split_mode": "off",
        "caption_engine": str(args.caption_engine),
        "caption_model": resolve_caption_model(
            str(args.caption_engine), str(args.caption_model)
        ),
        "caption_prompt": str(requested_caption_prompt),
        "caption_max_tokens": int(default_caption_max_tokens),
        "caption_temperature": float(args.caption_temperature),
        "caption_max_edge": int(args.caption_max_edge),
        "lmstudio_base_url": normalize_lmstudio_base_url(str(args.lmstudio_base_url)),
        "qwen_attn_implementation": normalize_qwen_attn_implementation(
            str(args.qwen_attn_implementation)
        ),
        "qwen_min_pixels": int(args.qwen_min_pixels),
        "qwen_max_pixels": int(args.qwen_max_pixels),
        "people_threshold": float(args.people_threshold),
        "object_threshold": float(args.object_threshold),
        "min_face_size": int(args.min_face_size),
        "model": str(args.model),
        "creator_tool": str(args.creator_tool),
    }

    manifest = load_manifest(manifest_path)
    archive_settings_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
    people_matcher_cache: dict[tuple[str, float, int], Any] = {}
    object_detector_cache: dict[tuple[str, float], Any] = {}
    ocr_engine_cache: dict[tuple[str, str, str, str], OCREngine] = {}
    caption_engine_cache: dict[
        tuple[str, str, str, int, float, str, str, int, int, int], CaptionEngine
    ] = {}
    archive_scan_ocr_cache: dict[str, ArchiveScanOCRAuthority] = {}
    album_title_cache: dict[str, str] = {}
    printed_album_title_cache: dict[str, str] = {}
    geocoder = NominatimGeocoder()

    processed = 0
    skipped = 0
    failures = 0
    processed_cast_manifest_keys: set[str] = set()
    completed_times: list[float] = []

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
            or "--qwen-prompt" in explicit_flags
            or "--caption-prompt-file" in explicit_flags
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
            effective["lmstudio_base_url"] = normalize_lmstudio_base_url(
                str(args.lmstudio_base_url)
            )
        if "--qwen-attn-implementation" in explicit_flags:
            effective["qwen_attn_implementation"] = normalize_qwen_attn_implementation(
                str(args.qwen_attn_implementation)
            )
        if "--qwen-min-pixels" in explicit_flags:
            effective["qwen_min_pixels"] = int(args.qwen_min_pixels)
        if "--qwen-max-pixels" in explicit_flags:
            effective["qwen_max_pixels"] = int(args.qwen_max_pixels)
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", defaults["caption_engine"])),
            str(effective.get("caption_model", defaults["caption_model"])),
        )
        settings_sig = _settings_signature(effective)
        creator_tool = str(effective.get("creator_tool", args.creator_tool))

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

        existing_sidecar_state: dict | None = None
        if has_valid_sidecar(image_path):
            existing_sidecar_state = read_ai_sidecar_state(sidecar_path)

        existing_sidecar_ocr_hash = _hash_text(
            str((existing_sidecar_state or {}).get("ocr_text") or "")
        )
        multi_scan_group_paths = _scan_group_paths(image_path)
        archive_stitched_ocr_required = (
            str(effective.get("ocr_engine", defaults["ocr_engine"])).strip().lower()
            != "none"
            and len(multi_scan_group_paths) > 1
        )
        multi_scan_group_signature = (
            _scan_group_signature(multi_scan_group_paths)
            if archive_stitched_ocr_required
            else ""
        )

        manifest_row = manifest.get(str(image_path))
        reprocess_required = False
        people_update_only = False
        if _sidecar_has_lmstudio_caption_error(existing_sidecar_state):
            reprocess_required = True
        if has_valid_sidecar(image_path) and not sidecar_has_expected_ai_fields(
            sidecar_path,
            creator_tool=creator_tool,
            enable_people=bool(effective.get("enable_people", True)),
            enable_objects=bool(effective.get("enable_objects", True)),
            ocr_engine=str(effective.get("ocr_engine", defaults["ocr_engine"])),
            caption_engine=str(
                effective.get("caption_engine", defaults["caption_engine"])
            ),
        ):
            reprocess_required = True
        if archive_stitched_ocr_required:
            manifest_source = str(
                (manifest_row or {}).get("ocr_authority_source") or ""
            ).strip()
            manifest_signature = str(
                (manifest_row or {}).get("ocr_authority_signature") or ""
            ).strip()
            manifest_hash = str(
                (manifest_row or {}).get("ocr_authority_hash") or ""
            ).strip()
            if (
                manifest_source != "archive_stitched"
                or manifest_signature != multi_scan_group_signature
                or not manifest_hash
                or manifest_hash != existing_sidecar_ocr_hash
            ):
                reprocess_required = True
        if manifest_row is not None:
            old_sig = str(manifest_row.get("settings_signature") or "")
            if old_sig != settings_sig:
                reprocess_required = True
            elif bool(effective.get("enable_people", True)):
                if (
                    str(manifest_row.get("cast_store_signature") or "")
                    != current_cast_signature
                ):
                    # Cast changed: only re-run people+caption if faces were detected here.
                    # None means old XMP without the flag — treat conservatively as True.
                    _pd = (existing_sidecar_state or {}).get("people_detected")
                    if _pd is True or _pd is None:
                        people_update_only = True
                    # _pd is False → no faces in this image, cast change is irrelevant

        needs_full = needs_processing(
            image_path,
            manifest_row,
            force_processing,
            reprocess_required=reprocess_required,
        )

        if not needs_full and not people_update_only:
            skipped += 1
            if args.verbose and not stdout_only:
                print(f"[{idx}/{len(files)}] skip  {image_path.name}")
            continue

        if bool(effective.get("skip", False)):
            skipped += 1
            if args.verbose and not stdout_only:
                print(
                    f"[{idx}/{len(files)}] skip  {image_path.name} (render_settings skip=true)"
                )
            continue

        # ── Fast path: cast changed but only people+caption need updating ──────
        if not needs_full and people_update_only and not stdout_only:
            state = existing_sidecar_state
            if not isinstance(state, dict):
                needs_full = True  # fall through to full processing
            else:
                file_start = time.monotonic()
                det = state.get("detections") or {}
                existing_ocr_text = str(state.get("ocr_text") or "").strip()
                existing_ocr_keywords = list(
                    (det.get("ocr") or {}).get("keywords") or []
                )
                existing_object_rows = [
                    r for r in list(det.get("objects") or []) if isinstance(r, dict)
                ]
                existing_object_labels = [
                    str(r.get("label") or "")
                    for r in existing_object_rows
                    if r.get("label")
                ]
                existing_location = dict(det.get("location") or {})

                eta_str = _format_eta(completed_times, len(files) - idx + 1)
                eta_part = f"  {eta_str}" if eta_str else ""
                prefix = f"[{idx}/{len(files)}]{eta_part}  {image_path.name}"
                print(prefix, flush=True)
                _pu_stop, _pu_step = _progress_ticker(prefix)

                try:
                    caption_key = (
                        str(
                            effective.get("caption_engine", defaults["caption_engine"])
                        ),
                        str(effective.get("caption_model", defaults["caption_model"])),
                        str(
                            effective.get("caption_prompt", defaults["caption_prompt"])
                        ),
                        int(
                            effective.get(
                                "caption_max_tokens", defaults["caption_max_tokens"]
                            )
                        ),
                        float(
                            effective.get(
                                "caption_temperature", defaults["caption_temperature"]
                            )
                        ),
                        str(
                            effective.get(
                                "lmstudio_base_url", defaults["lmstudio_base_url"]
                            )
                        ),
                        str(
                            effective.get(
                                "qwen_attn_implementation",
                                defaults["qwen_attn_implementation"],
                            )
                        ),
                        int(
                            effective.get(
                                "qwen_min_pixels", defaults["qwen_min_pixels"]
                            )
                        ),
                        int(
                            effective.get(
                                "qwen_max_pixels", defaults["qwen_max_pixels"]
                            )
                        ),
                        int(
                            effective.get(
                                "caption_max_edge", defaults["caption_max_edge"]
                            )
                        ),
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
                            qwen_attn_implementation=caption_key[6],
                            qwen_min_pixels=int(caption_key[7]),
                            qwen_max_pixels=int(caption_key[8]),
                            max_image_edge=int(caption_key[9]),
                            stream=True,
                        )
                        caption_engine_cache[caption_key] = pu_caption_engine

                    _pu_step("people")
                    pu_people_matches = (
                        people_matcher.match_image(
                            image_path,
                            source_path=image_path,
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
                    pu_person_names = _dedupe(
                        [r.name for r in pu_people_matches] + existing_xmp_people
                    )
                    pu_people_positions = _compute_people_positions(
                        pu_people_matches, image_path
                    )

                    _pu_step("caption")
                    pu_album_title = _resolve_album_title_hint(
                        image_path, album_title_cache
                    )
                    pu_printed_title = _resolve_album_printed_title_hint(
                        image_path, printed_album_title_cache
                    )
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
                        )
                        (
                            pu_people_matches,
                            pu_person_names,
                            pu_people_positions,
                            pu_caption_out,
                            pu_faces_detected,
                        ) = _maybe_run_people_recovery(
                            people_matcher=people_matcher,
                            people_recovery_mode=str(
                                effective.get(
                                    "people_recovery_mode",
                                    defaults["people_recovery_mode"],
                                )
                            ),
                            image_path=image_path,
                            people_source_path=image_path,
                            people_bbox_offset=(0, 0),
                            people_hint_text="",
                            extra_people_names=list(existing_xmp_people),
                            people_matches=pu_people_matches,
                            people_names=pu_person_names,
                            object_labels=existing_object_labels,
                            ocr_text=existing_ocr_text,
                            caption_output=pu_caption_out,
                            caption_engine=pu_caption_engine,
                            model_image_path=pu_model_path,
                            caption_source_path=image_path,
                            album_title=pu_album_title,
                            printed_album_title=pu_printed_title,
                            photo_count=1,
                            is_cover_page=False,
                            step_fn=_pu_step,
                        )
                    pu_description = pu_caption_out.text

                    pu_people_payload = _serialize_people_matches(pu_people_matches)
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
                    )
                    pu_caption_payload = _build_caption_metadata(
                        requested_engine=str(caption_key[0]),
                        effective_engine=str(pu_caption_out.engine),
                        fallback=bool(pu_caption_out.fallback),
                        error=str(pu_caption_out.error or ""),
                        model=str(
                            caption_key[1]
                            if caption_key[0] in {"qwen", "lmstudio"}
                            else ""
                        ),
                        people_present=pu_people_present,
                        estimated_people_count=pu_estimated_people_count,
                    )
                    pu_ocr_model = str(
                        dict(det.get("ocr") or {}).get("model")
                        or (
                            effective.get("ocr_model", defaults["ocr_model"])
                            if str(
                                effective.get("ocr_engine", defaults["ocr_engine"])
                            ).strip().lower()
                            in {"qwen", "lmstudio"}
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
                            if str(caption_key[0]).strip().lower() in {"qwen", "lmstudio"}
                            else ""
                        ),
                    )
                    pu_subjects = _dedupe(
                        existing_object_labels + existing_ocr_keywords
                    )
                    pu_source_text = read_embedded_source_text(
                        image_path
                    ) or _derived_source_text(image_path)

                    pu_people_detected = (
                        pu_faces_detected > 0 or len(pu_person_names) > 0
                    )
                    pu_people_identified = len(pu_person_names) > 0

                    if not dry_run:
                        pu_img_w, pu_img_h = _get_image_dimensions(image_path)
                        write_xmp_sidecar(
                            sidecar_path,
                            creator_tool=creator_tool,
                            person_names=pu_person_names,
                            subjects=pu_subjects,
                            description=pu_description,
                            album_title=pu_album_title,
                            gps_latitude=str(
                                existing_location.get("gps_latitude") or ""
                            ),
                            gps_longitude=str(
                                existing_location.get("gps_longitude") or ""
                            ),
                            source_text=pu_source_text,
                            ocr_text=existing_ocr_text,
                            detections_payload=pu_updated_det,
                            stitch_key=str(state.get("stitch_key") or ""),
                            ocr_authority_source=str(
                                state.get("ocr_authority_source") or ""
                            ),
                            image_width=pu_img_w,
                            image_height=pu_img_h,
                            ocr_ran=bool(state.get("ocr_ran") or True),
                            people_detected=pu_people_detected,
                            people_identified=pu_people_identified,
                        )

                    current_cast_signature = str(people_matcher.store_signature())
                    existing_manifest_row = manifest.get(str(image_path)) or {}
                    manifest[str(image_path)] = {
                        **existing_manifest_row,
                        "cast_store_signature": current_cast_signature,
                        "people_count": len(pu_person_names),
                    }
                    if bool(effective.get("enable_people", True)):
                        processed_cast_manifest_keys.add(str(image_path))

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
        album_title_hint = _resolve_album_title_hint(image_path, album_title_cache)
        printed_album_title_hint = _resolve_album_printed_title_hint(
            image_path, printed_album_title_cache
        )

        try:
            object_detector = None
            if bool(effective.get("enable_objects", True)):
                object_key = (
                    str(effective.get("model", defaults["model"])),
                    float(
                        effective.get("object_threshold", defaults["object_threshold"])
                    ),
                )
                object_detector = object_detector_cache.get(object_key)
                if object_detector is None:
                    object_detector = _init_object_detector(
                        model_name=str(object_key[0]),
                        confidence=float(object_key[1]),
                    )
                    object_detector_cache[object_key] = object_detector

            ocr_key = (
                str(effective.get("ocr_engine", defaults["ocr_engine"])),
                str(effective.get("ocr_lang", defaults["ocr_lang"])),
                str(effective.get("ocr_model", defaults["ocr_model"])),
                normalize_lmstudio_base_url(
                    str(
                        effective.get(
                            "lmstudio_base_url", defaults["lmstudio_base_url"]
                        )
                    )
                ),
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
                    ocr_engine=ocr_engine,
                    cache=archive_scan_ocr_cache,
                    step_fn=set_step,
                )

            caption_key = (
                str(effective.get("caption_engine", defaults["caption_engine"])),
                str(effective.get("caption_model", defaults["caption_model"])),
                str(effective.get("caption_prompt", defaults["caption_prompt"])),
                int(
                    effective.get("caption_max_tokens", defaults["caption_max_tokens"])
                ),
                float(
                    effective.get(
                        "caption_temperature", defaults["caption_temperature"]
                    )
                ),
                str(effective.get("lmstudio_base_url", defaults["lmstudio_base_url"])),
                str(
                    effective.get(
                        "qwen_attn_implementation", defaults["qwen_attn_implementation"]
                    )
                ),
                int(effective.get("qwen_min_pixels", defaults["qwen_min_pixels"])),
                int(effective.get("qwen_max_pixels", defaults["qwen_max_pixels"])),
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
                    qwen_attn_implementation=caption_key[6],
                    qwen_min_pixels=int(caption_key[7]),
                    qwen_max_pixels=int(caption_key[8]),
                    max_image_edge=int(caption_key[9]),
                    stream=not stdout_only,
                )
                caption_engine_cache[caption_key] = caption_engine

            with prepare_image_layout(
                image_path,
                split_mode=str(
                    effective.get("page_split_mode", defaults["page_split_mode"])
                ),
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
                source_text = read_embedded_source_text(
                    image_path
                ) or _derived_source_text(image_path)
                album_title_hint = (
                    _store_album_title_hint(
                        image_path,
                        album_title_cache,
                        infer_album_title(
                            image_path=image_path,
                            fallback_title=album_title_hint,
                            source_text=source_text,
                        ),
                    )
                    or album_title_hint
                )
                printed_album_title_hint = (
                    _store_album_printed_title_hint(
                        image_path,
                        printed_album_title_cache,
                        infer_printed_album_title(
                            ocr_text="", fallback_title=printed_album_title_hint
                        ),
                    )
                    or printed_album_title_hint
                )

                if layout.page_like and layout.split_mode == "auto":
                    if scan_ocr_authority is not None:
                        page_ocr_text = scan_ocr_authority.ocr_text
                        page_ocr_keywords = list(scan_ocr_authority.ocr_keywords)
                    else:
                        with _prepare_ai_model_image(
                            layout.content_path
                        ) as page_model_image:
                            if set_step:
                                set_step("ocr")
                            page_ocr_text = ocr_engine.read_text(page_model_image)
                            page_ocr_keywords = extract_keywords(
                                page_ocr_text, max_keywords=15
                            )
                    sub_results = [
                        _run_image_analysis(
                            image_path=subphoto.path,
                            people_matcher=people_matcher,
                            object_detector=object_detector,
                            ocr_engine=ocr_engine,
                            caption_engine=caption_engine,
                            requested_caption_engine=str(caption_key[0]),
                            requested_caption_model=str(caption_key[1]),
                            ocr_engine_name=ocr_key[0],
                            ocr_language=ocr_key[1],
                            people_hint_text=page_ocr_text,
                            people_source_path=image_path,
                            people_bbox_offset=_bounds_offset(subphoto.bounds),
                            people_recovery_mode=str(
                                effective.get(
                                    "people_recovery_mode",
                                    defaults["people_recovery_mode"],
                                )
                            ),
                            caption_source_path=image_path,
                            album_title=album_title_hint,
                            printed_album_title=printed_album_title_hint,
                            geocoder=geocoder,
                            step_fn=set_step,
                        )
                        for subphoto in layout.subphotos
                    ]
                    page_album_title = infer_album_title(
                        image_path=layout.original_path,
                        ocr_text=page_ocr_text,
                        fallback_title=album_title_hint,
                    )
                    page_printed_album_title = infer_printed_album_title(
                        ocr_text=page_ocr_text,
                        fallback_title=printed_album_title_hint,
                    )
                    _store_album_title_hint(
                        image_path, album_title_cache, page_album_title
                    )
                    _store_album_printed_title_hint(
                        image_path, printed_album_title_cache, page_printed_album_title
                    )
                    (
                        person_names,
                        object_labels,
                        subjects,
                        description,
                        payload,
                        subphotos_xml,
                    ) = _build_page_payload(
                        layout=layout,
                        sub_results=sub_results,
                        page_ocr_text=page_ocr_text,
                        page_ocr_keywords=page_ocr_keywords,
                        requested_caption_engine=str(caption_key[0]),
                        album_title=page_album_title,
                        printed_album_title=page_printed_album_title,
                    )
                    if str(caption_key[0]) in {"qwen", "lmstudio"}:
                        with _prepare_ai_model_image(
                            layout.content_path
                        ) as page_model_image:
                            if set_step:
                                set_step("caption")
                            page_caption_output = caption_engine.generate(
                                image_path=page_model_image,
                                people=person_names,
                                objects=object_labels,
                                ocr_text=page_ocr_text,
                                source_path=layout.original_path,
                                album_title=page_album_title,
                                printed_album_title=page_printed_album_title,
                                photo_count=len(sub_results),
                            )
                        if (
                            page_caption_output.text
                            and not page_caption_output.fallback
                        ):
                            description = page_caption_output.text
                        page_people_present = any(
                            bool(
                                dict(result.payload.get("caption") or {}).get(
                                    "people_present"
                                )
                            )
                            for result in sub_results
                        )
                        page_estimated_people_count = max(
                            len(person_names),
                            sum(
                                max(
                                    0,
                                    int(
                                        dict(result.payload.get("caption") or {}).get(
                                            "estimated_people_count",
                                            0,
                                        )
                                        or 0
                                    ),
                                )
                                for result in sub_results
                            ),
                        )
                        payload["caption"] = _build_caption_metadata(
                            requested_engine=str(caption_key[0]),
                            effective_engine=str(page_caption_output.engine),
                            fallback=bool(page_caption_output.fallback),
                            error=str(page_caption_output.error or ""),
                            model=str(caption_key[1]),
                            people_present=page_people_present,
                            estimated_people_count=page_estimated_people_count,
                        )
                        (
                            page_gps_latitude,
                            page_gps_longitude,
                            page_location_name,
                        ) = _resolve_location_metadata(
                            requested_caption_engine=str(caption_key[0]),
                            caption_engine=caption_engine,
                            model_image_path=page_model_image,
                            people=person_names,
                            objects=object_labels,
                            ocr_text=page_ocr_text,
                            source_path=image_path,
                            album_title=page_album_title,
                            printed_album_title=page_printed_album_title,
                            is_cover_page=False,
                            people_positions={},
                            fallback_location_name=str(
                                getattr(page_caption_output, "location_name", "") or ""
                            ).strip(),
                        )
                        page_location_payload = _resolve_location_payload(
                            geocoder=geocoder,
                            gps_latitude=page_gps_latitude,
                            gps_longitude=page_gps_longitude,
                            location_name=page_location_name,
                        )
                        if page_location_payload:
                            payload["location"] = page_location_payload
                    people_count = len(person_names)
                    object_count = len(object_labels)
                    ocr_text = page_ocr_text
                    analysis_mode = "page_subphotos"
                    split_applied = bool(layout.split_applied)
                    subphoto_count = len(sub_results)
                else:
                    analysis_target = (
                        layout.content_path if layout.page_like else image_path
                    )
                    analysis = _run_image_analysis(
                        image_path=analysis_target,
                        people_matcher=people_matcher,
                        object_detector=object_detector,
                        ocr_engine=ocr_engine,
                        caption_engine=caption_engine,
                        requested_caption_engine=str(caption_key[0]),
                        requested_caption_model=str(caption_key[1]),
                        ocr_engine_name=ocr_key[0],
                        ocr_language=ocr_key[1],
                        people_source_path=image_path,
                        people_bbox_offset=(
                            _bounds_offset(layout.content_bounds)
                            if layout.page_like
                            else (0, 0)
                        ),
                        people_recovery_mode=str(
                            effective.get(
                                "people_recovery_mode",
                                defaults["people_recovery_mode"],
                            )
                        ),
                        caption_source_path=(
                            image_path if layout.page_like else analysis_target
                        ),
                        album_title=album_title_hint,
                        printed_album_title=printed_album_title_hint,
                        geocoder=geocoder,
                        step_fn=set_step,
                        extra_people_names=existing_xmp_people,
                        is_page_scan=layout.page_like,
                        ocr_text_override=(
                            scan_ocr_authority.ocr_text
                            if scan_ocr_authority is not None
                            else None
                        ),
                    )
                    resolved_album_title = infer_album_title(
                        image_path=image_path,
                        ocr_text=analysis.ocr_text,
                        fallback_title=album_title_hint,
                    )
                    resolved_printed_album_title = infer_printed_album_title(
                        ocr_text=analysis.ocr_text,
                        fallback_title=printed_album_title_hint,
                    )
                    _store_album_title_hint(
                        image_path, album_title_cache, resolved_album_title
                    )
                    _store_album_printed_title_hint(
                        image_path,
                        printed_album_title_cache,
                        resolved_printed_album_title,
                    )
                    person_names = _dedupe(analysis.people_names + existing_xmp_people)
                    subjects = analysis.subjects
                    description = (
                        _build_flat_page_description(analysis=analysis)
                        if layout.page_like
                        else analysis.description
                    )
                    ocr_text = analysis.ocr_text
                    payload = _build_flat_payload(layout, analysis)
                    people_count = len(analysis.people_names)
                    object_count = len(analysis.object_labels)
                    analysis_mode = "page_flat" if layout.page_like else "single_image"

                payload = _refresh_detection_model_metadata(
                    payload,
                    ocr_model=(
                        str(ocr_engine.effective_model_name)
                        if str(ocr_key[0]).strip().lower() in {"qwen", "lmstudio"}
                        else ""
                    ),
                    caption_model=(
                        str(caption_engine.effective_model_name)
                        if str(caption_key[0]).strip().lower() in {"qwen", "lmstudio"}
                        else ""
                    ),
                )

                # Compute per-stage tracking flags for the XMP
                _ocr_ran_flag = (
                    str(effective.get("ocr_engine", defaults["ocr_engine"])).lower()
                    != "none"
                )
                if analysis_mode == "page_subphotos":
                    _total_faces = sum(r.faces_detected for r in sub_results)
                    _people_detected_flag = _total_faces > 0 or len(person_names) > 0
                else:
                    _people_detected_flag = (
                        analysis.faces_detected > 0 or len(person_names) > 0
                    )
                _people_identified_flag = len(person_names) > 0

                if not dry_run:
                    location_payload = (
                        dict(payload.get("location") or {})
                        if isinstance(payload, dict)
                        else {}
                    )
                    img_w, img_h = _get_image_dimensions(image_path)
                    write_xmp_sidecar(
                        sidecar_path,
                        creator_tool=creator_tool,
                        person_names=person_names,
                        subjects=subjects,
                        description=description,
                        album_title=_resolve_album_title_hint(
                            image_path, album_title_cache
                        ),
                        gps_latitude=str(location_payload.get("gps_latitude") or ""),
                        gps_longitude=str(location_payload.get("gps_longitude") or ""),
                        source_text=source_text,
                        ocr_text=ocr_text,
                        detections_payload=payload,
                        subphotos=subphotos_xml,
                        ocr_authority_source=(
                            "archive_stitched"
                            if scan_ocr_authority is not None
                            else ""
                        ),
                        image_width=img_w,
                        image_height=img_h,
                        ocr_ran=_ocr_ran_flag,
                        people_detected=_people_detected_flag,
                        people_identified=_people_identified_flag,
                    )

            if people_matcher is not None:
                current_cast_signature = str(people_matcher.store_signature())
            stat = image_path.stat()
            manifest[str(image_path)] = {
                "image_path": str(image_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sidecar_path": str(sidecar_path),
                "people_count": int(people_count),
                "object_count": int(object_count),
                "ocr_chars": int(len(ocr_text)),
                "analysis_mode": str(analysis_mode),
                "split_applied": bool(split_applied),
                "subphoto_count": int(subphoto_count),
                "processor_signature": PROCESSOR_SIGNATURE,
                "settings_signature": settings_sig,
                "ocr_authority_source": (
                    "archive_stitched" if scan_ocr_authority is not None else ""
                ),
                "ocr_authority_signature": (
                    str(scan_ocr_authority.signature)
                    if scan_ocr_authority is not None
                    else ""
                ),
                "ocr_authority_hash": (
                    str(scan_ocr_authority.ocr_hash)
                    if scan_ocr_authority is not None
                    else ""
                ),
                "cast_store_signature": (
                    current_cast_signature
                    if bool(effective.get("enable_people", True))
                    else ""
                ),
                "render_settings_path": (
                    str(settings_file) if settings_file is not None else ""
                ),
            }
            if bool(effective.get("enable_people", True)):
                processed_cast_manifest_keys.add(str(image_path))
            processed += 1
            completed_times.append(time.monotonic() - file_start)
            if stop_ticker is not None:
                stop_ticker()
            if stdout_only:
                caption_meta = (
                    dict(payload.get("caption") or {})
                    if isinstance(payload, dict)
                    else {}
                )
                fallback_error = str(caption_meta.get("error") or "").strip()
                if bool(caption_meta.get("fallback")) and fallback_error:
                    emit_error(
                        f"[{idx}/{len(files)}] warn  {image_path.name}: caption fallback: {fallback_error}"
                    )
                print(
                    f"{image_path.name}: {description}"
                    if description
                    else image_path.name
                )
            else:
                eta_str = _format_eta(completed_times, len(files) - idx)
                eta_part = f"  {eta_str}" if eta_str else ""
                print(
                    f"[{idx}/{len(files)}]{eta_part}  ok    {image_path.name}",
                    flush=True,
                )
        except Exception as exc:
            failures += 1
            if stop_ticker is not None:
                stop_ticker()
            emit_error(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")

    if not dry_run:
        if processed_cast_manifest_keys and people_matcher_cache:
            final_cast_signature = str(
                next(iter(people_matcher_cache.values())).store_signature()
            )
            for image_key in processed_cast_manifest_keys:
                row = manifest.get(image_key)
                if not isinstance(row, dict):
                    continue
                row["cast_store_signature"] = final_cast_signature
        save_manifest(manifest_path, manifest)

    stitch_failures = 0
    if bool(getattr(args, "stitch_scans", False)):
        emit_info(
            "Scan stitch pass skipped: archive scan OCR stitching now happens during normal processing."
        )

    if not stdout_only:
        print("\nSummary")
        print(f"- Processed: {processed}")
        print(f"- Skipped:   {skipped}")
        print(f"- Failed:    {failures + stitch_failures}")
        print(f"- Manifest:  {manifest_path}")
    return 1 if (failures or stitch_failures) else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
