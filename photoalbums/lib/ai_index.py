from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ai_caption import (
    CaptionEngine,
    DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
    build_page_caption,
    build_template_caption,
    infer_album_context,
    infer_printed_album_title,
    infer_album_title,
    looks_like_album_cover,
    normalize_lmstudio_base_url,
    normalize_qwen_attn_implementation,
    resolve_caption_model,
)
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
from ..naming import parse_album_filename, SCAN_NAME_RE
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MIN_EXISTING_SIDECAR_BYTES = 100
AI_MODEL_MAX_SOURCE_BYTES = 30 * 1024 * 1024
DEFAULT_CREATOR_TOOL = "imago-photoalbums-ai-index"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "ai_index_manifest.jsonl"
)
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"
PROCESSOR_SIGNATURE = "page_split_v13_page_scan_caption_hint"


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


def build_description(*, people: list[str], objects: list[str], ocr_text: str) -> str:
    return build_template_caption(people=people, objects=objects, ocr_text=ocr_text)


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
        "--min-face-size", type=int, default=40, help="Minimum face size in pixels."
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "qwen", "lmstudio"],
        default="lmstudio",
        help="OCR backend.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language.")
    parser.add_argument(
        "--caption-engine",
        choices=["none", "template", "qwen", "lmstudio"],
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
            "After individual processing, combine OCR text across _S#.tif scans of the same page "
            "and re-run the caption engine so cut-off text at scan edges is reconstructed. "
            "Updates XMP sidecars for all scans in each group."
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
        skip_artwork=True,
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
        fallback_to_template=True,
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
        "ocr_engine": str(settings.get("ocr_engine", "none")),
        "ocr_lang": str(settings.get("ocr_lang", "eng")),
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
) -> dict[str, Any]:
    return {
        "requested_engine": str(requested_engine),
        "effective_engine": str(effective_engine),
        "fallback": bool(fallback),
        "error": str(error or "")[:500],
        "model": str(model or ""),
    }


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
    if lat_text and lon_text:
        payload: dict[str, Any] = {
            "gps_latitude": float(lat_text),
            "gps_longitude": float(lon_text),
            "map_datum": "WGS-84",
            "source": "caption",
        }
        if query:
            payload["query"] = query
        if query and geocode_error:
            payload["error"] = geocode_error
        return payload
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
    caption_source_path: Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    step_fn=None,
    extra_people_names: list[str] | None = None,
    is_page_scan: bool = False,
) -> ImageAnalysis:
    use_combined = ocr_engine.engine == "qwen" and caption_engine.engine == "qwen"
    page_photo_count = 0 if is_page_scan else 1

    with _prepare_ai_model_image(image_path) as model_image_path:
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
            )
            ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
        else:
            if step_fn:
                step_fn("ocr")
            ocr_text = ocr_engine.read_text(model_image_path)
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
            )

    subjects = _dedupe(object_labels + ocr_keywords)
    description = caption_output.text or build_description(
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
    )

    payload = {
        "people": [
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
        ],
        "objects": [
            {"label": row.label, "score": round(row.score, 5)} for row in object_matches
        ],
        "ocr": {
            "engine": str(ocr_engine_name),
            "language": str(ocr_language),
            "keywords": ocr_keywords,
            "chars": len(ocr_text),
        },
        "caption": _build_caption_metadata(
            requested_engine=requested_caption_engine,
            effective_engine=str(caption_output.engine),
            fallback=bool(caption_output.fallback),
            error=str(caption_output.error or ""),
            model=str(
                requested_caption_model
                if requested_caption_engine in {"qwen", "lmstudio"}
                else ""
            ),
        ),
    }
    gps_latitude = str(getattr(caption_output, "gps_latitude", "") or "").strip()
    gps_longitude = str(getattr(caption_output, "gps_longitude", "") or "").strip()
    location_name = str(getattr(caption_output, "location_name", "") or "").strip()
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
        "footer_trimmed": bool(layout.footer_trimmed),
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
    album_context = infer_album_context(
        image_path=layout.original_path,
        ocr_text=page_ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )

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
        description = sub_results[0].description or build_page_caption(
            photo_count=1,
            people=people_names,
            objects=object_labels,
            ocr_text=page_ocr_text,
            album_context=album_context,
        )
    else:
        description = build_page_caption(
            photo_count=len(sub_results),
            people=people_names,
            objects=object_labels,
            ocr_text=page_ocr_text,
            album_context=album_context,
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
        ),
        "subphotos": subphoto_rows,
    }
    return people_names, object_labels, subjects, description, payload, subphoto_rows


def _build_flat_page_description(
    *,
    layout: PreparedImageLayout,
    analysis: ImageAnalysis,
    requested_caption_engine: str,
    album_title: str = "",
    printed_album_title: str = "",
) -> str:
    caption_meta = (
        dict(analysis.payload.get("caption") or {})
        if isinstance(analysis.payload, dict)
        else {}
    )
    fallback_used = bool(caption_meta.get("fallback"))
    effective_engine = str(caption_meta.get("effective_engine") or "").strip().lower()
    album_context = infer_album_context(
        image_path=layout.original_path,
        ocr_text=analysis.ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    if (
        fallback_used
        or effective_engine in {"template", "none"}
        or str(requested_caption_engine).strip().lower() in {"template", "none"}
    ):
        return build_page_caption(
            photo_count=1,
            people=analysis.people_names,
            objects=analysis.object_labels,
            ocr_text=analysis.ocr_text,
            album_context=album_context,
        )
    return analysis.description


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
                combined_description = caption_output.text or build_description(
                    people=person_names,
                    objects=object_labels,
                    ocr_text=combined_ocr,
                )
                gps_latitude = str(
                    getattr(caption_output, "gps_latitude", "") or ""
                ).strip()
                gps_longitude = str(
                    getattr(caption_output, "gps_longitude", "") or ""
                ).strip()
                location_name = str(
                    getattr(caption_output, "location_name", "") or ""
                ).strip()
            else:
                # Template / none: rebuild description from combined OCR without AI
                album_context = infer_album_context(
                    image_path=primary_path,
                    ocr_text=combined_ocr,
                    allow_ocr=True,
                    album_title=album_title,
                    printed_album_title=printed_album_title,
                )
                combined_description = build_page_caption(
                    photo_count=max(
                        1,
                        sum(
                            len(
                                list((s.get("detections") or {}).get("subphotos") or [])
                            )
                            for s in states
                        ),
                    ),
                    people=person_names,
                    objects=object_labels,
                    ocr_text=combined_ocr,
                    album_context=album_context,
                )
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

    files = discover_images(
        photos_root,
        include_archive=include_archive,
        include_view=include_view,
        extensions=ext_set,
    )
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
        "ocr_engine": str(args.ocr_engine),
        "ocr_lang": str(args.ocr_lang),
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
    ocr_engine_cache: dict[tuple[str, str, str], OCREngine] = {}
    caption_engine_cache: dict[
        tuple[str, str, str, int, float, str, str, int, int, int], CaptionEngine
    ] = {}
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

        manifest_row = manifest.get(str(image_path))
        reprocess_required = False
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
        if manifest_row is not None:
            old_sig = str(manifest_row.get("settings_signature") or "")
            if old_sig != settings_sig:
                reprocess_required = True
            elif bool(effective.get("enable_people", True)):
                if (
                    str(manifest_row.get("cast_store_signature") or "")
                    != current_cast_signature
                ):
                    reprocess_required = True
        if not needs_processing(
            image_path,
            manifest_row,
            force_processing,
            reprocess_required=reprocess_required,
        ):
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
                    engine=ocr_key[0], language=ocr_key[1], base_url=ocr_key[2]
                )
                ocr_engine_cache[ocr_key] = ocr_engine

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
                source_text = read_embedded_source_text(image_path)
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
                        payload["caption"] = _build_caption_metadata(
                            requested_engine=str(caption_key[0]),
                            effective_engine=str(page_caption_output.engine),
                            fallback=bool(page_caption_output.fallback),
                            error=str(page_caption_output.error or ""),
                            model=str(caption_key[1]),
                        )
                        page_gps_latitude = str(
                            getattr(page_caption_output, "gps_latitude", "") or ""
                        ).strip()
                        page_gps_longitude = str(
                            getattr(page_caption_output, "gps_longitude", "") or ""
                        ).strip()
                        page_location_name = str(
                            getattr(page_caption_output, "location_name", "") or ""
                        ).strip()
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
                        caption_source_path=(
                            image_path if layout.page_like else analysis_target
                        ),
                        album_title=album_title_hint,
                        printed_album_title=printed_album_title_hint,
                        geocoder=geocoder,
                        step_fn=set_step,
                        extra_people_names=existing_xmp_people,
                        is_page_scan=layout.page_like,
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
                        _build_flat_page_description(
                            layout=layout,
                            analysis=analysis,
                            requested_caption_engine=str(caption_key[0]),
                            album_title=resolved_album_title,
                            printed_album_title=resolved_printed_album_title,
                        )
                        if layout.page_like
                        else analysis.description
                    )
                    ocr_text = analysis.ocr_text
                    payload = _build_flat_payload(layout, analysis)
                    people_count = len(analysis.people_names)
                    object_count = len(analysis.object_labels)
                    analysis_mode = "page_flat" if layout.page_like else "single_image"

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
                        image_width=img_w,
                        image_height=img_h,
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
        emit_info("\nScan stitch pass")
        # Use the defaults-based caption engine (or initialise one if none was created yet)
        stitch_caption_key = (
            str(defaults["caption_engine"]),
            str(defaults["caption_model"]),
            str(defaults["caption_prompt"]),
            int(defaults["caption_max_tokens"]),
            float(defaults["caption_temperature"]),
            str(defaults["lmstudio_base_url"]),
            str(defaults["qwen_attn_implementation"]),
            int(defaults["qwen_min_pixels"]),
            int(defaults["qwen_max_pixels"]),
            int(defaults["caption_max_edge"]),
        )
        stitch_caption_engine = caption_engine_cache.get(stitch_caption_key)
        if stitch_caption_engine is None:
            stitch_caption_engine = _init_caption_engine(
                engine=stitch_caption_key[0],
                model_name=stitch_caption_key[1],
                caption_prompt=stitch_caption_key[2],
                max_tokens=int(stitch_caption_key[3]),
                temperature=float(stitch_caption_key[4]),
                lmstudio_base_url=stitch_caption_key[5],
                qwen_attn_implementation=stitch_caption_key[6],
                qwen_min_pixels=int(stitch_caption_key[7]),
                qwen_max_pixels=int(stitch_caption_key[8]),
                max_image_edge=int(stitch_caption_key[9]),
                stream=not stdout_only,
            )
        stitch_failures = _run_scan_stitch_pass(
            files,
            caption_engine=stitch_caption_engine,
            requested_caption_engine=str(defaults["caption_engine"]),
            creator_tool=str(defaults["creator_tool"]),
            dry_run=dry_run,
            stdout_only=stdout_only,
            album_title_cache=album_title_cache,
            printed_album_title_cache=printed_album_title_cache,
            geocoder=geocoder,
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
