from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ai_caption import (
    CaptionEngine,
    build_page_caption,
    build_template_caption,
    normalize_qwen_attn_implementation,
    resolve_caption_model,
)
from .ai_ocr import OCREngine, extract_keywords
from .ai_page_layout import PreparedImageLayout, classify_image_kind, prepare_image_layout
from .ai_render_settings import find_archive_dir_for_image, load_render_settings, resolve_effective_settings
from ..common import PHOTO_ALBUMS_DIR
from ..exiftool_utils import read_tag
from .xmp_sidecar import write_xmp_sidecar

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MIN_EXISTING_SIDECAR_BYTES = 100
DEFAULT_CREATOR_TOOL = "imago-photoalbums-ai-index"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "ai_index_manifest.jsonl"
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"
PROCESSOR_SIGNATURE = "page_split_v2_source_merge"


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


def _clean_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
        if int(stat.st_size) != recorded_size or int(stat.st_mtime_ns) != recorded_mtime:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index photo album images with cast people matching, YOLO objects, OCR, and XMP sidecars.",
    )
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR), help="Photo Albums root directory.")
    parser.add_argument("--cast-store", default=str(DEFAULT_CAST_STORE), help="Cast store directory.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="JSONL state file path.")
    parser.add_argument("--creator-tool", default=DEFAULT_CREATOR_TOOL, help="XMP CreatorTool value.")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument("--object-threshold", type=float, default=0.30, help="Object detection confidence.")
    parser.add_argument("--people-threshold", type=float, default=0.72, help="Face similarity threshold.")
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size in pixels.")
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "docstrange"],
        default="docstrange",
        help="OCR backend.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language.")
    parser.add_argument(
        "--caption-engine",
        choices=["none", "template", "blip", "qwen"],
        default="blip",
        help="Caption backend for XMP description.",
    )
    parser.add_argument(
        "--caption-model",
        default="",
        help="Optional model id/path used by the selected caption engine.",
    )
    parser.add_argument("--caption-max-tokens", type=int, default=96, help="Max new tokens for caption models.")
    parser.add_argument("--caption-temperature", type=float, default=0.2, help="Sampling temperature for qwen.")
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
    parser.add_argument("--max-images", type=int, default=0, help="Optional processing limit.")
    parser.add_argument("--force", action="store_true", help="Ignore manifest and process all files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write sidecar/manifest.")
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
    max_tokens: int,
    temperature: float,
    qwen_attn_implementation: str,
    qwen_min_pixels: int,
    qwen_max_pixels: int,
    max_image_edge: int,
):
    return CaptionEngine(
        engine=str(engine),
        model_name=str(model_name),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        qwen_attn_implementation=str(qwen_attn_implementation),
        qwen_min_pixels=int(qwen_min_pixels),
        qwen_max_pixels=int(qwen_max_pixels),
        max_image_edge=int(max_image_edge),
        fallback_to_template=True,
    )


def _settings_signature(settings: dict[str, Any]) -> str:
    caption_engine = str(settings.get("caption_engine", "blip"))
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
        "page_split_mode": str(settings.get("page_split_mode", "auto")),
        "people_threshold": float(settings.get("people_threshold", 0.72)),
        "object_threshold": float(settings.get("object_threshold", 0.30)),
        "min_face_size": int(settings.get("min_face_size", 40)),
        "model": str(settings.get("model", "models/yolo11n.pt")),
        "creator_tool": str(settings.get("creator_tool", DEFAULT_CREATOR_TOOL)),
        "caption_engine": caption_engine,
        "caption_model": caption_model,
        "caption_max_tokens": int(settings.get("caption_max_tokens", 96)),
        "caption_temperature": float(settings.get("caption_temperature", 0.2)),
        "caption_max_edge": int(settings.get("caption_max_edge", 0)),
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
) -> ImageAnalysis:
    people_matches = people_matcher.match_image(image_path) if people_matcher else []
    object_matches = object_detector.detect_image(image_path) if object_detector else []
    ocr_text = ocr_engine.read_text(image_path)
    ocr_keywords = extract_keywords(ocr_text, max_keywords=15)

    people_names = [row.name for row in people_matches]
    object_labels = [row.label for row in object_matches]
    subjects = _clean_list(object_labels + ocr_keywords)
    caption_output = caption_engine.generate(
        image_path=image_path,
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
    )
    description = caption_output.text or build_description(
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
    )
    payload = {
        "people": [{"name": row.name, "score": round(row.score, 5)} for row in people_matches],
        "objects": [{"label": row.label, "score": round(row.score, 5)} for row in object_matches],
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
            model=str(requested_caption_model if requested_caption_engine in {"qwen", "blip"} else ""),
        ),
    }
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


def _aggregate_best_rows(results: list[ImageAnalysis], section: str, key_name: str) -> list[dict[str, Any]]:
    best: dict[str, float] = {}
    for result in results:
        for row in list(result.payload.get(section) or []):
            name = str(row.get(key_name) or "").strip()
            if not name:
                continue
            score = float(row.get("score") or 0.0)
            current = best.get(name)
            if current is None or score > current:
                best[name] = score
    out = [{key_name: name, "score": round(score, 5)} for name, score in best.items()]
    out.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get(key_name) or "").casefold()))
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


def _build_flat_payload(layout: PreparedImageLayout, analysis: ImageAnalysis) -> dict[str, Any]:
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

    subjects = _clean_list(page_subjects)
    description = build_page_caption(
        photo_count=len(sub_results),
        people=people_names,
        objects=object_labels,
        ocr_text=page_ocr_text,
    )
    payload = {
        "layout": _layout_payload(layout),
        "people": aggregate_people,
        "objects": aggregate_objects,
        "ocr": {
            "engine": str(sub_results[0].payload["ocr"]["engine"]) if sub_results else "",
            "language": str(sub_results[0].payload["ocr"]["language"]) if sub_results else "",
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


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_flags = _explicit_cli_flags(argv)
    photos_root = Path(args.photos_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not photos_root.is_dir():
        raise SystemExit(f"Photo root is not a directory: {photos_root}")

    include_archive = bool(args.include_archive)
    include_view = bool(args.include_view)
    if not include_archive and not include_view:
        include_archive = True
        include_view = True

    ext_set = {
        item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
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

    print(f"Discovered {len(files)} image files")
    if not files:
        return 0

    defaults = {
        "skip": False,
        "enable_people": not bool(args.disable_people),
        "enable_objects": not bool(args.disable_objects),
        "ocr_engine": str(args.ocr_engine),
        "ocr_lang": str(args.ocr_lang),
        "page_split_mode": "auto",
        "caption_engine": str(args.caption_engine),
        "caption_model": resolve_caption_model(str(args.caption_engine), str(args.caption_model)),
        "caption_max_tokens": int(args.caption_max_tokens),
        "caption_temperature": float(args.caption_temperature),
        "caption_max_edge": int(args.caption_max_edge),
        "qwen_attn_implementation": normalize_qwen_attn_implementation(str(args.qwen_attn_implementation)),
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
    ocr_engine_cache: dict[tuple[str, str], OCREngine] = {}
    caption_engine_cache: dict[tuple[str, str, int, float, str, int, int, int], CaptionEngine] = {}

    processed = 0
    skipped = 0
    failures = 0

    for idx, image_path in enumerate(files, 1):
        sidecar_path = image_path.with_suffix(".xmp")
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
        if "--caption-engine" in explicit_flags:
            effective["caption_engine"] = str(args.caption_engine)
        if "--caption-model" in explicit_flags:
            effective["caption_model"] = str(args.caption_model)
        if "--caption-max-tokens" in explicit_flags:
            effective["caption_max_tokens"] = int(args.caption_max_tokens)
        if "--caption-temperature" in explicit_flags:
            effective["caption_temperature"] = float(args.caption_temperature)
        if "--caption-max-edge" in explicit_flags:
            effective["caption_max_edge"] = int(args.caption_max_edge)
        if "--qwen-attn-implementation" in explicit_flags:
            effective["qwen_attn_implementation"] = normalize_qwen_attn_implementation(str(args.qwen_attn_implementation))
        if "--qwen-min-pixels" in explicit_flags:
            effective["qwen_min_pixels"] = int(args.qwen_min_pixels)
        if "--qwen-max-pixels" in explicit_flags:
            effective["qwen_max_pixels"] = int(args.qwen_max_pixels)
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", defaults["caption_engine"])),
            str(effective.get("caption_model", defaults["caption_model"])),
        )
        settings_sig = _settings_signature(effective)

        manifest_row = manifest.get(str(image_path))
        reprocess_required = False
        if manifest_row is not None:
            old_sig = str(manifest_row.get("settings_signature") or "")
            if old_sig != settings_sig:
                reprocess_required = True
        if not needs_processing(
            image_path,
            manifest_row,
            bool(args.force),
            reprocess_required=reprocess_required,
        ):
            skipped += 1
            if args.verbose:
                print(f"[{idx}/{len(files)}] skip  {image_path.name}")
            continue

        if bool(effective.get("skip", False)):
            skipped += 1
            if args.verbose:
                print(f"[{idx}/{len(files)}] skip  {image_path.name} (render_settings skip=true)")
            continue

        try:
            people_matcher = None
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

            ocr_key = (
                str(effective.get("ocr_engine", defaults["ocr_engine"])),
                str(effective.get("ocr_lang", defaults["ocr_lang"])),
            )
            ocr_engine = ocr_engine_cache.get(ocr_key)
            if ocr_engine is None:
                ocr_engine = OCREngine(engine=ocr_key[0], language=ocr_key[1])
                ocr_engine_cache[ocr_key] = ocr_engine

            caption_key = (
                str(effective.get("caption_engine", defaults["caption_engine"])),
                str(effective.get("caption_model", defaults["caption_model"])),
                int(effective.get("caption_max_tokens", defaults["caption_max_tokens"])),
                float(effective.get("caption_temperature", defaults["caption_temperature"])),
                str(effective.get("qwen_attn_implementation", defaults["qwen_attn_implementation"])),
                int(effective.get("qwen_min_pixels", defaults["qwen_min_pixels"])),
                int(effective.get("qwen_max_pixels", defaults["qwen_max_pixels"])),
                int(effective.get("caption_max_edge", defaults["caption_max_edge"])),
            )
            caption_engine = caption_engine_cache.get(caption_key)
            if caption_engine is None:
                caption_engine = _init_caption_engine(
                    engine=caption_key[0],
                    model_name=caption_key[1],
                    max_tokens=int(caption_key[2]),
                    temperature=float(caption_key[3]),
                    qwen_attn_implementation=caption_key[4],
                    qwen_min_pixels=int(caption_key[5]),
                    qwen_max_pixels=int(caption_key[6]),
                    max_image_edge=int(caption_key[7]),
                )
                caption_engine_cache[caption_key] = caption_engine

            with prepare_image_layout(
                image_path,
                split_mode=str(effective.get("page_split_mode", defaults["page_split_mode"])),
            ) as layout:
                creator_tool = str(effective.get("creator_tool", args.creator_tool))
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

                if layout.page_like and layout.split_mode == "auto":
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
                        )
                        for subphoto in layout.subphotos
                    ]
                    page_ocr_text = ocr_engine.read_text(layout.content_path)
                    page_ocr_keywords = extract_keywords(page_ocr_text, max_keywords=15)
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
                    )
                    people_count = len(person_names)
                    object_count = len(object_labels)
                    ocr_text = page_ocr_text
                    analysis_mode = "page_subphotos"
                    split_applied = bool(layout.split_applied)
                    subphoto_count = len(sub_results)
                else:
                    analysis_target = layout.content_path if layout.page_like else image_path
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
                    )
                    person_names = analysis.people_names
                    subjects = analysis.subjects
                    description = analysis.description
                    ocr_text = analysis.ocr_text
                    payload = _build_flat_payload(layout, analysis)
                    people_count = len(analysis.people_names)
                    object_count = len(analysis.object_labels)
                    analysis_mode = "page_flat" if layout.page_like else "single_image"

                if not args.dry_run:
                    write_xmp_sidecar(
                        sidecar_path,
                        creator_tool=creator_tool,
                        person_names=person_names,
                        subjects=subjects,
                        description=description,
                        source_text=source_text,
                        ocr_text=ocr_text,
                        detections_payload=payload,
                        subphotos=subphotos_xml,
                    )

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
                "render_settings_path": str(settings_file) if settings_file is not None else "",
            }
            processed += 1
            print(f"[{idx}/{len(files)}] ok    {image_path.name}")
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(files)}] fail  {image_path.name}: {exc}")

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    print("\nSummary")
    print(f"- Processed: {processed}")
    print(f"- Skipped:   {skipped}")
    print(f"- Failed:    {failures}")
    print(f"- Manifest:  {manifest_path}")
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
