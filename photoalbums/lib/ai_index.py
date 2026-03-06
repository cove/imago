from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ai_caption import DEFAULT_QWEN_CAPTION_MODEL, CaptionEngine, build_template_caption
from .ai_ocr import OCREngine, extract_keywords
from .ai_render_settings import find_archive_dir_for_image, load_render_settings, resolve_effective_settings
from ..common import PHOTO_ALBUMS_DIR
from .xmp_sidecar import write_xmp_sidecar

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CREATOR_TOOL = "imago-photoalbums-ai-index"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "ai_index_manifest.jsonl"
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"


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


def needs_processing(path: Path, manifest_row: dict[str, Any] | None, force: bool) -> bool:
    if force or manifest_row is None:
        return True
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    recorded_size = int(manifest_row.get("size", -1))
    recorded_mtime = int(manifest_row.get("mtime_ns", -1))
    return int(stat.st_size) != recorded_size or int(stat.st_mtime_ns) != recorded_mtime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index photo album images with cast people matching, YOLO objects, OCR, and XMP sidecars.",
    )
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR), help="Photo Albums root directory.")
    parser.add_argument("--cast-store", default=str(DEFAULT_CAST_STORE), help="Cast store directory.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="JSONL state file path.")
    parser.add_argument("--creator-tool", default=DEFAULT_CREATOR_TOOL, help="XMP CreatorTool value.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument("--object-threshold", type=float, default=0.30, help="Object detection confidence.")
    parser.add_argument("--people-threshold", type=float, default=0.72, help="Face similarity threshold.")
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size in pixels.")
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "docstrange", "paddle"],
        default="docstrange",
        help="OCR backend.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language.")
    parser.add_argument(
        "--caption-engine",
        choices=["none", "template", "qwen"],
        default="template",
        help="Caption backend for XMP description.",
    )
    parser.add_argument(
        "--caption-model",
        default=DEFAULT_QWEN_CAPTION_MODEL,
        help="Model id/path used when caption engine is qwen.",
    )
    parser.add_argument("--caption-max-tokens", type=int, default=96, help="Max new tokens for qwen captions.")
    parser.add_argument("--caption-temperature", type=float, default=0.2, help="Sampling temperature for qwen.")
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
):
    return CaptionEngine(
        engine=str(engine),
        qwen_model=str(model_name),
        qwen_max_tokens=int(max_tokens),
        qwen_temperature=float(temperature),
        fallback_to_template=True,
    )


def _settings_signature(settings: dict[str, Any]) -> str:
    compact = {
        "skip": bool(settings.get("skip", False)),
        "enable_people": bool(settings.get("enable_people", True)),
        "enable_objects": bool(settings.get("enable_objects", True)),
        "ocr_engine": str(settings.get("ocr_engine", "none")),
        "ocr_lang": str(settings.get("ocr_lang", "eng")),
        "people_threshold": float(settings.get("people_threshold", 0.72)),
        "object_threshold": float(settings.get("object_threshold", 0.30)),
        "min_face_size": int(settings.get("min_face_size", 40)),
        "model": str(settings.get("model", "yolo11n.pt")),
        "creator_tool": str(settings.get("creator_tool", DEFAULT_CREATOR_TOOL)),
        "caption_engine": str(settings.get("caption_engine", "template")),
        "caption_model": str(settings.get("caption_model", DEFAULT_QWEN_CAPTION_MODEL)),
        "caption_max_tokens": int(settings.get("caption_max_tokens", 96)),
        "caption_temperature": float(settings.get("caption_temperature", 0.2)),
    }
    return json.dumps(compact, sort_keys=True, ensure_ascii=True)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        "caption_engine": str(args.caption_engine),
        "caption_model": str(args.caption_model),
        "caption_max_tokens": int(args.caption_max_tokens),
        "caption_temperature": float(args.caption_temperature),
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
    caption_engine_cache: dict[tuple[str, str, int, float], CaptionEngine] = {}

    processed = 0
    skipped = 0
    failures = 0

    for idx, image_path in enumerate(files, 1):
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
        effective["caption_engine"] = str(args.caption_engine)
        effective["caption_model"] = str(args.caption_model)
        effective["caption_max_tokens"] = int(args.caption_max_tokens)
        effective["caption_temperature"] = float(args.caption_temperature)
        settings_sig = _settings_signature(effective)

        manifest_row = manifest.get(str(image_path))
        if manifest_row is not None:
            old_sig = str(manifest_row.get("settings_signature") or "")
            if old_sig != settings_sig:
                manifest_row = None
        if not needs_processing(image_path, manifest_row, bool(args.force)):
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

            people_matches = people_matcher.match_image(image_path) if people_matcher else []
            object_matches = object_detector.detect_image(image_path) if object_detector else []
            ocr_text = ocr_engine.read_text(image_path)
            ocr_keywords = extract_keywords(ocr_text, max_keywords=15)

            people_names = [row.name for row in people_matches]
            object_labels = [row.label for row in object_matches]
            subjects = _clean_list(object_labels + ocr_keywords)
            caption_key = (
                str(effective.get("caption_engine", defaults["caption_engine"])),
                str(effective.get("caption_model", defaults["caption_model"])),
                int(effective.get("caption_max_tokens", defaults["caption_max_tokens"])),
                float(effective.get("caption_temperature", defaults["caption_temperature"])),
            )
            caption_engine = caption_engine_cache.get(caption_key)
            if caption_engine is None:
                caption_engine = _init_caption_engine(
                    engine=caption_key[0],
                    model_name=caption_key[1],
                    max_tokens=int(caption_key[2]),
                    temperature=float(caption_key[3]),
                )
                caption_engine_cache[caption_key] = caption_engine
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
                    "engine": str(effective.get("ocr_engine", args.ocr_engine)),
                    "language": str(effective.get("ocr_lang", args.ocr_lang)),
                    "keywords": ocr_keywords,
                    "chars": len(ocr_text),
                },
                "caption": {
                    "requested_engine": str(caption_key[0]),
                    "effective_engine": str(caption_output.engine),
                    "fallback": bool(caption_output.fallback),
                    "error": str(caption_output.error or "")[:500],
                    "model": str(caption_key[1] if caption_key[0] == "qwen" else ""),
                },
            }

            sidecar_path = image_path.with_suffix(".xmp")
            if not args.dry_run:
                write_xmp_sidecar(
                    sidecar_path,
                    creator_tool=str(effective.get("creator_tool", args.creator_tool)),
                    person_names=people_names,
                    subjects=subjects,
                    description=description,
                    ocr_text=ocr_text,
                    detections_payload=payload,
                )

            stat = image_path.stat()
            manifest[str(image_path)] = {
                "image_path": str(image_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sidecar_path": str(sidecar_path),
                "people_count": int(len(people_names)),
                "object_count": int(len(object_labels)),
                "ocr_chars": int(len(ocr_text)),
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
