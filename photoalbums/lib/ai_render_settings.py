from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_FILENAME = "render_settings.json"
OCR_ENGINES = {"none", "docstrange"}
PAGE_SPLIT_MODES = {"auto", "off"}
CAPTION_ENGINES = {"none", "template", "blip", "qwen", "lmstudio"}
QWEN_ATTN_IMPLEMENTATIONS = {"auto", "sdpa", "flash_attention_2", "eager"}


def render_settings_path(archive_dir: str | Path) -> Path:
    return Path(archive_dir) / SETTINGS_FILENAME


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not (parsed == parsed):
        parsed = float(default)
    return max(float(min_value), min(float(max_value), float(parsed)))


def _normalize_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(int(min_value), min(int(max_value), int(parsed)))


def _normalize_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else str(default)


def _normalize_ocr_engine(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in OCR_ENGINES:
        return text
    fallback = str(default or "none").strip().lower()
    if fallback in OCR_ENGINES:
        return fallback
    return "none"


def _normalize_page_split_mode(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in PAGE_SPLIT_MODES:
        return text
    fallback = str(default or "auto").strip().lower()
    if fallback in PAGE_SPLIT_MODES:
        return fallback
    return "auto"


def _normalize_caption_engine(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in CAPTION_ENGINES:
        return text
    fallback = str(default or "blip").strip().lower()
    if fallback in CAPTION_ENGINES:
        return fallback
    return "blip"


def _normalize_qwen_attn_implementation(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in QWEN_ATTN_IMPLEMENTATIONS:
        return text
    fallback = str(default or "auto").strip().lower()
    if fallback in QWEN_ATTN_IMPLEMENTATIONS:
        return fallback
    return "auto"


def _normalize_settings_block(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    block = dict(raw or {})
    return {
        "skip": _normalize_bool(block.get("skip"), bool(defaults.get("skip", False))),
        "enable_people": _normalize_bool(
            block.get("enable_people"),
            bool(defaults.get("enable_people", True)),
        ),
        "enable_objects": _normalize_bool(
            block.get("enable_objects"),
            bool(defaults.get("enable_objects", True)),
        ),
        "ocr_engine": _normalize_ocr_engine(
            block.get("ocr_engine"),
            str(defaults.get("ocr_engine", "none")),
        ),
        "ocr_lang": _normalize_text(
            block.get("ocr_lang"),
            str(defaults.get("ocr_lang", "eng")),
        ),
        "page_split_mode": _normalize_page_split_mode(
            block.get("page_split_mode"),
            str(defaults.get("page_split_mode", "auto")),
        ),
        "caption_engine": _normalize_caption_engine(
            block.get("caption_engine"),
            str(defaults.get("caption_engine", "blip")),
        ),
        "caption_model": _normalize_text(
            block.get("caption_model"),
            str(defaults.get("caption_model", "")),
        ),
        "caption_prompt": _normalize_text(
            block.get("caption_prompt", block.get("qwen_prompt")),
            str(defaults.get("caption_prompt", "")),
        ),
        "lmstudio_base_url": _normalize_text(
            block.get("lmstudio_base_url"),
            str(defaults.get("lmstudio_base_url", "http://127.0.0.1:1234/v1")),
        ),
        "caption_max_tokens": _normalize_int(
            block.get("caption_max_tokens"),
            int(defaults.get("caption_max_tokens", 96)),
            min_value=8,
            max_value=4096,
        ),
        "caption_temperature": _normalize_float(
            block.get("caption_temperature"),
            float(defaults.get("caption_temperature", 0.2)),
            min_value=0.0,
            max_value=2.0,
        ),
        "caption_max_edge": _normalize_int(
            block.get("caption_max_edge"),
            int(defaults.get("caption_max_edge", 0)),
            min_value=0,
            max_value=8192,
        ),
        "qwen_attn_implementation": _normalize_qwen_attn_implementation(
            block.get("qwen_attn_implementation"),
            str(defaults.get("qwen_attn_implementation", "auto")),
        ),
        "qwen_min_pixels": _normalize_int(
            block.get("qwen_min_pixels"),
            int(defaults.get("qwen_min_pixels", 0)),
            min_value=0,
            max_value=32_000_000,
        ),
        "qwen_max_pixels": _normalize_int(
            block.get("qwen_max_pixels"),
            int(defaults.get("qwen_max_pixels", 0)),
            min_value=0,
            max_value=32_000_000,
        ),
        "people_threshold": _normalize_float(
            block.get("people_threshold"),
            float(defaults.get("people_threshold", 0.72)),
            min_value=-1.0,
            max_value=1.0,
        ),
        "object_threshold": _normalize_float(
            block.get("object_threshold"),
            float(defaults.get("object_threshold", 0.30)),
            min_value=0.0,
            max_value=1.0,
        ),
        "min_face_size": _normalize_int(
            block.get("min_face_size"),
            int(defaults.get("min_face_size", 40)),
            min_value=10,
            max_value=2000,
        ),
        "model": _normalize_text(
            block.get("model"),
            str(defaults.get("model", "models/yolo11n.pt")),
        ),
        "creator_tool": _normalize_text(
            block.get("creator_tool"),
            str(defaults.get("creator_tool", "imago-photoalbums-ai-index")),
        ),
    }


def _empty_settings_template() -> dict[str, Any]:
    return {
        "version": 1,
        "_comments": {
            "archive_settings": "Defaults for all photos in this _Archive.",
            "image_settings": "Optional per-image overrides keyed by image filename.",
        },
        "archive_settings": {},
        "image_settings": {},
    }


def load_render_settings(
    archive_dir: str | Path,
    *,
    defaults: dict[str, Any],
    create: bool = False,
) -> tuple[Path, dict[str, Any]]:
    path = render_settings_path(archive_dir)
    payload: dict[str, Any]

    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = {}
        payload = dict(parsed) if isinstance(parsed, dict) else {}
    else:
        payload = {}

    if create and not path.exists():
        template = _empty_settings_template()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
        payload = template

    archive_raw = payload.get("archive_settings")
    archive_block = dict(archive_raw or {}) if isinstance(archive_raw, dict) else {}
    normalized_archive = _normalize_settings_block(archive_block, defaults)

    image_raw = payload.get("image_settings")
    image_settings: dict[str, dict[str, Any]] = {}
    if isinstance(image_raw, dict):
        for key, value in image_raw.items():
            filename = str(key or "").strip()
            if not filename:
                continue
            if not isinstance(value, dict):
                continue
            image_settings[filename] = _normalize_settings_block(value, normalized_archive)

    return path, {
        "version": int(payload.get("version") or 1),
        "archive_settings": normalized_archive,
        "image_settings": image_settings,
    }


def resolve_effective_settings(
    image_path: str | Path,
    *,
    defaults: dict[str, Any],
    loaded: dict[str, Any] | None,
) -> dict[str, Any]:
    base = _normalize_settings_block({}, defaults)
    if not loaded:
        return base

    archive_settings = loaded.get("archive_settings")
    if isinstance(archive_settings, dict):
        base = _normalize_settings_block(archive_settings, base)

    image_name = Path(image_path).name
    image_settings = loaded.get("image_settings")
    if isinstance(image_settings, dict):
        selected = image_settings.get(image_name)
        if isinstance(selected, dict):
            base = _normalize_settings_block(selected, base)

    return base


def find_archive_dir_for_image(image_path: str | Path) -> Path | None:
    path = Path(image_path)
    for parent in path.parents:
        name = str(parent.name or "")
        if name.endswith("_Archive"):
            return parent
        if name.endswith("_View"):
            sibling = parent.with_name(name[: -len("_View")] + "_Archive")
            if sibling.exists() and sibling.is_dir():
                return sibling
    return None
