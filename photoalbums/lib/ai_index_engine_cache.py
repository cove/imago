from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ai_caption import (
    CaptionEngine,
    normalize_lmstudio_base_url,
    resolve_caption_model,
)
from .ai_date import DateEstimateEngine
from .ai_model_settings import default_lmstudio_base_url


PROCESSOR_SIGNATURE = "page_split_v17_people_recovery_any_people"


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
