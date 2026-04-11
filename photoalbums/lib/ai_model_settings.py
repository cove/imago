from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import tomllib

DEFAULT_OCR_MODEL = ""
DEFAULT_CAPTION_MODEL = ""
DEFAULT_CTM_MODEL = ""
DEFAULT_VIEW_REGION_MODEL = "google/gemma-4-26b-a4b"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
AI_MODEL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "ai_models.toml"
DEFAULT_CTM_VALIDATION_SETTINGS = {
    "min_confidence": 0.35,
    "max_abs_coefficient": 3.0,
    "max_row_sum": 4.0,
    "max_clipping_ratio": 0.34,
}


def _normalize_model_value(value: Any) -> str:
    return str(value or "").strip()


def _normalize_model_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"AI model settings 'models' must be a TOML table mapping aliases to model names: {AI_MODEL_SETTINGS_PATH}"
        )
    models: dict[str, str] = {}
    seen_model_names: set[str] = set()
    for raw_alias, raw_model_name in value.items():
        alias = _normalize_model_value(raw_alias)
        model_name = _normalize_model_value(raw_model_name)
        if not alias:
            raise RuntimeError(f"AI model settings model aliases must be non-empty strings: {AI_MODEL_SETTINGS_PATH}")
        if not model_name:
            raise RuntimeError(
                f"AI model settings model '{alias}' must be a non-empty string: {AI_MODEL_SETTINGS_PATH}"
            )
        if model_name in seen_model_names:
            raise RuntimeError(f"AI model settings model '{alias}' duplicates '{model_name}': {AI_MODEL_SETTINGS_PATH}")
        seen_model_names.add(model_name)
        models[alias] = model_name
    if not models:
        raise RuntimeError(f"AI model settings 'models' must contain at least one model: {AI_MODEL_SETTINGS_PATH}")
    return models


def _resolve_selected_alias(payload: dict[str, Any], models: dict[str, str], field_name: str) -> str:
    if field_name not in payload:
        raise RuntimeError(f"AI model settings must define '{field_name}': {AI_MODEL_SETTINGS_PATH}")
    selected = _normalize_model_value(payload.get(field_name))
    if not selected:
        raise RuntimeError(
            f"AI model settings '{field_name}' must be a non-empty model alias: {AI_MODEL_SETTINGS_PATH}"
        )
    if selected not in models:
        raise RuntimeError(
            f"AI model settings '{field_name}' must match one of the configured model aliases: {AI_MODEL_SETTINGS_PATH}"
        )
    return selected


def _resolve_lmstudio_base_url(payload: dict[str, Any]) -> str:
    text = _normalize_model_value(payload.get("lmstudio_base_url"))
    return text or DEFAULT_LMSTUDIO_BASE_URL


def _resolve_selected_alias_optional(payload: dict[str, Any], models: dict[str, str], field_name: str) -> str:
    selected = _normalize_model_value(payload.get(field_name))
    if not selected:
        return ""
    if selected not in models:
        raise RuntimeError(
            f"AI model settings '{field_name}' must match one of the configured model aliases: {AI_MODEL_SETTINGS_PATH}"
        )
    return selected


def _resolve_ctm_validation_settings(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("ctm_validation")
    if raw is None:
        return dict(DEFAULT_CTM_VALIDATION_SETTINGS)
    if not isinstance(raw, dict):
        raise RuntimeError(f"AI model settings 'ctm_validation' must be a TOML table: {AI_MODEL_SETTINGS_PATH}")
    resolved = dict(DEFAULT_CTM_VALIDATION_SETTINGS)
    for key in resolved:
        value = raw.get(key)
        if value is None:
            continue
        try:
            resolved[key] = float(value)
        except Exception as exc:
            raise RuntimeError(
                f"AI model settings ctm_validation.{key} must be numeric: {AI_MODEL_SETTINGS_PATH}"
            ) from exc
    return resolved


@lru_cache(maxsize=1)
def load_ai_model_settings() -> dict[str, Any]:
    with open(AI_MODEL_SETTINGS_PATH, "rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"AI model settings must be a TOML table: {AI_MODEL_SETTINGS_PATH}")
    models = _normalize_model_map(payload.get("models"))
    selected_ocr_model = _resolve_selected_alias(payload, models, "selected_ocr_model")
    selected_caption_model = _resolve_selected_alias(payload, models, "selected_caption_model")
    selected_ctm_model = _resolve_selected_alias_optional(payload, models, "selected_ctm_model")
    view_region_model = _normalize_model_value(payload.get("view_region_model")) or DEFAULT_VIEW_REGION_MODEL
    return {
        "models": models,
        "selected_ocr_model": selected_ocr_model,
        "selected_caption_model": selected_caption_model,
        "selected_ctm_model": selected_ctm_model,
        "ocr_model": models.get(selected_ocr_model, DEFAULT_OCR_MODEL),
        "caption_model": models.get(selected_caption_model, DEFAULT_CAPTION_MODEL),
        "ctm_model": models.get(selected_ctm_model, DEFAULT_CTM_MODEL),
        "ctm_validation": _resolve_ctm_validation_settings(payload),
        "view_region_model": view_region_model,
        "lmstudio_base_url": _resolve_lmstudio_base_url(payload),
    }


def default_ocr_model() -> str:
    return str(load_ai_model_settings()["ocr_model"])


def default_caption_model() -> str:
    return str(load_ai_model_settings()["caption_model"])


def default_view_region_model() -> str:
    return str(load_ai_model_settings()["view_region_model"])


def default_lmstudio_base_url() -> str:
    return str(load_ai_model_settings()["lmstudio_base_url"])


def default_ctm_model() -> str:
    return str(load_ai_model_settings()["ctm_model"])


def default_ctm_validation_settings() -> dict[str, float]:
    return dict(load_ai_model_settings()["ctm_validation"])
