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


def _normalize_model_candidates(value: Any, *, alias: str) -> list[str]:
    raw_values = value if isinstance(value, list) else [value]
    models: list[str] = []
    seen_model_names: set[str] = set()
    for raw_model_name in raw_values:
        model_name = _normalize_model_value(raw_model_name)
        if not model_name:
            raise RuntimeError(
                f"AI model settings model '{alias}' must contain only non-empty strings: {AI_MODEL_SETTINGS_PATH}"
            )
        if model_name in seen_model_names:
            raise RuntimeError(
                f"AI model settings model '{alias}' duplicates '{model_name}': {AI_MODEL_SETTINGS_PATH}"
            )
        seen_model_names.add(model_name)
        models.append(model_name)
    if not models:
        raise RuntimeError(
            f"AI model settings model '{alias}' must contain at least one model name: {AI_MODEL_SETTINGS_PATH}"
        )
    return models


def _normalize_model_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"AI model settings 'models' must be a TOML table mapping aliases to model names: {AI_MODEL_SETTINGS_PATH}"
        )
    models: dict[str, list[str]] = {}
    for raw_alias, raw_model_names in value.items():
        alias = _normalize_model_value(raw_alias)
        if not alias:
            raise RuntimeError(f"AI model settings model aliases must be non-empty strings: {AI_MODEL_SETTINGS_PATH}")
        model_names = _normalize_model_candidates(raw_model_names, alias=alias)
        models[alias] = model_names
    if not models:
        raise RuntimeError(f"AI model settings 'models' must contain at least one model: {AI_MODEL_SETTINGS_PATH}")
    return models


def _resolve_selected_alias(payload: dict[str, Any], models: dict[str, list[str]], field_name: str) -> str:
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


def _resolve_selected_alias_optional(payload: dict[str, Any], models: dict[str, list[str]], field_name: str) -> str:
    selected = _normalize_model_value(payload.get(field_name))
    if not selected:
        return ""
    if selected not in models:
        raise RuntimeError(
            f"AI model settings '{field_name}' must match one of the configured model aliases: {AI_MODEL_SETTINGS_PATH}"
        )
    return selected


def _resolve_model_reference(
    payload: dict[str, Any],
    models: dict[str, list[str]],
    field_name: str,
    default: str,
) -> list[str]:
    selected = _normalize_model_value(payload.get(field_name))
    if not selected:
        return [default]
    if selected in models:
        return list(models[selected])
    return [selected]


def _first_model_name(models: list[str], default: str = "") -> str:
    for model_name in list(models or []):
        text = _normalize_model_value(model_name)
        if text:
            return text
    return default


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
    ocr_models = list(models.get(selected_ocr_model, []))
    caption_models = list(models.get(selected_caption_model, []))
    ctm_models = list(models.get(selected_ctm_model, []))
    view_region_models = _resolve_model_reference(payload, models, "view_region_model", DEFAULT_VIEW_REGION_MODEL)
    return {
        "models": models,
        "selected_ocr_model": selected_ocr_model,
        "selected_caption_model": selected_caption_model,
        "selected_ctm_model": selected_ctm_model,
        "ocr_models": ocr_models,
        "caption_models": caption_models,
        "ctm_models": ctm_models,
        "view_region_models": view_region_models,
        "ocr_model": _first_model_name(ocr_models, DEFAULT_OCR_MODEL),
        "caption_model": _first_model_name(caption_models, DEFAULT_CAPTION_MODEL),
        "ctm_model": _first_model_name(ctm_models, DEFAULT_CTM_MODEL),
        "ctm_validation": _resolve_ctm_validation_settings(payload),
        "view_region_model": _first_model_name(view_region_models, DEFAULT_VIEW_REGION_MODEL),
        "lmstudio_base_url": _resolve_lmstudio_base_url(payload),
        "docling_preset": str((payload.get("docling_pipeline") or {}).get("preset") or "granite_docling"),
    }


def default_ocr_model() -> str:
    return str(load_ai_model_settings()["ocr_model"])


def default_ocr_models() -> list[str]:
    return list(load_ai_model_settings()["ocr_models"])


def default_caption_model() -> str:
    return str(load_ai_model_settings()["caption_model"])


def default_caption_models() -> list[str]:
    return list(load_ai_model_settings()["caption_models"])


def default_view_region_model() -> str:
    return str(load_ai_model_settings()["view_region_model"])


def default_view_region_models() -> list[str]:
    return list(load_ai_model_settings()["view_region_models"])


def default_lmstudio_base_url() -> str:
    return str(load_ai_model_settings()["lmstudio_base_url"])


def default_ctm_model() -> str:
    return str(load_ai_model_settings()["ctm_model"])


def default_ctm_models() -> list[str]:
    return list(load_ai_model_settings()["ctm_models"])


def default_ctm_validation_settings() -> dict[str, float]:
    return dict(load_ai_model_settings()["ctm_validation"])


def default_docling_preset() -> str:
    return str(load_ai_model_settings()["docling_preset"])
