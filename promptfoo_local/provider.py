from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.lib._caption_lmstudio import (
    _build_data_url,
    _lmstudio_caption_response_format,
    _lmstudio_page_caption_response_format,
    _lmstudio_request_json,
)

DEFAULT_TIMEOUT_SECONDS = 300.0
DIRECT_GENERATION_KEYS = (
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def _resolve_response_format(name: str) -> dict[str, object]:
    text = str(name or "page_caption_payload").strip()
    if text == "page_caption_payload":
        return _lmstudio_page_caption_response_format()
    if text == "caption_payload":
        return _lmstudio_caption_response_format()
    raise ValueError(f"Unsupported response_format: {text}")


def _normalize_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _normalize_path_list(value: object) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [Path(str(item)) for item in value if str(item or "").strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [Path(str(item)) for item in parsed if str(item or "").strip()]
        return [Path(text)]
    return []


def _merge_settings(options: dict, context: dict) -> dict:
    config = _normalize_mapping(options.get("config"))
    vars_payload = _normalize_mapping(context.get("vars"))
    merged = _normalize_mapping(vars_payload.get("run_config"))
    merged.update(config)
    return merged


def call_api(prompt: str, options: dict, context: dict) -> dict:
    settings = _merge_settings(options, context)
    vars_payload = _normalize_mapping(context.get("vars"))
    image_paths = _normalize_path_list(vars_payload.get("image_paths"))
    if not image_paths:
        photo_path = str(vars_payload.get("photo_path") or "").strip()
        if photo_path:
            image_paths = [Path(photo_path)]
    if not image_paths:
        raise ValueError("No image_paths or photo_path found in promptfoo test vars")

    content = [{"type": "text", "text": prompt}]
    image_max_edge = int(settings.get("image_max_edge") or 1600)
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _build_data_url(image_path, image_max_edge)},
            }
        )

    payload = {
        "model": str(settings.get("model") or ""),
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ],
        "response_format": _resolve_response_format(str(settings.get("response_format") or "")),
        "max_tokens": int(settings.get("max_tokens") or 256),
        "temperature": float(settings.get("temperature") or 0.0),
        "stream": bool(settings.get("stream", False)),
    }
    for key in DIRECT_GENERATION_KEYS:
        if key in settings:
            payload[key] = settings[key]

    endpoint = str(settings.get("endpoint") or "http://127.0.0.1:1234/v1/chat/completions")
    timeout_seconds = float(settings.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    response = _lmstudio_request_json(endpoint, payload=payload, timeout=timeout_seconds)
    choices = list(response.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    choice = dict(choices[0] or {})
    message = dict(choice.get("message") or {})
    raw_output = str(message.get("content") or "")
    metadata = {
        "finish_reason": str(choice.get("finish_reason") or ""),
        "model": str(settings.get("model") or ""),
        "settings": settings,
        "image_paths": [str(path) for path in image_paths],
    }
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        metadata["json_parse_error"] = f"{type(exc).__name__}: {exc}"
    else:
        metadata["parsed_json"] = parsed
    return {"output": raw_output, "metadata": metadata}
