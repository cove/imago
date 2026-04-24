from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .ai_caption import CaptionEngine, normalize_lmstudio_base_url
from .ai_date import DateEstimateEngine
from .ai_geocode import NominatimGeocoder
from .ai_index_runner import IndexRunner, _write_sidecar_and_record
from .ai_model_settings import default_lmstudio_base_url
from .ai_sidecar_state import read_ai_sidecar_state
from .ai_location import run_locations_step
from ._caption_lmstudio import (
    DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _build_data_url,
    _format_lmstudio_debug_response,
    _lmstudio_request_json,
    _select_lmstudio_model,
)
from ._lmstudio_helpers import emit_prompt_debug, json_schema_response_format
from .ai_prompt_assets import load_params, load_prompt, params_metadata, prompt_metadata
from .ai_photo_crops import _expected_crop_output_paths
from .prompt_debug import PromptDebugSession
from .xmp_review import load_ai_xmp_review
from .xmp_sidecar import read_pipeline_state, read_region_list, write_pipeline_steps, xmp_datetime_now

VERIFICATION_CONCERNS = ("caption", "gps", "shown_location", "date", "overall")
ROUTABLE_CONCERNS = ("caption", "gps", "shown_location", "date")
VERDICTS = {"good", "bad", "uncertain"}


def verification_system_prompt() -> str:
    return load_prompt("verify-crops/verification/system.md").rendered


def _verification_params() -> dict[str, object]:
    return dict(load_params("verify-crops/verification/params.toml").values)


def _verify_prompt_metadata(*, variant: str, resolved_params: dict[str, object]) -> dict[str, object]:
    if variant == "retry":
        prompt_assets = [load_prompt("verify-crops/retry/user.md")]
        params_asset = load_params("verify-crops/retry/params.toml")
    elif variant == "parameter-suggestion":
        prompt_assets = [load_prompt("verify-crops/parameter-suggestion/user.md")]
        params_asset = load_params("verify-crops/parameter-suggestion/params.toml")
    else:
        prompt_assets = [
            load_prompt("verify-crops/verification/system.md"),
            load_prompt("verify-crops/verification/user.md"),
        ]
        params_asset = load_params("verify-crops/verification/params.toml")
    metadata = {}
    metadata.update(prompt_metadata(*prompt_assets))
    metadata.update(params_metadata(params_asset, resolved_params))
    return metadata


def verify_crops_prompt_hash_payload() -> dict[str, object]:
    return {
        "verification": _verify_prompt_metadata(variant="verification", resolved_params=_verification_params()),
        "retry": _verify_prompt_metadata(variant="retry", resolved_params=dict(load_params("verify-crops/retry/params.toml").values)),
        "parameter_suggestion": _verify_prompt_metadata(
            variant="parameter-suggestion",
            resolved_params=dict(load_params("verify-crops/parameter-suggestion/params.toml").values),
        ),
    }


def _concern_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": sorted(VERDICTS)},
            "reasoning": {"type": "string"},
            "failure_reason": {"type": "string"},
        },
        "required": ["verdict", "reasoning", "failure_reason"],
        "additionalProperties": False,
    }


def verification_response_format() -> dict[str, object]:
    properties = {name: _concern_schema() for name in VERIFICATION_CONCERNS}
    properties["human_inference"] = {"type": "string"}
    routing_items = {"type": "string", "enum": list(ROUTABLE_CONCERNS)}
    properties["needs_another_pass"] = {"type": "array", "items": routing_items}
    properties["needs_human_review"] = {"type": "array", "items": routing_items}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "crop_metadata_verification_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": [
                    *VERIFICATION_CONCERNS,
                    "human_inference",
                    "needs_another_pass",
                    "needs_human_review",
                ],
                "additionalProperties": False,
            },
        },
    }


def parameter_suggestion_response_format() -> dict[str, object]:
    return json_schema_response_format(
        schema_name="crop_metadata_verification_parameter_suggestion_payload",
        properties={
            "caption_max_tokens": {"type": "integer"},
            "caption_temperature": {"type": "number"},
            "caption_max_edge": {"type": "integer"},
            "reason": {"type": "string"},
        },
        required=[
            "caption_max_tokens",
            "caption_temperature",
            "caption_max_edge",
            "reason",
        ],
    )


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _format_location_payload(review: dict[str, object]) -> str:
    fields = (
        ("city", review.get("location_city")),
        ("state", review.get("location_state")),
        ("country", review.get("location_country")),
        ("sublocation", review.get("location_sublocation")),
        ("gps_latitude", review.get("gps_latitude")),
        ("gps_longitude", review.get("gps_longitude")),
    )
    parts = [f"{key}={_clean_text(value)}" for key, value in fields if _clean_text(value)]
    return ", ".join(parts)


def render_xmp_review_text(review: dict[str, object], *, include_ocr_text: bool = True) -> str:
    lines: list[str] = []
    simple_fields: list[tuple[str, object]] = [
        ("title", review.get("title")),
        ("album_title", review.get("album_title")),
        ("description", review.get("description")),
        ("dc_date", review.get("dc_date")),
    ]
    if include_ocr_text:
        simple_fields.extend(
            [
                ("author_text", review.get("author_text")),
                ("scene_text", review.get("scene_text")),
                ("ocr_text", review.get("ocr_text")),
            ]
        )
    simple_fields.append(("source_text", review.get("source_text")))
    for label, value in simple_fields:
        text = _clean_text(value)
        if text:
            lines.append(f"{label}: {text}")
    if include_ocr_text:
        location_text = _format_location_payload(review)
        if location_text:
            lines.append(f"location: {location_text}")
        detections = review.get("detections")
        if isinstance(detections, dict):
            locations_shown = list(detections.get("locations_shown") or [])
            if locations_shown:
                lines.append(f"locations_shown: {json.dumps(locations_shown, ensure_ascii=False, sort_keys=True)}")
    person_names = list(review.get("person_names") or [])
    if person_names:
        lines.append(f"person_names: {json.dumps(person_names, ensure_ascii=False)}")
    subjects = list(review.get("subjects") or [])
    if subjects:
        lines.append(f"subjects: {json.dumps(subjects, ensure_ascii=False)}")
    return "\n".join(lines).strip()


def _location_verification_entries(
    review: dict[str, object],
    *,
    geocoder: NominatimGeocoder | None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    primary_lat = _clean_text(review.get("gps_latitude"))
    primary_lon = _clean_text(review.get("gps_longitude"))
    if primary_lat and primary_lon:
        entries.append(
            {
                "kind": "primary_gps",
                "description": _clean_text(review.get("description")),
                "gps_latitude": primary_lat,
                "gps_longitude": primary_lon,
            }
        )

    detections = review.get("detections")
    locations_shown = list(detections.get("locations_shown") or []) if isinstance(detections, dict) else []
    for idx, location in enumerate(locations_shown, 1):
        if not isinstance(location, dict):
            continue
        lat = _clean_text(location.get("gps_latitude"))
        lon = _clean_text(location.get("gps_longitude"))
        if not lat or not lon:
            continue
        entries.append(
            {
                "kind": "shown_location",
                "index": idx,
                "description": _clean_text(review.get("description")),
                "gps_latitude": lat,
                "gps_longitude": lon,
            }
        )

    if geocoder is None:
        return entries
    for entry in entries:
        result = geocoder.reverse_geocode(
            _clean_text(entry.get("gps_latitude")),
            _clean_text(entry.get("gps_longitude")),
        )
        if result is None:
            entry["nominatim_reverse_lookup"] = "miss"
            continue
        entry["nominatim_reverse_lookup"] = {
            "display_name": _clean_text(getattr(result, "display_name", "")),
            "city": _clean_text(getattr(result, "city", "")),
            "state": _clean_text(getattr(result, "state", "")),
            "country": _clean_text(getattr(result, "country", "")),
            "sublocation": _clean_text(getattr(result, "sublocation", "")),
            "source": _clean_text(getattr(result, "source", "")) or "nominatim",
        }
    return entries


def render_location_verification_text(
    review: dict[str, object],
    *,
    geocoder: NominatimGeocoder | None,
) -> str:
    entries = _location_verification_entries(review, geocoder=geocoder)
    if not entries:
        return ""
    return json.dumps(entries, ensure_ascii=False, sort_keys=True)


def _review_text_without_ocr(review: dict[str, object]) -> str:
    return render_xmp_review_text(review, include_ocr_text=False)


def _context_payload_hash(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _image_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _artifact_output_path(page_image_path: Path) -> Path:
    output_dir = page_image_path.parent.parent / "_debug" / "verify-crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{page_image_path.stem}.json"


def _read_page_crop_paths(page_image_path: Path) -> list[Path]:
    page_xmp = page_image_path.with_suffix(".xmp")
    if not page_xmp.is_file():
        return []
    try:
        regions = read_region_list(page_xmp, *_image_dimensions(page_image_path))
    except Exception:
        return []
    if not regions:
        return []
    photos_dir = page_image_path.parent.parent / page_image_path.parent.name.replace("_Pages", "_Photos")
    candidates = _expected_crop_output_paths(page_image_path, photos_dir, len(regions))
    return [path for path in candidates if path.is_file()]


def _image_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    with Image.open(str(path)) as image:
        return image.size


def load_page_verifier_inputs(
    page_image_path: Path,
    *,
    geocoder: NominatimGeocoder | None = None,
) -> dict[str, object]:
    page_path = Path(page_image_path)
    page_xmp = page_path.with_suffix(".xmp")
    page_image_exists = page_path.is_file()
    page_xmp_exists = page_xmp.is_file()
    missing_context: list[str] = []
    if not page_image_exists:
        missing_context.append("page_image")
    if not page_xmp_exists:
        missing_context.append("page_xmp")

    page_review = load_ai_xmp_review(page_xmp) if page_xmp_exists else {}
    page_context_text = _review_text_without_ocr(page_review) if isinstance(page_review, dict) else ""
    page_location_text = (
        render_location_verification_text(page_review, geocoder=geocoder) if isinstance(page_review, dict) else ""
    )

    crop_inputs: list[dict[str, object]] = []
    for crop_path in _read_page_crop_paths(page_path):
        crop_xmp = crop_path.with_suffix(".xmp")
        crop_review = load_ai_xmp_review(crop_xmp) if crop_xmp.is_file() else {}
        crop_context_text = _review_text_without_ocr(crop_review) if isinstance(crop_review, dict) else ""
        crop_location_text = (
            render_location_verification_text(crop_review, geocoder=geocoder) if isinstance(crop_review, dict) else ""
        )
        crop_inputs.append(
            {
                "crop_image_path": str(crop_path.resolve()),
                "crop_xmp_path": str(crop_xmp.resolve()),
                "crop_xmp_exists": crop_xmp.is_file(),
                "crop_xmp_text": crop_context_text,
                "crop_location_verification_text": crop_location_text,
            }
        )

    return {
        "page_image_path": str(page_path.resolve()),
        "page_xmp_path": str(page_xmp.resolve()),
        "page_image_exists": page_image_exists,
        "page_xmp_exists": page_xmp_exists,
        "page_xmp_text": page_context_text,
        "page_location_verification_text": page_location_text,
        "missing_context": missing_context,
        "crops": crop_inputs,
    }


def build_verification_prompt(
    *,
    page_image_name: str,
    crop_image_name: str,
    page_xmp_text: str,
    crop_xmp_text: str,
    page_location_verification_text: str = "",
    crop_location_verification_text: str = "",
) -> str:
    parts = [load_prompt("verify-crops/verification/user.md").rendered]
    parts.append(f"Page image file: {page_image_name}")
    parts.append(f"Crop image file: {crop_image_name}")
    parts.append("Page XMP text:")
    parts.append(page_xmp_text or "(missing)")
    parts.append("Crop XMP text:")
    parts.append(crop_xmp_text or "(missing)")
    parts.append("Page location verification evidence:")
    parts.append(page_location_verification_text or "(no GPS coordinates)")
    parts.append("Crop location verification evidence:")
    parts.append(crop_location_verification_text or "(no GPS coordinates)")
    return "\n\n".join(part for part in parts if part).strip()


def build_retry_prompt(*, concern: str, issue: str, problem_to_fix: str) -> str:
    return load_prompt(
        "verify-crops/retry/user.md",
        {
            "concern": str(concern or "").strip(),
            "issue": str(issue or "").strip(),
            "problem_to_fix": str(problem_to_fix or "").strip(),
        },
    ).rendered


def build_parameter_suggestion_prompt(*, concern: str, failure_reason: str) -> str:
    return load_prompt(
        "verify-crops/parameter-suggestion/user.md",
        {
            "concern": str(concern or "").strip(),
            "failure_reason": str(failure_reason or "").strip(),
        },
    ).rendered


def parse_parameter_suggestion_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Parameter suggestion payload is not an object.")
    try:
        return {
            "caption_max_tokens": int(payload.get("caption_max_tokens") or 0),
            "caption_temperature": float(payload.get("caption_temperature") or 0.0),
            "caption_max_edge": int(payload.get("caption_max_edge") or 0),
            "reason": _clean_text(payload.get("reason")),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter suggestion payload is invalid: {payload}") from exc


def _current_crop_xmp_text(crop_image_path: Path) -> str:
    crop_xmp = crop_image_path.with_suffix(".xmp")
    if not crop_xmp.is_file():
        return ""
    review = load_ai_xmp_review(crop_xmp)
    if not isinstance(review, dict):
        return ""
    return _review_text_without_ocr(review)


def _current_crop_location_verification_text(
    crop_image_path: Path,
    *,
    geocoder: NominatimGeocoder | None,
) -> str:
    crop_xmp = crop_image_path.with_suffix(".xmp")
    if not crop_xmp.is_file():
        return ""
    review = load_ai_xmp_review(crop_xmp)
    if not isinstance(review, dict):
        return ""
    return render_location_verification_text(review, geocoder=geocoder)


def _call_structured_vision_payload(
    *,
    page_image_path: Path,
    crop_image_path: Path,
    prompt: str,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    max_image_edge: int,
    response_format: dict[str, object],
    error_label: str,
) -> tuple[dict[str, object], str, str, str]:
    resolved_model = _select_lmstudio_model(base_url, model_name, DEFAULT_LMSTUDIO_TIMEOUT_SECONDS)
    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": verification_system_prompt(),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Supporting page image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _build_data_url(page_image_path, max_image_edge)},
                    },
                    {"type": "text", "text": "Crop under review."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _build_data_url(crop_image_path, max_image_edge)},
                    },
                ],
            },
        ],
        "response_format": response_format,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    response = _lmstudio_request_json(
        f"{base_url}/chat/completions",
        payload=payload,
        timeout=DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    )
    choices = list(response.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    message = dict(choices[0].get("message") or {})
    finish_reason = str(choices[0].get("finish_reason") or "")
    response_text = _format_lmstudio_debug_response(message.get("content"))
    parsed = message.get("content")
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{error_label} response was not valid JSON: {response_text}")
    return parsed, resolved_model, response_text, finish_reason


def _call_structured_vision_request(
    *,
    page_image_path: Path,
    crop_image_path: Path,
    prompt: str,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    max_image_edge: int,
    response_format: dict[str, object],
    error_label: str,
    parser,
) -> tuple[dict[str, object], str, str, str]:
    resolved_model = ""
    response_text = ""
    try:
        payload, resolved_model, response_text, finish_reason = _call_structured_vision_payload(
            page_image_path=page_image_path,
            crop_image_path=crop_image_path,
            prompt=prompt,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_edge=max_image_edge,
            response_format=response_format,
            error_label=error_label,
        )
        return parser(payload), resolved_model, response_text, finish_reason
    except Exception as exc:
        focused_prompt = f"{prompt}\n\nYour previous response failed: {exc}. Please ensure you return valid JSON adhering exactly to the requested schema."
        try:
            payload, resolved_model, response_text, finish_reason = _call_structured_vision_payload(
                page_image_path=page_image_path,
                crop_image_path=crop_image_path,
                prompt=focused_prompt,
                model_name=model_name,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
                response_format=response_format,
                error_label=f"{error_label} Retry",
            )
            return parser(payload), resolved_model, response_text, finish_reason
        except Exception as exc2:
            if parser is parse_parameter_suggestion_payload:
                return {
                    "caption_max_tokens": max_tokens,
                    "caption_temperature": temperature,
                    "caption_max_edge": max_image_edge,
                    "reason": f"Fallback parameters used because suggestion failed: {exc2}",
                }, resolved_model or model_name, response_text, "error"
            return {
                **{
                    name: {
                        "verdict": "uncertain",
                        "reasoning": "Verification failed due to parse or API error.",
                        "failure_reason": str(exc2),
                    }
                    for name in VERIFICATION_CONCERNS
                },
                "human_inference": "Verification failed completely.",
                "needs_another_pass": [],
                "needs_human_review": list(ROUTABLE_CONCERNS),
            }, resolved_model or model_name, response_text, "error"


def _normalize_routing(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text not in ROUTABLE_CONCERNS or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_concern(name: str, payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"Concern '{name}' payload is not an object.")
    verdict = _clean_text(payload.get("verdict")).lower()
    if verdict not in VERDICTS:
        raise ValueError(f"Concern '{name}' has invalid verdict: {verdict!r}")
    reasoning = _clean_text(payload.get("reasoning"))
    if not reasoning:
        raise ValueError(f"Concern '{name}' reasoning is required.")
    failure_reason = _clean_text(payload.get("failure_reason"))
    if verdict == "good":
        failure_reason = ""
    elif not failure_reason:
        failure_reason = reasoning
    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "failure_reason": failure_reason,
    }


def parse_verification_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Verification payload is not an object.")
    normalized = {name: _normalize_concern(name, payload.get(name)) for name in VERIFICATION_CONCERNS}
    human_inference = _clean_text(payload.get("human_inference"))
    if any(normalized[name]["verdict"] in {"bad", "uncertain"} for name in VERIFICATION_CONCERNS):
        if not human_inference:
            raise ValueError("human_inference is required when any concern is bad or uncertain.")
    normalized["human_inference"] = human_inference
    normalized["needs_another_pass"] = _normalize_routing(payload.get("needs_another_pass"))
    normalized["needs_human_review"] = _normalize_routing(payload.get("needs_human_review"))
    good_concerns = {
        name for name in ROUTABLE_CONCERNS if normalized[name]["verdict"] == "good"
    }
    normalized["needs_another_pass"] = [
        name for name in list(normalized["needs_another_pass"]) if name not in good_concerns
    ]
    normalized["needs_human_review"] = [
        name for name in list(normalized["needs_human_review"]) if name not in good_concerns
    ]
    return normalized


def _concern_issue_text(concern_payload: dict[str, object]) -> str:
    return _clean_text(concern_payload.get("reasoning")) or _clean_text(concern_payload.get("failure_reason"))


def _concern_value_text(concern: str, state: dict[str, object]) -> str:
    if concern == "caption":
        return _clean_text(state.get("description"))
    if concern == "date":
        return _clean_text(state.get("dc_date"))
    if concern == "gps":
        lat = _clean_text(state.get("gps_latitude"))
        lon = _clean_text(state.get("gps_longitude"))
        return ", ".join(part for part in (lat, lon) if part)
    if concern == "shown_location":
        return ", ".join(
            part
            for part in (
                _clean_text(state.get("location_sublocation")),
                _clean_text(state.get("location_city")),
                _clean_text(state.get("location_state")),
                _clean_text(state.get("location_country")),
            )
            if part
        )
    return ""


def _location_payload_from_state(state: dict[str, object]) -> dict[str, object]:
    detections = dict(state.get("detections") or {})
    payload = dict(detections.get("location") or {})
    if not payload:
        payload = {
            "query": _clean_text(state.get("location_query")),
            "display_name": _clean_text(state.get("location_display_name")),
            "gps_latitude": _clean_text(state.get("gps_latitude")),
            "gps_longitude": _clean_text(state.get("gps_longitude")),
            "city": _clean_text(state.get("location_city")),
            "state": _clean_text(state.get("location_state")),
            "country": _clean_text(state.get("location_country")),
            "sublocation": _clean_text(state.get("location_sublocation")),
        }
    return {key: value for key, value in payload.items() if _clean_text(value)}


def _write_retry_sidecar(
    *,
    image_path: Path,
    state: dict[str, object],
    description: str | None = None,
    dc_date: str | None = None,
    location_payload: dict[str, object] | None = None,
    detections_payload: dict[str, object] | None = None,
) -> None:
    review = load_ai_xmp_review(image_path.with_suffix(".xmp")) if image_path.with_suffix(".xmp").is_file() else {}
    current_location_payload = _location_payload_from_state(state)
    _write_sidecar_and_record(
        image_path.with_suffix(".xmp"),
        image_path,
        creator_tool=_clean_text(state.get("creator_tool")),
        person_names=list(review.get("person_names") or []),
        subjects=list(review.get("subjects") or []),
        title=_clean_text(state.get("title")),
        title_source=_clean_text(state.get("title_source")),
        description=_clean_text(description if description is not None else state.get("description")),
        album_title=_clean_text(state.get("album_title")),
        location_payload=dict(location_payload or current_location_payload),
        source_text=_clean_text(state.get("source_text")),
        ocr_text=_clean_text(state.get("ocr_text")),
        ocr_lang=_clean_text(state.get("ocr_lang")),
        author_text=_clean_text(state.get("author_text")),
        scene_text=_clean_text(state.get("scene_text")),
        detections_payload=dict(detections_payload if detections_payload is not None else state.get("detections") or {}),
        stitch_key=_clean_text(state.get("stitch_key")),
        ocr_authority_source=_clean_text(state.get("ocr_authority_source")),
        create_date=_clean_text(state.get("create_date")),
        dc_date=_clean_text(dc_date if dc_date is not None else state.get("dc_date")),
        date_time_original=_clean_text(state.get("date_time_original")),
        ocr_ran=bool(state.get("ocr_ran")),
        people_detected=bool(state.get("people_detected")),
        people_identified=bool(state.get("people_identified")),
        title_page_location=None,
    )


def _retry_settings(image_path: Path) -> tuple[dict[str, object], str]:
    runner = IndexRunner(["--photo", str(image_path), "--include-view"])
    effective, _settings_sig, creator_tool, _date_estimation_enabled = runner._resolve_effective_settings(image_path)
    effective["lmstudio_base_url"] = normalize_lmstudio_base_url(
        str(effective.get("lmstudio_base_url") or ""),
        default=default_lmstudio_base_url(),
    )
    return dict(effective), str(creator_tool or "")


def _effective_tuning_params(effective: dict[str, object]) -> dict[str, object]:
    return {
        "caption_max_tokens": int(effective.get("caption_max_tokens") or 96),
        "caption_temperature": float(effective.get("caption_temperature") or 0.2),
        "caption_max_edge": int(effective.get("caption_max_edge") or 0),
    }


def _apply_tuning_params(effective: dict[str, object], tuning_params: dict[str, object] | None) -> dict[str, object]:
    tuned = dict(effective)
    if not isinstance(tuning_params, dict):
        return tuned
    max_tokens = int(tuning_params.get("caption_max_tokens") or 0)
    if max_tokens > 0:
        tuned["caption_max_tokens"] = max_tokens
    max_edge = int(tuning_params.get("caption_max_edge") or 0)
    if max_edge >= 0:
        tuned["caption_max_edge"] = max_edge
    temperature = float(tuning_params.get("caption_temperature") or 0.0)
    if temperature >= 0.0:
        tuned["caption_temperature"] = temperature
    return tuned


def _run_pass2_retry(
    *,
    crop_image_path: Path,
    concern: str,
    issue: str,
    failure_reason: str,
    retry_pass: int = 2,
    prompt_variant: str = "retry",
    tuning_params: dict[str, object] | None = None,
    logger=None,
) -> dict[str, object]:
    state = read_ai_sidecar_state(crop_image_path.with_suffix(".xmp"))
    if not isinstance(state, dict):
        raise RuntimeError(f"Concern retry failed because sidecar state could not be read: {crop_image_path.with_suffix('.xmp')}")
    effective, _creator_tool = _retry_settings(crop_image_path)
    effective = _apply_tuning_params(effective, tuning_params)
    applied_tuning = _effective_tuning_params(effective)
    prompt_prefix = build_retry_prompt(
        concern=concern,
        issue=issue,
        problem_to_fix=failure_reason,
    )
    before_value = _concern_value_text(concern, state)
    if callable(logger):
        logger(f"verify-crops retry pass {retry_pass} {crop_image_path.name} {concern}: {failure_reason}")

    retry_model = ""

    if concern == "caption":
        caption_engine = CaptionEngine(
            engine=str(effective.get("caption_engine") or "lmstudio"),
            model_name=str(effective.get("caption_model") or ""),
            max_tokens=int(effective.get("caption_max_tokens") or 96),
            temperature=float(effective.get("caption_temperature") or 0.2),
            lmstudio_base_url=str(effective.get("lmstudio_base_url") or DEFAULT_LMSTUDIO_BASE_URL),
            max_image_edge=int(effective.get("caption_max_edge") or 0),
            stream=False,
        )
        result = caption_engine.generate(
            crop_image_path,
            people=list(state.get("person_names") or []),
            objects=[
                _clean_text(row.get("label"))
                for row in list((state.get("detections") or {}).get("objects") or [])
                if isinstance(row, dict) and _clean_text(row.get("label"))
            ],
            ocr_text=_clean_text(state.get("ocr_text")),
            source_path=crop_image_path,
            album_title=_clean_text(state.get("album_title")),
            printed_album_title=_clean_text(state.get("album_title")),
            prompt_prefix=prompt_prefix,
        )
        if result.fallback:
            raise RuntimeError(f"Concern retry failed due to caption rerun error: {result.error}")
        retry_model = str(caption_engine.effective_model_name or "")
        detections = dict(state.get("detections") or {})
        caption_meta = dict(detections.get("caption") or {})
        caption_meta["model"] = str(caption_engine.effective_model_name)
        detections["caption"] = caption_meta
        _write_retry_sidecar(
            image_path=crop_image_path,
            state=state,
            description=str(result.text or ""),
            detections_payload=detections,
        )
    elif concern in {"gps", "shown_location"}:
        caption_engine = CaptionEngine(
            engine=str(effective.get("caption_engine") or "lmstudio"),
            model_name=str(effective.get("caption_model") or ""),
            max_tokens=int(effective.get("caption_max_tokens") or 96),
            temperature=float(effective.get("caption_temperature") or 0.2),
            lmstudio_base_url=str(effective.get("lmstudio_base_url") or DEFAULT_LMSTUDIO_BASE_URL),
            max_image_edge=int(effective.get("caption_max_edge") or 0),
            stream=False,
        )
        prompt_debug = PromptDebugSession(crop_image_path, label=crop_image_path.name)
        locations_output = run_locations_step(
            caption_engine=caption_engine,
            image_path=crop_image_path,
            caption_text=_clean_text(state.get("description")),
            ocr_text=_clean_text(state.get("ocr_text")),
            source_path=crop_image_path,
            album_title=_clean_text(state.get("album_title")),
            printed_album_title=_clean_text(state.get("album_title")),
            geocoder=NominatimGeocoder(),
            prompt_debug=prompt_debug,
            prompt_prefix=prompt_prefix,
        )
        if locations_output is None:
            raise RuntimeError(f"Concern retry failed because locations step is not configured for: {crop_image_path}")
        retry_model = str(caption_engine.effective_model_name or "")
        location_payload = dict(locations_output.get("location") or {})
        current_location = _location_payload_from_state(state)
        if concern == "gps":
            for key in ("city", "state", "country", "sublocation"):
                if _clean_text(current_location.get(key)):
                    location_payload[key] = current_location[key]
        else:
            for key in ("gps_latitude", "gps_longitude", "query", "display_name"):
                if _clean_text(current_location.get(key)):
                    location_payload[key] = current_location[key]
        detections = dict(state.get("detections") or {})
        detections["location"] = location_payload
        detections["locations_shown"] = list(locations_output.get("locations_shown") or [])
        detections["location_shown_ran"] = bool(locations_output.get("location_shown_ran"))
        _write_retry_sidecar(
            image_path=crop_image_path,
            state=state,
            location_payload=location_payload,
            detections_payload=detections,
        )
    elif concern == "date":
        date_engine = DateEstimateEngine(
            engine=str(effective.get("caption_engine") or "lmstudio"),
            model_name=str(effective.get("caption_model") or ""),
            lmstudio_base_url=str(effective.get("lmstudio_base_url") or DEFAULT_LMSTUDIO_BASE_URL),
            max_tokens=int(effective.get("caption_max_tokens") or 96),
            temperature=float(effective.get("caption_temperature") or 0.0),
        )
        result = date_engine.estimate(
            ocr_text=_clean_text(state.get("ocr_text")),
            album_title=_clean_text(state.get("album_title")),
            source_path=crop_image_path,
            prompt_prefix=prompt_prefix,
        )
        if result.fallback:
            raise RuntimeError(f"Concern retry failed due to date rerun error: {result.error}")
        retry_model = str(date_engine.effective_model_name or "")
        _write_retry_sidecar(
            image_path=crop_image_path,
            state=state,
            dc_date=str(result.date or ""),
        )
    else:
        raise RuntimeError(f"Unsupported concern retry: {concern}")

    updated_state = read_ai_sidecar_state(crop_image_path.with_suffix(".xmp"))
    if not isinstance(updated_state, dict):
        raise RuntimeError(
            f"Concern retry completed but updated sidecar state could not be read: {crop_image_path.with_suffix('.xmp')}"
        )
    after_value = _concern_value_text(concern, updated_state)
    return {
        "pass": retry_pass,
        "concern": concern,
        "prompt_variant": prompt_variant,
        "issue": issue,
        "failure_reason": failure_reason,
        "before_value": before_value,
        "after_value": after_value,
        "changed": before_value != after_value,
        "model": retry_model,
        "tuning_params": applied_tuning,
    }


def _rerun_verification_for_crop(
    *,
    page_path: Path,
    crop_path: Path,
    page_xmp_text: str,
    page_location_verification_text: str,
    geocoder: NominatimGeocoder | None,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    max_image_edge: int,
    debug_recorder=None,
    verification_pass: str,
) -> tuple[dict[str, object], str, str, str]:
    crop_xmp_text = _current_crop_xmp_text(crop_path)
    crop_location_verification_text = _current_crop_location_verification_text(crop_path, geocoder=geocoder)
    prompt = build_verification_prompt(
        page_image_name=page_path.name,
        crop_image_name=crop_path.name,
        page_xmp_text=page_xmp_text,
        crop_xmp_text=crop_xmp_text,
        page_location_verification_text=page_location_verification_text,
        crop_location_verification_text=crop_location_verification_text,
    )
    parsed, resolved_model, response_text, finish_reason = _call_structured_vision_request(
        page_image_path=page_path,
        crop_image_path=crop_path,
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        max_image_edge=max_image_edge,
        response_format=verification_response_format(),
        error_label="Verification",
        parser=parse_verification_payload,
    )
    emit_prompt_debug(
        debug_recorder,
        step="verify-crops",
        engine="lmstudio",
        model=resolved_model,
        prompt=prompt,
        system_prompt=verification_system_prompt(),
        source_path=crop_path,
        prompt_source="photoalbums/prompts/verify-crops/verification",
        response=response_text,
        finish_reason=finish_reason,
        metadata={
            **_verify_prompt_metadata(
                variant="verification",
                resolved_params={
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "max_image_edge": int(max_image_edge),
                    "timeout_seconds": DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
                },
            ),
            "page_image_path": str(page_path.resolve()),
            "verification_pass": verification_pass,
        },
    )
    return parsed, resolved_model, response_text, finish_reason


def _follow_up_verification_summary(
    *,
    concern: str,
    review: dict[str, object],
    model: str,
) -> dict[str, object]:
    concern_payload = dict(review.get(concern) or {})
    return {
        "concern": concern,
        "verdict": _clean_text(concern_payload.get("verdict")),
        "reasoning": _clean_text(concern_payload.get("reasoning")),
        "failure_reason": _clean_text(concern_payload.get("failure_reason")),
        "review_model": model,
    }


def _update_routing_for_escalation(review: dict[str, object], concern: str) -> dict[str, object]:
    updated = dict(review)
    updated["needs_another_pass"] = [name for name in list(updated.get("needs_another_pass") or []) if name != concern]
    human_review = list(updated.get("needs_human_review") or [])
    if concern not in human_review:
        human_review.append(concern)
    updated["needs_human_review"] = human_review
    return updated


def run_verify_crops_page(
    page_image_path: str | Path,
    *,
    model_name: str = "",
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    max_tokens: int = 512,
    temperature: float = 0.0,
    max_image_edge: int = DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE,
    debug_recorder=None,
    logger=None,
) -> dict[str, object]:
    default_params = _verification_params()
    max_tokens = int(max_tokens if max_tokens is not None else default_params.get("max_tokens", 512))
    temperature = float(temperature if temperature is not None else default_params.get("temperature", 0.0))
    max_image_edge = int(
        max_image_edge if max_image_edge is not None else default_params.get("max_image_edge", DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
    )
    resolved_params = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "max_image_edge": int(max_image_edge),
        "timeout_seconds": float(default_params.get("timeout_seconds", DEFAULT_LMSTUDIO_TIMEOUT_SECONDS)),
    }
    page_path = Path(page_image_path)
    geocoder = NominatimGeocoder()
    inputs = load_page_verifier_inputs(page_path, geocoder=geocoder)
    page_xmp_text = _clean_text(inputs.get("page_xmp_text"))
    page_location_verification_text = _clean_text(inputs.get("page_location_verification_text"))
    missing_context = list(inputs.get("missing_context") or [])
    crop_inputs = list(inputs.get("crops") or [])
    artifact: dict[str, object] = {
        "kind": "verify-crops",
        "page_image_path": str(page_path.resolve()),
        "page_xmp_path": str(Path(str(inputs.get("page_xmp_path") or "")).resolve()),
        "timestamp": xmp_datetime_now(),
        "missing_context": list(missing_context),
        "results": [],
    }
    page_input_hash = _context_payload_hash(
        {
            "page_image": _image_signature(page_path) if page_path.is_file() else None,
            "page_xmp_text": page_xmp_text,
            "page_location_verification_text": page_location_verification_text,
            "page_pipeline_state": read_pipeline_state(page_path.with_suffix(".xmp")),
            "prompt_assets": verify_crops_prompt_hash_payload(),
            "resolved_params": resolved_params,
            "crops": [
                {
                    "crop_image": _image_signature(Path(str(item.get("crop_image_path") or ""))),
                    "crop_xmp_text": _clean_text(item.get("crop_xmp_text")),
                    "crop_location_verification_text": _clean_text(item.get("crop_location_verification_text")),
                }
                for item in crop_inputs
                if Path(str(item.get("crop_image_path") or "")).is_file()
            ],
        }
    )

    if missing_context:
        artifact["summary"] = {
            "status": "missing-context",
            "missing_context": list(missing_context),
        }
        output_path = _artifact_output_path(page_path)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "missing-context",
            "page_input_hash": page_input_hash,
            "missing_context": missing_context,
            "artifact_path": str(output_path),
            "results": [],
        }

    for crop_input in crop_inputs:
        crop_path = Path(str(crop_input.get("crop_image_path") or ""))
        crop_xmp_text = _clean_text(crop_input.get("crop_xmp_text"))
        crop_location_verification_text = _clean_text(crop_input.get("crop_location_verification_text"))
        prompt = build_verification_prompt(
            page_image_name=page_path.name,
            crop_image_name=crop_path.name,
            page_xmp_text=page_xmp_text,
            crop_xmp_text=crop_xmp_text,
            page_location_verification_text=page_location_verification_text,
            crop_location_verification_text=crop_location_verification_text,
        )
        parsed, resolved_model, response_text, finish_reason = _call_structured_vision_request(
            page_image_path=page_path,
            crop_image_path=crop_path,
            prompt=prompt,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_edge=max_image_edge,
            response_format=verification_response_format(),
            error_label="Verification",
            parser=parse_verification_payload,
        )
        emit_prompt_debug(
            debug_recorder,
            step="verify-crops",
            engine="lmstudio",
            model=resolved_model,
            prompt=prompt,
            system_prompt=verification_system_prompt(),
            source_path=crop_path,
            prompt_source="photoalbums/prompts/verify-crops/verification",
            response=response_text,
            finish_reason=finish_reason,
            metadata={
                **_verify_prompt_metadata(variant="verification", resolved_params=resolved_params),
                "page_image_path": str(page_path.resolve()),
            },
        )
        row = {
            "crop_image_path": str(crop_path.resolve()),
            "crop_xmp_path": str(crop_path.with_suffix(".xmp").resolve()),
            "review": parsed,
            "response_text": response_text,
            "model": resolved_model,
            "finish_reason": finish_reason,
        }
        current_review = dict(parsed)
        review_provenance = {
            name: {
                "prompt_variant": "base",
                "model": resolved_model,
                "tuning_params": {},
                "retry_count": 0,
            }
            for name in VERIFICATION_CONCERNS
        }
        retry_attempts: list[dict[str, object]] = []
        for concern in list(parsed.get("needs_another_pass") or []):
            concern_payload = dict(current_review.get(concern) or {})
            failure_reason = _clean_text(concern_payload.get("failure_reason"))
            if not failure_reason:
                continue
            retry_attempt = (
                _run_pass2_retry(
                    crop_image_path=crop_path,
                    concern=concern,
                    issue=_concern_issue_text(concern_payload),
                    failure_reason=failure_reason,
                    retry_pass=2,
                    prompt_variant="retry",
                    logger=logger,
                )
            )
            if callable(logger):
                before_value = _clean_text(retry_attempt.get("before_value")) or "(empty)"
                after_value = _clean_text(retry_attempt.get("after_value")) or "(empty)"
                change_text = "changed" if bool(retry_attempt.get("changed")) else "unchanged"
                logger(
                    f"verify-crops retry pass 2 result {crop_path.name} {concern}: "
                    f"{before_value} -> {after_value} ({change_text})"
                )
            if bool(retry_attempt.get("changed")):
                current_review, follow_up_model, _follow_up_text, _follow_up_finish = _rerun_verification_for_crop(
                    page_path=page_path,
                    crop_path=crop_path,
                    page_xmp_text=page_xmp_text,
                    page_location_verification_text=page_location_verification_text,
                    geocoder=geocoder,
                    model_name=model_name,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_image_edge=max_image_edge,
                    debug_recorder=debug_recorder,
                    verification_pass="follow-up-pass-2",
                )
                retry_attempt["verification"] = _follow_up_verification_summary(
                    concern=concern,
                    review=current_review,
                    model=follow_up_model,
                )
                review_provenance[concern] = {
                    "prompt_variant": "retry",
                    "model": _clean_text(retry_attempt.get("model")),
                    "tuning_params": dict(retry_attempt.get("tuning_params") or {}),
                    "retry_count": 1,
                }
                if callable(logger):
                    verdict = _clean_text((retry_attempt.get("verification") or {}).get("verdict"))
                    if verdict == "good":
                        logger(f"verify-crops follow-up verification {crop_path.name} {concern}: good")
                    else:
                        logger(f"verify-crops follow-up verification {crop_path.name} {concern}: remained unresolved")
            elif callable(logger):
                logger(f"verify-crops follow-up verification {crop_path.name} {concern}: remained unresolved")
            retry_attempts.append(retry_attempt)
            if _clean_text(dict(current_review.get(concern) or {}).get("verdict")) == "good":
                continue

            current_failure_reason = _clean_text(dict(current_review.get(concern) or {}).get("failure_reason")) or failure_reason
            current_issue = _concern_issue_text(dict(current_review.get(concern) or {}))
            current_crop_xmp_text = _current_crop_xmp_text(crop_path)
            current_crop_location_verification_text = _current_crop_location_verification_text(
                crop_path,
                geocoder=geocoder,
            )
            parameter_prompt = build_parameter_suggestion_prompt(
                concern=concern,
                failure_reason=current_failure_reason,
            )
            parameter_context_prompt = "\n\n".join(
                part
                for part in (
                    parameter_prompt,
                    "Page XMP text:",
                    page_xmp_text or "(missing)",
                    "Crop XMP text:",
                    current_crop_xmp_text or "(missing)",
                    "Page location verification evidence:",
                    page_location_verification_text or "(no GPS coordinates)",
                    "Crop location verification evidence:",
                    current_crop_location_verification_text or "(no GPS coordinates)",
                )
                if part
            )
            if callable(logger):
                logger(f"verify-crops parameter suggestion pass 3 {crop_path.name} {concern}: full-context session")
            suggested_params, suggestion_model, suggestion_response_text, suggestion_finish_reason = _call_structured_vision_request(
                page_image_path=page_path,
                crop_image_path=crop_path,
                prompt=parameter_context_prompt,
                model_name=model_name,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
                response_format=parameter_suggestion_response_format(),
                error_label="Parameter suggestion",
                parser=parse_parameter_suggestion_payload,
            )
            emit_prompt_debug(
                debug_recorder,
                step="verify-crops",
                engine="lmstudio",
                model=suggestion_model,
                prompt=parameter_context_prompt,
                system_prompt=verification_system_prompt(),
                source_path=crop_path,
                prompt_source="photoalbums/prompts/verify-crops/parameter-suggestion",
                response=suggestion_response_text,
                finish_reason=suggestion_finish_reason,
                metadata={
                    **_verify_prompt_metadata(variant="parameter-suggestion", resolved_params=resolved_params),
                    "page_image_path": str(page_path.resolve()),
                    "verification_pass": "parameter-suggestion-pass-3",
                },
            )
            if callable(logger):
                logger(
                    "verify-crops selected pass 3 params "
                    f"{crop_path.name} {concern}: "
                    f"max_tokens={suggested_params['caption_max_tokens']}, "
                    f"temperature={suggested_params['caption_temperature']}, "
                    f"max_edge={suggested_params['caption_max_edge']}"
                )
            pass3_attempt = _run_pass2_retry(
                crop_image_path=crop_path,
                concern=concern,
                issue=current_issue,
                failure_reason=current_failure_reason,
                retry_pass=3,
                prompt_variant="parameter-suggestion",
                tuning_params=suggested_params,
                logger=logger,
            )
            pass3_attempt["parameter_suggestion"] = {
                "model": suggestion_model,
                "params": suggested_params,
                "reason": _clean_text(suggested_params.get("reason")),
            }
            if callable(logger):
                before_value = _clean_text(pass3_attempt.get("before_value")) or "(empty)"
                after_value = _clean_text(pass3_attempt.get("after_value")) or "(empty)"
                change_text = "changed" if bool(pass3_attempt.get("changed")) else "unchanged"
                logger(
                    f"verify-crops retry pass 3 result {crop_path.name} {concern}: "
                    f"{before_value} -> {after_value} ({change_text})"
                )
            if bool(pass3_attempt.get("changed")):
                current_review, follow_up_model, _follow_up_text, _follow_up_finish = _rerun_verification_for_crop(
                    page_path=page_path,
                    crop_path=crop_path,
                    page_xmp_text=page_xmp_text,
                    page_location_verification_text=page_location_verification_text,
                    geocoder=geocoder,
                    model_name=model_name,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_image_edge=max_image_edge,
                    debug_recorder=debug_recorder,
                    verification_pass="follow-up-pass-3",
                )
                pass3_attempt["verification"] = _follow_up_verification_summary(
                    concern=concern,
                    review=current_review,
                    model=follow_up_model,
                )
                if callable(logger):
                    verdict = _clean_text((pass3_attempt.get("verification") or {}).get("verdict"))
                    if verdict == "good":
                        logger(f"verify-crops follow-up verification {crop_path.name} {concern}: good")
                    else:
                        logger(f"verify-crops follow-up verification {crop_path.name} {concern}: remained unresolved")
            elif callable(logger):
                logger(f"verify-crops follow-up verification {crop_path.name} {concern}: remained unresolved")
            review_provenance[concern] = {
                "prompt_variant": "parameter-suggestion",
                "model": _clean_text(pass3_attempt.get("model")),
                "tuning_params": dict(pass3_attempt.get("tuning_params") or {}),
                "retry_count": 2,
            }
            retry_attempts.append(pass3_attempt)
            if _clean_text(dict(current_review.get(concern) or {}).get("verdict")) != "good":
                current_review = _update_routing_for_escalation(current_review, concern)
        if retry_attempts:
            row["retry_attempts"] = retry_attempts
            row["initial_review"] = parsed
        row["review"] = current_review
        row["review_provenance"] = review_provenance
        artifact["results"].append(row)

    artifact["summary"] = {
        "status": "ok",
        "crop_count": len(artifact["results"]),
        "retry_attempt_count": sum(len(list(row.get("retry_attempts") or [])) for row in list(artifact["results"])),
        "bad_or_uncertain": [
            {
                "crop_image_path": row["crop_image_path"],
                "needs_another_pass": row["review"]["needs_another_pass"],
                "needs_human_review": row["review"]["needs_human_review"],
            }
            for row in list(artifact["results"])
            if any(
                row["review"][name]["verdict"] in {"bad", "uncertain"}
                for name in VERIFICATION_CONCERNS
            )
        ],
    }
    output_path = _artifact_output_path(page_path)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "page_input_hash": page_input_hash,
        "missing_context": [],
        "artifact_path": str(output_path),
        "results": list(artifact["results"]),
    }


def persist_verify_crops_state(
    page_image_path: str | Path,
    verify_result: dict[str, object],
) -> None:
    page_path = Path(page_image_path)
    page_xmp = page_path.with_suffix(".xmp")
    crop_results = list(verify_result.get("results") or [])
    page_entry = {
        "timestamp": xmp_datetime_now(),
        "input_hash": str(verify_result.get("page_input_hash") or ""),
        "result": "ok" if str(verify_result.get("status") or "") == "ok" else "missing-context",
        "page_verification_ran": True,
        "missing_context": list(verify_result.get("missing_context") or []),
        "artifact_path": _clean_text(verify_result.get("artifact_path")),
        "reviewed_crop_count": len(crop_results),
        "needs_another_pass": sorted(
            {
                concern
                for row in crop_results
                for concern in list((row.get("review") or {}).get("needs_another_pass") or [])
            }
        ),
        "needs_human_review": sorted(
            {
                concern
                for row in crop_results
                for concern in list((row.get("review") or {}).get("needs_human_review") or [])
            }
        ),
    }
    write_pipeline_steps(page_xmp, {"verify-crops": page_entry})

    for row in crop_results:
        crop_xmp = Path(str(row.get("crop_xmp_path") or ""))
        review = dict(row.get("review") or {})
        review_provenance = dict(row.get("review_provenance") or {})
        concern_states = {
            name: {
                "status": review[name]["verdict"],
                "reasoning": review[name]["reasoning"],
                "failure_reason": review[name]["failure_reason"],
                "provenance": dict(review_provenance.get(name) or {}),
            }
            for name in ROUTABLE_CONCERNS
            if isinstance(review.get(name), dict)
        }
        concern_states["overall"] = {
            "status": review["overall"]["verdict"],
            "reasoning": review["overall"]["reasoning"],
            "failure_reason": review["overall"]["failure_reason"],
            "provenance": dict(review_provenance.get("overall") or {}),
        }
        write_pipeline_steps(
            crop_xmp,
            {
                "verify-crops": {
                    "timestamp": xmp_datetime_now(),
                    "input_hash": str(verify_result.get("page_input_hash") or ""),
                    "result": "ok",
                    "page_verification_ran": True,
                    "artifact_path": _clean_text(verify_result.get("artifact_path")),
                    "human_inference": _clean_text(review.get("human_inference")),
                    "needs_another_pass": list(review.get("needs_another_pass") or []),
                    "needs_human_review": list(review.get("needs_human_review") or []),
                    "concerns": concern_states,
                }
            },
        )
