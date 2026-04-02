from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._caption_text import clean_text, clean_lines
from .image_limits import allow_large_pillow_images
from ._lmstudio_helpers import LMStudioModelResolverMixin
from ._prompt_skill import required_section_text

DEFAULT_LMSTUDIO_MAX_NEW_TOKENS = 8129
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LMSTUDIO_TIMEOUT_SECONDS = 300.0
DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE = 2048
LMSTUDIO_VISION_MODEL_HINTS = (
    "vl",
    "vision",
    "llava",
    "minicpm",
    "moondream",
    "pixtral",
    "internvl",
    "phi-3.5-vision",
    "phi-4-multimodal",
    "qvq",
)


@dataclass(frozen=True)
class CaptionDetails:
    text: str
    ocr_text: str = ""
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""
    author_text: str = ""
    scene_text: str = ""
    people_present: bool = False
    estimated_people_count: int = 0
    name_suggestions: list[dict[str, object]] = None
    image_regions: list[dict[str, object]] = None
    album_title: str = ""
    title: str = ""
    ocr_lang: str = ""
    locations_shown: list[dict[str, object]] = None

    def __post_init__(self):
        if self.name_suggestions is None:
            object.__setattr__(self, "name_suggestions", [])
        if self.image_regions is None:
            object.__setattr__(self, "image_regions", [])
        if self.locations_shown is None:
            object.__setattr__(self, "locations_shown", [])

    def __str__(self) -> str:
        return self.text

    def __contains__(self, item: object) -> bool:
        return str(item or "") in self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CaptionDetails):
            return (
                self.text == other.text
                and self.ocr_text == other.ocr_text
                and self.gps_latitude == other.gps_latitude
                and self.gps_longitude == other.gps_longitude
                and self.location_name == other.location_name
                and self.author_text == other.author_text
                and self.scene_text == other.scene_text
                and self.people_present == other.people_present
                and self.estimated_people_count == other.estimated_people_count
                and self.name_suggestions == other.name_suggestions
                and self.image_regions == other.image_regions
                and self.album_title == other.album_title
                and self.title == other.title
            )
        if isinstance(other, str):
            return self.text == other
        return False


_CAPTION_REASONING_MARKERS = (
    "the user wants",
    "analyze the input data",
    "visual analysis",
    "synthesize the visual content",
    "filename hint",
    "folder hint",
    "album title hint",
    "ocr/text in image",
    "ocr text hint",
    "detected objects",
    "album classification hint",
    "album focus hint",
    "cordell photo albums rules",
    "based on the rules",
    "i need to",
    "i should",
)


def _looks_like_reasoning_or_prompt_echo(value: str) -> bool:
    """Return True if the text looks like a model reasoning trace or prompt echo rather than a caption."""
    text = clean_text(value)
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _CAPTION_REASONING_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if text.startswith(("**1.", "1.", "* **filename:**", "- **filename:**")):
        return True
    if re.search(
        r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:filename|folder|ocr/text in image|detected objects)\s*:",
        lowered,
    ):
        return True
    if re.search(
        r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:album classification hint|album focus hint)\s*:",
        lowered,
    ):
        return True
    return False


def _truncate_long_decimals(text: str) -> str:
    """Truncate absurdly long decimal fractions to prevent JSON parse failures.

    LM Studio occasionally generates repeating decimals (e.g. 0.29782608695652282608...)
    for numeric fields, exhausting the token budget and leaving the JSON truncated or
    so long that json.loads raises 'Expecting delimiter'. Six decimal places is more
    than enough precision for normalised image coordinates.
    """
    return re.sub(r"(\d+\.\d{6})\d+", r"\1", text)


def _iter_structured_json_payloads(text: str):
    raw = _truncate_long_decimals(str(text or "").strip())
    if not raw:
        return
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _extract_structured_json_payload(
    text: str,
    *,
    is_valid=None,
) -> dict[str, object] | None:
    fallback: dict[str, object] | None = None
    matched: dict[str, object] | None = None
    for payload in _iter_structured_json_payloads(text):
        fallback = payload
        if is_valid is not None and bool(is_valid(payload)):
            matched = payload
    return matched if is_valid is not None else fallback


def _lanczos_resize(image, new_size: tuple[int, int]):
    resampling = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resampling is None:
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        except Exception:  # pragma: no cover - Pillow always present in runtime
            resampling = 1
    return image.resize(new_size, resampling)


def _resize_caption_image(image, max_image_edge: int):
    if int(max_image_edge) <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= int(max_image_edge):
        return image
    scale = float(max_image_edge) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return _lanczos_resize(image, new_size)


def _format_decimal_coordinate(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_dms_coordinate(value: str) -> float | None:
    match = re.search(
        r"(?P<deg>\d{1,3})\s*[°º]\s*(?P<min>\d{1,2})\s*[′']\s*(?P<sec>\d{1,2}(?:\.\d+)?)?\s*[″\"]?\s*(?P<hem>[NSEW])",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    deg = float(match.group("deg"))
    minutes = float(match.group("min"))
    seconds = float(match.group("sec") or 0.0)
    hemisphere = str(match.group("hem") or "").upper()
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _parse_decimal_coordinate(value: str, *, positive_hemisphere: str, negative_hemisphere: str) -> float | None:
    text = clean_text(value).replace("−", "-")
    if not text:
        return None
    match = re.fullmatch(
        rf"(?P<number>[+-]?\d{{1,3}}(?:\.\d+)?)\s*(?P<hem>[{positive_hemisphere}{negative_hemisphere}])?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    decimal = float(match.group("number"))
    hemisphere = str(match.group("hem") or "").upper()
    if hemisphere == negative_hemisphere.upper():
        decimal = -abs(decimal)
    elif hemisphere == positive_hemisphere.upper():
        decimal = abs(decimal)
    return decimal


def _normalize_gps_value(value: str, *, axis: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    decimal: float | None = None
    if any(symbol in text for symbol in ("°", "º", "′", "″", "'")):
        decimal = _parse_dms_coordinate(text)
    if decimal is None:
        if axis == "lat":
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="N", negative_hemisphere="S")
        else:
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="E", negative_hemisphere="W")
    if decimal is None:
        return ""
    return _format_decimal_coordinate(decimal)


def _build_data_url(image_path: str | Path, max_image_edge: int) -> str:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    path = Path(image_path)
    image = Image.open(str(path)).convert("RGB")
    try:
        working_image = _resize_caption_image(image, max_image_edge)
        buffer = io.BytesIO()
        if path.suffix.lower() == ".png":
            mime = "image/png"
            working_image.save(buffer, format="PNG")
        else:
            mime = "image/jpeg"
            working_image.save(buffer, format="JPEG", quality=95)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{data}"
    finally:
        if "working_image" in locals() and working_image is not image:
            working_image.close()
        image.close()


def _decode_lmstudio_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _format_lmstudio_debug_response(value: object) -> str:
    if isinstance(value, (str, list)):
        return _decode_lmstudio_text(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _extract_lmstudio_error_message(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    direct_message = str(payload.get("message") or "").strip()
    error = payload.get("error")
    if isinstance(error, dict):
        nested_message = str(error.get("message") or "").strip()
        if nested_message:
            return nested_message
    elif isinstance(error, str):
        error_text = str(error).strip()
        if error_text:
            return error_text
    return direct_message


def _is_lmstudio_caption_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(
        isinstance(payload.get(field, ""), str) for field in ("ocr_text", "author_text", "scene_text", "location_name")
    )


def _is_lmstudio_people_count_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    people_present = payload.get("people_present")
    estimated_people_count = payload.get("estimated_people_count")
    if not isinstance(people_present, bool):
        return False
    if isinstance(estimated_people_count, bool):
        return False
    try:
        return int(estimated_people_count) >= 0
    except Exception:
        return False


def _is_lmstudio_location_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    location_name = payload.get("location_name")
    gps_latitude = payload.get("gps_latitude")
    gps_longitude = payload.get("gps_longitude")
    return isinstance(location_name, str) and isinstance(gps_latitude, str) and isinstance(gps_longitude, str)


def _lmstudio_caption_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "caption_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ocr_text": {"type": "string"},
                    "author_text": {"type": "string"},
                    "scene_text": {"type": "string"},
                    "location_name": {"type": "string"},
                    "album_title": {"type": "string"},
                    "ocr_lang": {"type": "string"},
                },
                "required": [
                    "ocr_text",
                    "author_text",
                    "scene_text",
                    "location_name",
                    "album_title",
                    "ocr_lang",
                ],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_location_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "location_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "location_name": {"type": "string"},
                    "gps_latitude": {"type": "string"},
                    "gps_longitude": {"type": "string"},
                },
                "required": [
                    "location_name",
                    "gps_latitude",
                    "gps_longitude",
                ],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_locations_shown_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "locations_shown_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "locations_shown": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "world_region": {"type": "string"},
                                "country_name": {"type": "string"},
                                "country_code": {"type": "string"},
                                "province_or_state": {"type": "string"},
                                "city": {"type": "string"},
                                "sublocation": {"type": "string"},
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["locations_shown"],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_people_count_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "people_count_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "people_present": {"type": "boolean"},
                    "estimated_people_count": {
                        "type": "integer",
                        "minimum": 0,
                    },
                },
                "required": [
                    "people_present",
                    "estimated_people_count",
                ],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_page_caption_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "page_caption_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ocr_text": {"type": "string"},
                    "author_text": {"type": "string"},
                    "scene_text": {"type": "string"},
                    "location_name": {"type": "string"},
                    "photo_regions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "w": {"type": "number"},
                                "h": {"type": "number"},
                                "author_text": {"type": "string"},
                                "scene_text": {"type": "string"},
                            },
                            "required": ["x", "y", "w", "h", "author_text", "scene_text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "ocr_text",
                    "author_text",
                    "scene_text",
                    "location_name",
                    "photo_regions",
                ],
                "additionalProperties": False,
            },
        },
    }


def _parse_image_regions(payload: dict) -> list[dict]:
    """Extract and validate photo_regions from a parsed payload dict.

    Clamps x/y/w/h to [0, 1] and discards any entry with w<=0 or h<=0.
    """
    raw = list(payload.get("photo_regions") or [])
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x = max(0.0, min(1.0, float(item.get("x") or 0)))
            y = max(0.0, min(1.0, float(item.get("y") or 0)))
            w = max(0.0, min(1.0, float(item.get("w") or 0)))
            h = max(0.0, min(1.0, float(item.get("h") or 0)))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        result.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "author_text": clean_text(str(item.get("author_text") or "")),
                "scene_text": clean_text(str(item.get("scene_text") or "")),
            }
        )
    return result


def _lmstudio_error_preview(value: str, *, limit: int = 180) -> str:
    text = clean_text(value)
    if not text:
        return "<empty>"
    snippet = text[: max(1, int(limit))]
    if len(text) > len(snippet):
        snippet += "..."
    return snippet


def _parse_lmstudio_structured_caption_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> tuple[str, str, str, str, str, str, bool, int, list[dict[str, object]], str]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured caption content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(
            text,
            is_valid=_is_lmstudio_caption_payload,
        )
        if payload is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid structured caption JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not isinstance(payload, dict):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio returned structured caption JSON that is not an object; raw={preview!r}.{finish_note}"
        )
    ocr_text = payload.get("ocr_text")
    if not isinstance(ocr_text, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio structured caption JSON is missing an ocr_text string; raw={preview!r}.{finish_note}"
        )
    author_text = payload.get("author_text")
    if not isinstance(author_text, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio structured caption JSON is missing an author_text string; raw={preview!r}.{finish_note}"
        )
    scene_text = clean_lines(str(payload.get("scene_text") or ""))
    gps_latitude = _normalize_gps_value(str(payload.get("gps_latitude") or ""), axis="lat")
    gps_longitude = _normalize_gps_value(str(payload.get("gps_longitude") or ""), axis="lon")
    location_name = clean_text(str(payload.get("location_name") or ""))
    people_present = bool(payload.get("people_present") or False)
    try:
        estimated_people_count = max(0, int(payload.get("estimated_people_count") or 0))
    except Exception:
        estimated_people_count = 0
    name_suggestions = list(payload.get("name_suggestions") or [])
    album_title = clean_text(str(payload.get("album_title") or ""))
    return (
        str(ocr_text),
        clean_lines(author_text),
        scene_text,
        gps_latitude,
        gps_longitude,
        location_name,
        people_present,
        estimated_people_count,
        name_suggestions,
        album_title,
    )


def _parse_lmstudio_structured_people_count_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> tuple[bool, int]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured people-count content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(
            text,
            is_valid=_is_lmstudio_people_count_payload,
        )
        if payload is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid structured people-count JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not _is_lmstudio_people_count_payload(payload):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(f"LM Studio structured people-count JSON is invalid; raw={preview!r}.{finish_note}")
    return (
        bool(payload.get("people_present")),
        max(0, int(payload.get("estimated_people_count") or 0)),
    )


def _parse_lmstudio_structured_location_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> tuple[str, str, str]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured location content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(
            text,
            is_valid=_is_lmstudio_location_payload,
        )
        if payload is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid structured location JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not _is_lmstudio_location_payload(payload):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(f"LM Studio structured location JSON is invalid; raw={preview!r}.{finish_note}")
    return (
        _normalize_gps_value(str(payload.get("gps_latitude") or ""), axis="lat"),
        _normalize_gps_value(str(payload.get("gps_longitude") or ""), axis="lon"),
        clean_text(str(payload.get("location_name") or "")),
    )


def _is_lmstudio_locations_shown_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    locations = payload.get("locations_shown")
    if not isinstance(locations, list):
        return False
    for loc in locations:
        if not isinstance(loc, dict):
            return False
    return True


def _parse_lmstudio_locations_shown_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> list[dict[str, Any]]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(
            text,
            is_valid=_is_lmstudio_locations_shown_payload,
        )
        if payload is None:
            return []
    if not _is_lmstudio_locations_shown_payload(payload):
        return []
    locations = payload.get("locations_shown") or []
    result: list[dict[str, Any]] = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        result.append(
            {
                "name": str(loc.get("name") or "").strip(),
                "world_region": str(loc.get("world_region") or "").strip(),
                "country_name": str(loc.get("country_name") or "").strip(),
                "country_code": str(loc.get("country_code") or "").strip(),
                "province_or_state": str(loc.get("province_or_state") or "").strip(),
                "city": str(loc.get("city") or "").strip(),
                "sublocation": str(loc.get("sublocation") or "").strip(),
            }
        )
    return result


def _parse_lmstudio_structured_caption(
    value: object,
    *,
    finish_reason: str = "",
) -> CaptionDetails:
    (
        ocr_text,
        author_text,
        scene_text,
        gps_latitude,
        gps_longitude,
        location_name,
        people_present,
        estimated_people_count,
        name_suggestions,
        album_title,
    ) = _parse_lmstudio_structured_caption_payload(value, finish_reason=finish_reason)
    return CaptionDetails(
        text=author_text,
        ocr_text=str(ocr_text),
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_name=location_name,
        author_text=author_text,
        scene_text=scene_text,
        people_present=people_present,
        estimated_people_count=estimated_people_count,
        name_suggestions=name_suggestions,
        album_title=album_title,
    )


def _parse_lmstudio_page_caption(
    value: object,
    *,
    finish_reason: str = "",
) -> CaptionDetails:
    """Parse a page-caption response (includes photo_regions) into CaptionDetails."""
    raw = _decode_lmstudio_text(value)
    text = _truncate_long_decimals(str(raw or "").strip())
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(f"LM Studio returned empty page caption content.{finish_note}")
    try:
        payload_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        payload_dict = _extract_structured_json_payload(text, is_valid=_is_lmstudio_caption_payload)
        if payload_dict is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid page caption JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not isinstance(payload_dict, dict):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(f"LM Studio page caption JSON is not an object; raw={preview!r}.{finish_note}")
    ocr_text = payload_dict.get("ocr_text")
    if not isinstance(ocr_text, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(f"LM Studio page caption JSON is missing an ocr_text string; raw={preview!r}.{finish_note}")
    author_text = payload_dict.get("author_text")
    if not isinstance(author_text, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio page caption JSON is missing an author_text string; raw={preview!r}.{finish_note}"
        )
    scene_text = clean_lines(str(payload_dict.get("scene_text") or ""))
    location_name = clean_text(str(payload_dict.get("location_name") or ""))
    image_regions = _parse_image_regions(payload_dict)
    return CaptionDetails(
        text=clean_lines(author_text),
        ocr_text=str(ocr_text),
        location_name=location_name,
        author_text=clean_lines(author_text),
        scene_text=scene_text,
        image_regions=image_regions,
    )


def _lmstudio_request_json(url: str, *, payload: dict | None = None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or f"HTTP {exc.code}"
        raise RuntimeError(f"LM Studio request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _lmstudio_stream_tokens(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            current_event = ""
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("event: "):
                    current_event = line[7:].strip().lower()
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    if current_event == "error":
                        raise RuntimeError(f"LM Studio request failed: {data}")
                    continue
                error_message = _extract_lmstudio_error_message(chunk)
                if current_event == "error" or (error_message and not list(chunk.get("choices") or [])):
                    raise RuntimeError(f"LM Studio request failed: {error_message}")
                choices = list(chunk.get("choices") or [])
                if not choices:
                    continue
                delta = dict(choices[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    yield str(content)
                current_event = ""
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or f"HTTP {exc.code}"
        raise RuntimeError(f"LM Studio request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _select_lmstudio_model(base_url: str, requested_model: str, timeout: float) -> str:
    text = str(requested_model or "").strip()
    if text:
        return text
    payload = _lmstudio_request_json(f"{base_url}/models", timeout=timeout)
    model_ids = [
        str(row.get("id") or "").strip() for row in list(payload.get("data") or []) if str(row.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError("LM Studio did not return any models. Load a model or pass --caption-model.")
    for model_id in model_ids:
        lowered = model_id.casefold()
        if any(hint in lowered for hint in LMSTUDIO_VISION_MODEL_HINTS):
            return model_id
    return model_ids[0]


def normalize_lmstudio_base_url(value: str, default: str = DEFAULT_LMSTUDIO_BASE_URL) -> str:
    text = str(value or "").strip() or str(default or DEFAULT_LMSTUDIO_BASE_URL)
    text = text.rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


_DESCRIBE_CONFIGS: dict[str, tuple] = {
    "photo": (_lmstudio_caption_response_format, _parse_lmstudio_structured_caption),
    "page": (_lmstudio_page_caption_response_format, _parse_lmstudio_page_caption),
}


def people_count_system_prompt() -> str:
    return required_section_text("System Prompt - People Count")


def location_system_prompt() -> str:
    return required_section_text("System Prompt - Location")


def location_shown_system_prompt() -> str:
    return required_section_text("System Prompt - Location Shown")


class LMStudioCaptioner(LMStudioModelResolverMixin):
    _select_model_name = staticmethod(_select_lmstudio_model)

    def __init__(
        self,
        *,
        model_name: str = "",
        prompt_text: str = "",
        max_new_tokens: int = DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
        temperature: float = 0.2,
        base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        timeout_seconds: float = DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
        max_image_edge: int = 0,
        stream: bool = False,
    ):
        self.model_name = str(model_name or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.base_url = normalize_lmstudio_base_url(base_url)
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_image_edge = max(0, int(max_image_edge))
        self.stream = bool(stream)
        self._resolved_model_name = ""
        self.last_response_text = ""
        self.last_finish_reason = ""

    def _call_chat_completion(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        system_prompt: str,
        response_format: dict,
        parse_fn: Callable,
    ) -> CaptionDetails:
        self.last_response_text = ""
        self.last_finish_reason = ""
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "response_format": response_format,
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "stream": self.stream,
        }
        if self.stream:
            print(
                f"  Running LM Studio model ({self._resolve_model_name()})...",
                end="",
                flush=True,
            )
            tokens: list[str] = []
            for token in _lmstudio_stream_tokens(
                f"{self.base_url}/chat/completions",
                payload,
                self.timeout_seconds,
            ):
                tokens.append(token)
            print("\r\033[K", end="", flush=True)
            self.last_response_text = "".join(tokens)
            return parse_fn(self.last_response_text)
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        self.last_finish_reason = str(choices[0].get("finish_reason") or "")
        self.last_response_text = _format_lmstudio_debug_response(message.get("content"))
        if not self.last_response_text:
            self.last_response_text = _format_lmstudio_debug_response(message)
        return parse_fn(
            message.get("content"),
            finish_reason=self.last_finish_reason,
        )

    def _describe_by_mode(self, image_path: str | Path, *, prompt: str, mode: str) -> CaptionDetails:
        fmt_fn, parse_fn = _DESCRIBE_CONFIGS[mode]
        return self._call_chat_completion(
            image_path,
            prompt=prompt,
            system_prompt="",
            response_format=fmt_fn(),
            parse_fn=parse_fn,
        )

    def describe(self, image_path: str | Path, *, prompt: str) -> CaptionDetails:
        return self._describe_by_mode(image_path, prompt=prompt, mode="photo")

    def describe_page(self, image_path: str | Path, *, prompt: str) -> CaptionDetails:
        """Describe a multi-photo album page, returning per-photo regions in image_regions."""
        return self._describe_by_mode(image_path, prompt=prompt, mode="page")

    def estimate_people(
        self,
        image_path: str | Path,
        *,
        prompt: str,
    ) -> CaptionDetails:
        self.last_response_text = ""
        self.last_finish_reason = ""
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": people_count_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "response_format": _lmstudio_people_count_response_format(),
            "max_tokens": min(48, int(self.max_new_tokens)),
            "temperature": 0.0,
            "stream": False,
        }
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        self.last_finish_reason = str(choices[0].get("finish_reason") or "")
        self.last_response_text = _format_lmstudio_debug_response(message.get("content"))
        if not self.last_response_text:
            self.last_response_text = _format_lmstudio_debug_response(message)
        people_present, estimated_people_count = _parse_lmstudio_structured_people_count_payload(
            message.get("content"),
            finish_reason=self.last_finish_reason,
        )
        return CaptionDetails(
            text="",
            people_present=people_present,
            estimated_people_count=estimated_people_count,
        )

    def estimate_location(
        self,
        image_path: str | Path,
        *,
        prompt: str,
    ) -> CaptionDetails:
        self.last_response_text = ""
        self.last_finish_reason = ""
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": location_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "response_format": _lmstudio_location_response_format(),
            "max_tokens": min(96, int(self.max_new_tokens)),
            "temperature": 0.0,
            "stream": False,
        }
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        self.last_finish_reason = str(choices[0].get("finish_reason") or "")
        self.last_response_text = _format_lmstudio_debug_response(message.get("content"))
        if not self.last_response_text:
            self.last_response_text = _format_lmstudio_debug_response(message)
        gps_latitude, gps_longitude, location_name = _parse_lmstudio_structured_location_payload(
            message.get("content"),
            finish_reason=self.last_finish_reason,
        )
        return CaptionDetails(
            text="",
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_name=location_name,
        )

    def estimate_locations_shown(
        self,
        image_path: str | Path,
        *,
        prompt: str,
    ) -> CaptionDetails:
        self.last_response_text = ""
        self.last_finish_reason = ""
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)

        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": location_shown_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "response_format": _lmstudio_locations_shown_response_format(),
            "max_tokens": min(256, int(self.max_new_tokens)),
            "temperature": 0.0,
            "stream": False,
        }
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        self.last_finish_reason = str(choices[0].get("finish_reason") or "")
        self.last_response_text = _format_lmstudio_debug_response(message.get("content"))
        if not self.last_response_text:
            self.last_response_text = _format_lmstudio_debug_response(message)
        locations_shown = _parse_lmstudio_locations_shown_payload(
            message.get("content"),
            finish_reason=self.last_finish_reason,
        )
        return CaptionDetails(
            text="",
            locations_shown=locations_shown,
        )
