from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .ai_caption import _normalize_gps_value
from .ai_geocode import NominatimGeocoder
from .prompt_debug import PromptDebugSession

_COORDINATE_LABEL_RE = re.compile(
    r"\b(?P<label>lat(?:itude)?|lon(?:gitude)?|long)\b\s*[:=]?\s*"
    r"(?P<value>.+?)(?=(?:\b(?:lat(?:itude)?|lon(?:gitude)?|long)\b)|[\n\r;]|$)",
    flags=re.IGNORECASE,
)
_COORDINATE_HEMISPHERE_RE = re.compile(
    r"(?:\d{1,3}(?:\.\d+)?\s*[NSEW])"
    r"|(?:\d{1,3}\s*[°º]\s*\d{1,2}\s*[′']\s*\d{1,2}(?:\.\d+)?\s*[″\"]?\s*[NSEW])",
    flags=re.IGNORECASE,
)


def _xmp_gps_to_decimal(value: object, *, axis: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "," not in text or len(text) < 2:
        return _normalize_gps_value(text, axis=axis)
    hemisphere = text[-1:].upper()
    body = text[:-1]
    if axis == "lat" and hemisphere not in {"N", "S"}:
        return _normalize_gps_value(text, axis=axis)
    if axis == "lon" and hemisphere not in {"E", "W"}:
        return _normalize_gps_value(text, axis=axis)
    degrees_text, minutes_text = body.split(",", 1)
    try:
        degrees = int(degrees_text.strip())
        minutes = float(minutes_text.strip())
    except ValueError:
        return _normalize_gps_value(text, axis=axis)
    decimal = float(degrees) + (minutes / 60.0)
    if hemisphere in {"S", "W"}:
        decimal = -decimal
    return f"{decimal:.8f}".rstrip("0").rstrip(".")


def _extract_explicit_gps_from_text(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""

    lat_text = ""
    lon_text = ""
    for match in _COORDINATE_LABEL_RE.finditer(raw):
        label = str(match.group("label") or "").casefold()
        axis = "lat" if label.startswith("lat") else "lon"
        value = _normalize_gps_value(str(match.group("value") or ""), axis=axis)
        if not value:
            continue
        if axis == "lat" and not lat_text:
            lat_text = value
        if axis == "lon" and not lon_text:
            lon_text = value
        if lat_text and lon_text:
            return lat_text, lon_text

    for match in _COORDINATE_HEMISPHERE_RE.finditer(raw):
        value = str(match.group(0) or "").strip()
        if not value:
            continue
        upper_value = value.upper()
        if any(marker in upper_value for marker in ("N", "S")) and not lat_text:
            lat_text = _normalize_gps_value(value, axis="lat")
        if any(marker in upper_value for marker in ("E", "W")) and not lon_text:
            lon_text = _normalize_gps_value(value, axis="lon")
        if lat_text and lon_text:
            return lat_text, lon_text

    return ("", "") if not (lat_text and lon_text) else (lat_text, lon_text)


def _merge_location_estimates(
    *,
    local_gps_latitude: str,
    local_gps_longitude: str,
    model_gps_latitude: str,
    model_gps_longitude: str,
    model_location_name: str,
) -> tuple[str, str, str]:
    lat_text = str(local_gps_latitude or "").strip()
    lon_text = str(local_gps_longitude or "").strip()
    if lat_text and lon_text:
        return lat_text, lon_text, str(model_location_name or "").strip()
    model_lat = str(model_gps_latitude or "").strip()
    model_lon = str(model_gps_longitude or "").strip()
    return model_lat, model_lon, str(model_location_name or "").strip()


def _resolve_location_metadata(
    *,
    requested_caption_engine: str,
    caption_engine: Any,
    model_image_path: Path,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: Path,
    album_title: str,
    printed_album_title: str,
    people_positions: dict[str, str],
    fallback_location_name: str,
    prompt_debug: PromptDebugSession | None = None,
    debug_step: str = "location",
) -> tuple[str, str, str]:
    local_gps_latitude, local_gps_longitude = _extract_explicit_gps_from_text(ocr_text)
    if str(requested_caption_engine or "").strip().lower() != "lmstudio":
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    estimate_location = getattr(caption_engine, "estimate_location", None)
    if not callable(estimate_location):
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    try:
        result = estimate_location(
            image_path=model_image_path,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step=debug_step,
        )
    except Exception:
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    fallback = getattr(result, "fallback", False)
    if not isinstance(fallback, bool) or fallback:
        return (
            local_gps_latitude,
            local_gps_longitude,
            str(fallback_location_name or "").strip(),
        )
    return _merge_location_estimates(
        local_gps_latitude=local_gps_latitude,
        local_gps_longitude=local_gps_longitude,
        model_gps_latitude=str(getattr(result, "gps_latitude", "") or "").strip(),
        model_gps_longitude=str(getattr(result, "gps_longitude", "") or "").strip(),
        model_location_name=(
            str(getattr(result, "location_name", "") or "").strip() or str(fallback_location_name or "").strip()
        ),
    )


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
    # Reject generic place-type descriptions (e.g. "a beach", "a park") — they
    # are not named places and produce spurious Nominatim results.
    if re.match(r"^(?:a|an)\s+\S", query, re.IGNORECASE):
        query = ""
    if lat_text and lon_text:
        payload: dict[str, Any] = {
            "gps_latitude": float(lat_text),
            "gps_longitude": float(lon_text),
            "map_datum": "WGS-84",
            "source": "caption",
        }
        if query:
            payload["query"] = query
        return payload
    geocode_error = ""
    if query and geocoder is not None:
        try:
            result = geocoder.geocode(query)
        except Exception as exc:
            result = None
            geocode_error = str(exc or "").strip()
        if result is not None:
            loc: dict[str, Any] = {
                "query": result.query,
                "display_name": result.display_name,
                "gps_latitude": float(result.latitude),
                "gps_longitude": float(result.longitude),
                "map_datum": "WGS-84",
                "source": result.source,
            }
            if str(getattr(result, "city", "") or "").strip():
                loc["city"] = str(result.city).strip()
            if str(getattr(result, "state", "") or "").strip():
                loc["state"] = str(result.state).strip()
            if str(getattr(result, "country", "") or "").strip():
                loc["country"] = str(result.country).strip()
            return loc
    if query and geocode_error:
        return {
            "query": query,
            "error": geocode_error,
            "source": "nominatim",
        }
    return {}
