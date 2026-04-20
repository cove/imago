from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .ai_caption import _normalize_gps_value
from .ai_geocode import NominatimGeocoder
from .prompt_debug import PromptDebugSession

if TYPE_CHECKING:
    from .ai_caption import CaptionEngine

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


def _serialize_geocode_result(result: Any) -> dict[str, Any]:
    return {
        "query": str(getattr(result, "query", "") or "").strip(),
        "display_name": str(getattr(result, "display_name", "") or "").strip(),
        "latitude": str(getattr(result, "latitude", "") or "").strip(),
        "longitude": str(getattr(result, "longitude", "") or "").strip(),
        "source": str(getattr(result, "source", "") or "nominatim").strip(),
        "city": str(getattr(result, "city", "") or "").strip(),
        "state": str(getattr(result, "state", "") or "").strip(),
        "country": str(getattr(result, "country", "") or "").strip(),
        "sublocation": str(getattr(result, "sublocation", "") or "").strip(),
    }


def _record_geocode_lookup(
    *,
    artifact_recorder,
    step: str,
    query: str,
    result: Any = None,
    error: str = "",
) -> None:
    if not callable(artifact_recorder):
        return
    payload: dict[str, Any] = {
        "step": str(step or "").strip(),
        "service": "nominatim",
        "query": str(query or "").strip(),
    }
    if result is not None:
        payload["status"] = "ok"
        payload["result"] = _serialize_geocode_result(result)
    elif error:
        payload["status"] = "error"
        payload["error"] = str(error).strip()
    else:
        payload["status"] = "miss"
    artifact_recorder(payload)


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
    artifact_recorder=None,
    artifact_step: str = "location",
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
            _record_geocode_lookup(
                artifact_recorder=artifact_recorder,
                step=artifact_step,
                query=query,
                error=geocode_error,
            )
        else:
            _record_geocode_lookup(
                artifact_recorder=artifact_recorder,
                step=artifact_step,
                query=query,
                result=result,
            )
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
            if str(getattr(result, "sublocation", "") or "").strip():
                loc["sublocation"] = str(result.sublocation).strip()
            return loc
    if query and geocode_error:
        return {
            "query": query,
            "error": geocode_error,
            "source": "nominatim",
        }
    return {}


def _build_locations_shown_query(location: dict[str, Any]) -> str:
    name = str(location.get("name") or "").strip()
    if not name:
        return ""
    parts = [name]
    seen = {name.casefold()}
    for chunk in name.split(","):
        normalized_chunk = str(chunk or "").strip()
        if normalized_chunk:
            seen.add(normalized_chunk.casefold())
    for key in ("sublocation", "city", "province_or_state", "country_name"):
        value = str(location.get(key) or "").strip()
        if not value:
            continue
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        parts.append(value)
    return ", ".join(parts)


def _has_legacy_ai_locations_shown_gps(sidecar_state: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar_state, dict):
        return False
    detections = sidecar_state.get("detections")
    if not isinstance(detections, dict):
        return False
    for location in list(detections.get("locations_shown") or []):
        if not isinstance(location, dict):
            continue
        has_gps = bool(
            str(location.get("gps_latitude") or "").strip() and str(location.get("gps_longitude") or "").strip()
        )
        if not has_gps:
            continue
        if str(location.get("gps_source") or "").strip().lower() not in {"nominatim", "manual"}:
            return True
    return False


def _resolve_locations_shown(
    *,
    requested_caption_engine: str,
    caption_engine: Any,
    model_image_path: Path,
    ocr_text: str,
    source_path: Path,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    prompt_debug: PromptDebugSession | None = None,
    debug_step: str = "locations_shown",
    artifact_recorder=None,
) -> tuple[list[dict[str, Any]], bool]:
    """Resolve locations shown in the image using AI.

    Returns tuple of (locations_list, ran_flag).
    """
    if str(requested_caption_engine or "").strip().lower() != "lmstudio":
        return [], False
    estimate_locations_shown = getattr(caption_engine, "estimate_locations_shown", None)
    if not callable(estimate_locations_shown):
        return [], False
    try:
        result = estimate_locations_shown(
            image_path=model_image_path,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step=debug_step,
        )
    except Exception:
        return [], False
    fallback = getattr(result, "fallback", False)
    if not isinstance(fallback, bool) or fallback:
        return [], False
    locations = getattr(result, "locations_shown", None)
    if not isinstance(locations, list):
        return [], True
    validated: list[dict[str, Any]] = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        normalized = {
            "name": str(loc.get("name") or "").strip(),
            "world_region": str(loc.get("world_region") or "").strip(),
            "country_name": str(loc.get("country_name") or "").strip(),
            "country_code": str(loc.get("country_code") or "").strip(),
            "province_or_state": str(loc.get("province_or_state") or "").strip(),
            "city": str(loc.get("city") or "").strip(),
            "sublocation": str(loc.get("sublocation") or "").strip(),
        }
        if geocoder is not None:
            query = _build_locations_shown_query(normalized)
            if query:
                try:
                    geocoded = geocoder.geocode(query)
                except Exception as exc:
                    geocoded = None
                    _record_geocode_lookup(
                        artifact_recorder=artifact_recorder,
                        step=debug_step,
                        query=query,
                        error=str(exc or "").strip(),
                    )
                else:
                    _record_geocode_lookup(
                        artifact_recorder=artifact_recorder,
                        step=debug_step,
                        query=query,
                        result=geocoded,
                    )
                if geocoded is not None:
                    normalized["gps_latitude"] = str(geocoded.latitude).strip()
                    normalized["gps_longitude"] = str(geocoded.longitude).strip()
                    normalized["gps_source"] = "nominatim"
                    if not str(normalized.get("sublocation") or "").strip():
                        normalized["sublocation"] = str(getattr(geocoded, "sublocation", "") or "").strip()
        validated.append(normalized)
    return validated, True


def run_locations_step(
    *,
    caption_engine: "CaptionEngine",
    image_path: Path,
    caption_text: str,
    ocr_text: str = "",
    source_path: Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    prompt_debug: PromptDebugSession | None = None,
    artifact_recorder=None,
) -> dict[str, Any] | None:
    """Run the consolidated locations step.

    Returns a dict with location, locations_shown, location_shown_ran keys,
    or None if the engine is not configured (not-applicable).
    """
    if str(getattr(caption_engine, "engine", "") or "").strip().lower() != "lmstudio":
        return None

    debug_recorder = prompt_debug.record if prompt_debug is not None else None
    result = caption_engine.generate_location_queries(
        image_path=image_path,
        caption_text=caption_text,
        ocr_text=ocr_text,
        source_path=source_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        debug_recorder=debug_recorder,
        debug_step="location_queries",
    )

    if result.fallback:
        # AI call failed — record ran with empty results so downstream can still proceed
        return {
            "location": {},
            "locations_shown": [],
            "location_shown_ran": True,
        }

    # Resolve primary GPS query
    location_payload = _resolve_location_payload(
        geocoder=geocoder,
        gps_latitude="",
        gps_longitude="",
        location_name=result.primary_query,
        artifact_recorder=artifact_recorder,
        artifact_step="location",
    )

    # Resolve named location queries
    locations_shown: list[dict[str, Any]] = []
    for query in (result.named_queries or []):
        query = str(query or "").strip()
        if not query:
            continue
        entry: dict[str, Any] = {"name": query}
        if geocoder is not None:
            try:
                geocoded = geocoder.geocode(query)
            except Exception as exc:
                geocoded = None
                _record_geocode_lookup(
                    artifact_recorder=artifact_recorder,
                    step="locations_shown",
                    query=query,
                    error=str(exc or "").strip(),
                )
            else:
                _record_geocode_lookup(
                    artifact_recorder=artifact_recorder,
                    step="locations_shown",
                    query=query,
                    result=geocoded,
                )
            if geocoded is not None:
                entry["gps_latitude"] = str(geocoded.latitude).strip()
                entry["gps_longitude"] = str(geocoded.longitude).strip()
                entry["gps_source"] = "nominatim"
                for field in ("city", "state", "country", "sublocation"):
                    val = str(getattr(geocoded, field, "") or "").strip()
                    if val:
                        entry[field] = val
        # Normalise into the canonical locations_shown schema
        normalized: dict[str, Any] = {
            "name": entry.get("name", query),
            "world_region": "",
            "country_name": entry.get("country", ""),
            "country_code": "",
            "province_or_state": entry.get("state", ""),
            "city": entry.get("city", ""),
            "sublocation": entry.get("sublocation", ""),
        }
        if entry.get("gps_latitude"):
            normalized["gps_latitude"] = entry["gps_latitude"]
            normalized["gps_longitude"] = entry["gps_longitude"]
            normalized["gps_source"] = entry.get("gps_source", "nominatim")
        locations_shown.append(normalized)

    return {
        "location": location_payload,
        "locations_shown": locations_shown,
        "location_shown_ran": True,
    }
