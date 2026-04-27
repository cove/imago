from __future__ import annotations

import re
from typing import Any

from .xmp_sidecar import _dedupe

_LOCATION_KEYS = (
    "address",
    "city",
    "state",
    "country",
    "sublocation",
    "gps_latitude",
    "gps_longitude",
)
_LOCATION_TEXT_RE = re.compile(r"[^A-Z0-9]+")
_MIN_TOKEN_MATCH_COUNT = 2
_LOCATION_CAPTION_MAX_WORDS = 6


def normalize_location_payload(payload: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    normalized = {key: str(payload.get(key) or "").strip() for key in _LOCATION_KEYS}
    return {key: value for key, value in normalized.items() if value}


def location_payload_from_location_shown(location: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(location, dict):
        return {}
    payload = {
        "address": str(location.get("name") or "").strip(),
        "city": str(location.get("city") or "").strip(),
        "state": str(location.get("province_or_state") or location.get("state") or "").strip(),
        "country": str(location.get("country_name") or location.get("country") or "").strip(),
        "sublocation": str(location.get("sublocation") or "").strip(),
        "gps_latitude": str(location.get("gps_latitude") or "").strip(),
        "gps_longitude": str(location.get("gps_longitude") or "").strip(),
    }
    if not payload["address"]:
        parts = [
            payload["sublocation"],
            payload["city"],
            payload["state"],
            payload["country"],
        ]
        payload["address"] = ", ".join(part for part in parts if part)
    return {key: value for key, value in payload.items() if value}


def location_shown_from_payload(payload: dict[str, Any] | None) -> dict[str, str]:
    normalized = normalize_location_payload(payload)
    if not normalized:
        return {}
    address = str(normalized.get("address") or "").strip()
    sublocation = str(normalized.get("sublocation") or "").strip()
    city = str(normalized.get("city") or "").strip()
    state = str(normalized.get("state") or "").strip()
    country = str(normalized.get("country") or "").strip()
    name = address or sublocation or ", ".join(part for part in (city, state, country) if part)
    location = {
        "name": name,
        "city": city,
        "province_or_state": state,
        "country_name": country,
        "sublocation": sublocation,
        "gps_latitude": str(normalized.get("gps_latitude") or "").strip(),
        "gps_longitude": str(normalized.get("gps_longitude") or "").strip(),
    }
    return {key: value for key, value in location.items() if value}


def location_payload_from_caption(caption: str) -> dict[str, str]:
    text = " ".join(str(caption or "").replace("\n", " ").split()).strip(" ,.;")
    if not text or "," not in text:
        return {}
    words = [word for word in _normalize_location_text(text).split() if word]
    if len(words) > _LOCATION_CAPTION_MAX_WORDS:
        return {}
    return {"address": text}


def _normalize_location_text(value: str) -> str:
    text = _LOCATION_TEXT_RE.sub(" ", str(value or "").upper()).strip()
    return " ".join(text.split())


def _token_match_score(normalized_caption: str, normalized_variant: str) -> int:
    tokens = [token for token in normalized_variant.split() if len(token) > 2]
    if len(tokens) < _MIN_TOKEN_MATCH_COUNT:
        return -1
    compact_caption = normalized_caption.replace(" ", "")
    matched_tokens = [token for token in tokens if token in normalized_caption or token in compact_caption]
    if len(matched_tokens) < _MIN_TOKEN_MATCH_COUNT:
        return -1
    if len(matched_tokens) < len(tokens) and "HOTEL" not in matched_tokens:
        return -1
    return sum(len(token) for token in matched_tokens)


def _location_text_variants(location: dict[str, Any] | None) -> list[str]:
    if not isinstance(location, dict):
        return []
    payload = location_payload_from_location_shown(location)
    if not payload:
        payload = normalize_location_payload(location)
    address = str(payload.get("address") or "").strip()
    city = str(payload.get("city") or "").strip()
    state = str(payload.get("state") or "").strip()
    country = str(payload.get("country") or "").strip()
    sublocation = str(payload.get("sublocation") or "").strip()
    variants = [
        address,
        ", ".join(part for part in (sublocation, city, state, country) if part),
        ", ".join(part for part in (city, state, country) if part),
        ", ".join(part for part in (state, country) if part),
        city,
        state,
        country,
        sublocation,
    ]
    return [variant for variant in _dedupe(variants) if str(variant or "").strip()]


def build_location_filter_set(
    *,
    locations_shown: list[dict[str, Any]] | None = None,
    location_payload: dict[str, Any] | None = None,
) -> set[str]:
    filters: set[str] = set()
    for location in list(locations_shown or []):
        for variant in _location_text_variants(location):
            normalized = _normalize_location_text(variant)
            if normalized:
                filters.add(normalized)
    for variant in _location_text_variants(location_payload or {}):
        normalized = _normalize_location_text(variant)
        if normalized:
            filters.add(normalized)
    return filters


def filter_location_names_from_people(
    person_names: list[str] | tuple[str, ...] | None,
    location_filter_set: set[str] | None,
) -> list[str]:
    if not location_filter_set:
        return _dedupe([str(name or "").strip() for name in list(person_names or []) if str(name or "").strip()])
    filtered: list[str] = []
    for name in list(person_names or []):
        clean_name = str(name or "").strip()
        if not clean_name:
            continue
        normalized_name = _normalize_location_text(clean_name)
        if normalized_name and any(
            normalized_name == location_text
            or normalized_name in location_text
            or location_text in normalized_name
            for location_text in location_filter_set
        ):
            continue
        filtered.append(clean_name)
    return _dedupe(filtered)


def resolve_person_in_image(
    person_names: list[str] | tuple[str, ...] | None,
    *,
    locations_shown: list[dict[str, Any]] | None = None,
    location_payload: dict[str, Any] | None = None,
) -> list[str]:
    return filter_location_names_from_people(
        person_names,
        build_location_filter_set(locations_shown=locations_shown, location_payload=location_payload),
    )


def match_caption_to_location_shown(
    caption: str,
    locations_shown: list[dict[str, Any]] | None,
) -> dict[str, str] | None:
    normalized_caption = _normalize_location_text(caption)
    if not normalized_caption:
        return None
    best_match: dict[str, str] | None = None
    best_score = -1
    best_has_gps = False
    for location in list(locations_shown or []):
        if not isinstance(location, dict):
            continue
        payload = location_payload_from_location_shown(location)
        if not payload:
            continue
        has_gps = bool(payload.get("gps_latitude") and payload.get("gps_longitude"))
        for variant in _location_text_variants(location):
            normalized_variant = _normalize_location_text(variant)
            if not normalized_variant:
                continue
            if normalized_variant == normalized_caption:
                score = len(normalized_variant) + 1000
            elif normalized_variant in normalized_caption:
                score = len(normalized_variant)
            elif normalized_caption in normalized_variant:
                score = len(normalized_caption)
            else:
                score = _token_match_score(normalized_caption, normalized_variant)
                if score < 0:
                    continue
            if score > best_score or (score == best_score and has_gps and not best_has_gps):
                best_match = payload
                best_score = score
                best_has_gps = has_gps
    return best_match


def match_location_shown_row(
    caption: str,
    locations_shown: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    normalized_caption = _normalize_location_text(caption)
    if not normalized_caption:
        return None
    best_match: dict[str, Any] | None = None
    best_score = -1
    best_has_gps = False
    for location in list(locations_shown or []):
        if not isinstance(location, dict):
            continue
        payload = location_payload_from_location_shown(location)
        if not payload:
            continue
        has_gps = bool(payload.get("gps_latitude") and payload.get("gps_longitude"))
        for variant in _location_text_variants(location):
            normalized_variant = _normalize_location_text(variant)
            if not normalized_variant:
                continue
            if normalized_variant == normalized_caption:
                score = len(normalized_variant) + 1000
            elif normalized_variant in normalized_caption:
                score = len(normalized_variant)
            elif normalized_caption in normalized_variant:
                score = len(normalized_caption)
            else:
                score = _token_match_score(normalized_caption, normalized_variant)
                if score < 0:
                    continue
            if score > best_score or (score == best_score and has_gps and not best_has_gps):
                best_match = dict(location)
                best_score = score
                best_has_gps = has_gps
    return best_match


def resolve_crop_location(
    *,
    region_location_override: dict[str, Any] | None,
    region_location_assigned: dict[str, Any] | None,
    caption: str,
    locations_shown: list[dict[str, Any]] | None,
    page_location: dict[str, Any] | None,
) -> dict[str, str]:
    del page_location
    override_payload = normalize_location_payload(region_location_override)
    if override_payload:
        return override_payload
    assigned_payload = normalize_location_payload(region_location_assigned)
    if assigned_payload:
        return assigned_payload
    matched_location = match_caption_to_location_shown(caption, locations_shown)
    if matched_location:
        return matched_location
    caption_location = location_payload_from_caption(caption)
    if caption_location:
        return caption_location
    if len(list(locations_shown or [])) == 1:
        return location_payload_from_location_shown(list(locations_shown or [])[0])
    return {}


def resolve_crop_locations_shown(
    *,
    region_location_override: dict[str, Any] | None,
    region_location_assigned: dict[str, Any] | None,
    caption: str,
    locations_shown: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    override_location = location_shown_from_payload(region_location_override)
    if override_location:
        return [override_location]
    assigned_location = location_shown_from_payload(region_location_assigned)
    if assigned_location:
        return [assigned_location]
    matched_location = match_location_shown_row(caption, locations_shown)
    if isinstance(matched_location, dict) and matched_location:
        return [matched_location]
    if len(list(locations_shown or [])) == 1:
        only_location = list(locations_shown or [])[0]
        return [only_location] if isinstance(only_location, dict) else []
    return []
