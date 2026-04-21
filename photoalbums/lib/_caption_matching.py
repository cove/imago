"""Sort photo regions into reading order and match captions via LM Studio."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

CAPTION_MATCHING_PROMPT = (
    "Use the visible numbered region overlay as the authoritative region identity contract.\n"
    '`{"region-1": "", "region-2": "", "region-3": ""}`\n'
    "- Number as many `region-N` keys as there are visible numbered regions on the overlay.\n"
    "- Each value is the caption text that belongs to that region; empty string if there is no caption.\n"
    "- Use the visible overlay numbers directly. Do not renumber the regions and do not infer a separate left-to-right/top-to-bottom ordering.\n"
    "- A single visible page caption may apply to multiple photos. When one caption clearly covers multiple regions, repeat that exact caption text for every covered region instead of assigning it to only one region.\n"
    "- Grouped-photo captions are valid: for example stacked photos in one column, adjacent photos under one label, or any nearby set that is clearly covered by the same page caption.\n"
    "- Do not use page-wide headers, place/date banners, or general page context text as a region caption when a more specific nearby photo caption is visible. Prefer the photo-specific caption.\n"
    "- Example: if two photos are vertically stacked in one column and there is one specific caption strip under that column, assign that same caption to both photos in the column.\n"
    "- If a caption refers to subjects shown in an adjacent photo (e.g. \"Their new home\" paired with a photo labelled \"GILBERT & HELEN\"), prepend the missing subject so the caption reads standalone (e.g. \"GILBERT & HELEN — Their new home\"). Do not rewrite or summarise; only prepend the minimum context needed.\n"
    "- Just return the JSON without any extra text or explanation."
)
MULTI_LOCATION_PROMPT_TEMPLATE = (
    "Use the visible numbered region overlay as the authoritative region identity contract.\n"
    '`{{"region-1": {{"caption": "", "location": ""}}, "region-2": {{"caption": "", "location": ""}}}}`\n'
    "- Number as many `region-N` keys as there are visible numbered regions on the overlay.\n"
    '- "caption": the caption text that belongs to that region; empty string if there is no caption.\n'
    '- "location": copy exactly one item from the Known locations list below; do not invent a new place name, do not repeat the caption text, and do not add dates or extra words. If none apply, return empty string.\n'
    "Known locations: {known_locations}\n"
    "- Use the visible overlay numbers directly. Do not renumber the regions and do not infer a separate left-to-right/top-to-bottom ordering.\n"
    "- A single visible page caption may apply to multiple photos. When one caption clearly covers multiple regions, repeat that exact caption text for every covered region instead of assigning it to only one region.\n"
    '- Choose "location" independently for each region even when multiple regions share the same caption text.\n'
    "- Do not use page-wide headers, place/date banners, or general page context text as a region caption when a more specific nearby photo caption is visible. Prefer the photo-specific caption.\n"
    "- Example: if two photos are vertically stacked in one column and there is one specific caption strip under that column, assign that same caption to both photos in the column.\n"
    "- If a caption refers to subjects shown in an adjacent photo, prepend the missing subject so the caption reads standalone. Do not rewrite or summarise; only prepend the minimum context needed.\n"
    "- Just return the JSON without any extra text or explanation."
)

DEFAULT_TIMEOUT = 300.0
_DEFAULT_MAX_IMAGE_EDGE = 2048


def sort_regions_reading_order(
    regions: list,
    img_height: int,
    row_tolerance_frac: float = 0.10,
) -> list:
    """Sort regions left-to-right/top-to-bottom via coordinate-based scanline sort.

    Docling does not output boxes in reading order — always call this before
    mapping regions to LM Studio photo-N indices.
    """
    if not regions:
        return []
    tolerance = max(1.0, row_tolerance_frac * img_height)
    sorted_by_y = sorted(regions, key=lambda r: r.y)
    rows: list[list] = []
    current_row: list = [sorted_by_y[0]]
    current_row_y = float(sorted_by_y[0].y)
    for region in sorted_by_y[1:]:
        if region.y <= current_row_y + tolerance:
            current_row.append(region)
        else:
            rows.append(sorted(current_row, key=lambda r: r.x))
            current_row = [region]
            current_row_y = float(region.y)
    rows.append(sorted(current_row, key=lambda r: r.x))
    return [region for row in rows for region in row]


def _encode_image(image_path: Path, max_edge: int = _DEFAULT_MAX_IMAGE_EDGE) -> str:
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    image = Image.open(image_path)
    try:
        w, h = image.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        if image_path.suffix.lower() == ".png":
            mime = "image/png"
            image.save(buffer, format="PNG")
        else:
            mime = "image/jpeg"
            image.save(buffer, format="JPEG", quality=95)
    finally:
        image.close()
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"LM Studio request failed: {details or f'HTTP {exc.code}'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _resolve_model(base_url: str, model: str, timeout: float) -> str:
    """Return model name, auto-discovering the loaded model if model is empty."""
    if model:
        return model
    payload = _get_json(f"{base_url}/models", timeout)
    model_ids = [
        str(row.get("id") or "").strip()
        for row in list(payload.get("data") or [])
        if str(row.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError("LM Studio did not return any models — load a model or configure caption_matching_model in ai_models.toml")
    return model_ids[0]


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"LM Studio request failed: {details or f'HTTP {exc.code}'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _extract_json_object(text: str) -> str | None:
    """Extract the first {...} JSON object from a possibly-wrapped response."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _known_locations_list(locations_shown: list[dict] | None) -> list[str]:
    if not isinstance(locations_shown, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for location in locations_shown:
        if not isinstance(location, dict):
            continue
        name = str(location.get("name") or "").strip()
        if not name:
            parts: list[str] = []
            for key in ("sublocation", "city", "province_or_state", "country_name"):
                value = str(location.get(key) or "").strip()
                if value and value not in parts:
                    parts.append(value)
            name = ", ".join(parts)
        if not name:
            continue
        folded = name.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        names.append(name)
    return names


def _build_caption_matching_prompt(locations_shown: list[dict] | None) -> str:
    known_locations = _known_locations_list(locations_shown)
    if len(known_locations) < 2:
        return CAPTION_MATCHING_PROMPT
    return MULTI_LOCATION_PROMPT_TEMPLATE.format(known_locations=", ".join(known_locations))


def _normalize_location_choice(location_text: str, known_locations: list[str]) -> str:
    clean_location = str(location_text or "").strip()
    if not clean_location or len(known_locations) < 2:
        return clean_location

    folded_location = clean_location.casefold()
    for candidate in known_locations:
        if folded_location == candidate.casefold():
            return candidate

    matches = [candidate for candidate in known_locations if candidate.casefold() in folded_location]
    if len(matches) == 1:
        return matches[0]
    return ""


def call_lmstudio_caption_matching(
    image_path: str | Path,
    base_url: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT,
    locations_shown: list[dict] | None = None,
) -> dict[int, dict[str, str]]:
    """Call the configured LM Studio model to match captions to numbered regions.

    Returns a dict mapping 1-based overlay region number to caption/location fields.
    Returns empty dict on any error (LM Studio offline, bad JSON, etc.).
    """
    path = Path(image_path)
    try:
        resolved_model = _resolve_model(base_url, model, timeout)
    except RuntimeError as exc:
        log.warning("LM Studio caption matching: %s", exc)
        return {}

    try:
        image_data = _encode_image(path)
    except Exception as exc:
        log.warning("LM Studio caption matching: failed to encode image %s: %s", path, exc)
        return {}

    payload: dict = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": _build_caption_matching_prompt(locations_shown)},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    try:
        response = _post_json(f"{base_url}/chat/completions", payload, timeout)
    except RuntimeError as exc:
        log.warning("LM Studio caption matching: %s", exc)
        return {}

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        log.warning("LM Studio caption matching: unexpected response structure: %r", response)
        return {}

    raw_text = str(content or "").strip()
    json_str = _extract_json_object(raw_text)
    if not json_str:
        log.warning("LM Studio caption matching: no JSON object found in response: %r", raw_text[:200])
        return {}

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        log.warning("LM Studio caption matching: malformed JSON in response: %r", json_str[:200])
        return {}

    if not isinstance(parsed, dict):
        log.warning("LM Studio caption matching: JSON is not an object: %r", parsed)
        return {}

    known_locations = _known_locations_list(locations_shown)
    result: dict[int, dict[str, str]] = {}
    for key, value in parsed.items():
        key_str = str(key or "").strip()
        if not key_str.startswith("region-"):
            continue
        try:
            idx = int(key_str[7:])
        except ValueError:
            continue
        if isinstance(value, dict):
            result[idx] = {
                "caption": str(value.get("caption") or "").strip(),
                "location": _normalize_location_choice(str(value.get("location") or "").strip(), known_locations),
            }
            continue
        result[idx] = {
            "caption": str(value or "").strip(),
            "location": "",
        }
    return result


def assign_captions_from_lmstudio(regions: list, captions: dict[int, str | dict[str, str]]) -> list:
    """Return regions with caption_hint populated from LM Studio's overlay mapping."""
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    result = []
    for region in regions:
        raw_value = captions.get(int(getattr(region, "index", -1)) + 1, "")
        if isinstance(raw_value, dict):
            caption = str(raw_value.get("caption") or "").strip()
            location = str(raw_value.get("location") or "").strip()
        else:
            caption = str(raw_value or "").strip()
            location = ""
        result.append(replace(region, caption_hint=caption, location_hint=location))
    return result
