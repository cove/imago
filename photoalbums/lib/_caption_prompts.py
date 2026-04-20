from __future__ import annotations

import re
from pathlib import Path

from ._prompt_skill import section as _section

# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _position_label(cx: float, cy: float) -> str:
    """Return a human-readable position label from normalised centre coordinates.

    cx and cy are in [0, 1] where (0, 0) is the top-left corner.
    """
    v = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "centre")
    h = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    if v == "centre" and h == "centre":
        return "centre"
    if v == "centre":
        return h
    if h == "centre":
        return v
    return f"{v}-{h}"


def _looks_like_title_page(source_path: str | Path | None) -> bool:
    name = Path(source_path).name if source_path else ""
    return bool(re.search(r"_P0[01](?:_|\.|$)", name, flags=re.IGNORECASE))


def _build_local_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    people_positions: dict[str, str] | None = None,
    context_ocr_text: str = "",
) -> str:
    del people, objects, album_title, printed_album_title, people_positions
    lines = _section("Preamble Describe")
    if _looks_like_title_page(source_path):
        lines.extend(_section("Preamble Cover Page", ocr_text=ocr_text))
    clean_context_ocr = str(context_ocr_text or "").strip()
    if clean_context_ocr:
        lines.extend(_section("Preamble Upstream OCR Context", context_ocr_text=clean_context_ocr))
    lines.extend(_section("Output Format – Describe Page"))

    return "\n".join(lines)


def _build_people_count_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    people_positions: dict[str, str] | None = None,
) -> str:
    lines = _section("Preamble People Count")
    lines.extend(_section("Output Format – People Count"))
    return "\n".join(lines)


def _build_location_prompt(*, ocr_text: str = "", album_title: str = "", printed_album_title: str = "") -> str:
    album_hint = str(album_title or "").strip() or str(printed_album_title or "").strip()
    lines = _section("Preamble Location", album_title=album_hint, ocr_text=str(ocr_text or "").strip())
    lines.extend(_section("Output Format – Location"))
    return "\n".join(lines)


def _build_location_shown_prompt(*, ocr_text: str = "", album_title: str = "", printed_album_title: str = "") -> str:
    album_hint = str(album_title or "").strip() or str(printed_album_title or "").strip()
    lines = _section("Preamble Location", album_title=album_hint, ocr_text=str(ocr_text or "").strip())
    lines.extend(_section("Output Format – Location Shown"))
    return "\n".join(lines)


def _build_location_queries_prompt(
    *,
    caption_text: str = "",
    ocr_text: str = "",
    album_title: str = "",
    printed_album_title: str = "",
) -> str:
    album_hint = str(album_title or "").strip() or str(printed_album_title or "").strip()
    parts: list[str] = []
    if album_hint:
        parts.append(f"Album: {album_hint}")
    if caption_text:
        parts.append(f"Caption: {caption_text.strip()}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text.strip()}")
    parts.append(
        "Based on the image and the above context, identify the location(s) shown.\n"
        "Nominatim accepts free-form place name queries in any language (e.g. \"Eiffel Tower, Paris, France\", "
        "\"Cafe Paris, New York\"). Do NOT return raw latitude/longitude values — return human-readable place names "
        "that Nominatim can resolve.\n"
        "Return:\n"
        "- primary_query: the single most specific location for the primary GPS (empty string if unknown)\n"
        "- named_queries: list of named place queries shown in the image (may be empty)"
    )
    return "\n\n".join(parts)
