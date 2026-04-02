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
) -> str:
    lines = _section("Preamble Describe")
    if _looks_like_title_page(source_path):
        lines.extend(_section("Preamble Cover Page", ocr_text=ocr_text))
    lines.extend(_section("Preamble Page Photo Regions Compact"))
    lines.extend(_section("Output Format – Describe Page (with photo regions)"))

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


def _build_location_prompt() -> str:
    lines = _section("Preamble Location")
    lines.extend(_section("Output Format – Location"))
    return "\n".join(lines)


def _build_location_shown_prompt(*, ocr_text: str = "") -> str:
    lines = _section("Preamble Location Shown")
    text = str(ocr_text or "").strip()
    if text:
        lines.append("- OCR text hints about the general location:")
        lines.append(text)
    else:
        lines.append("No OCR text hints are available for this page.")
    lines.extend(_section("Output Format – Location Shown"))
    return "\n".join(lines)
