from __future__ import annotations

from pathlib import Path

from ._caption_album import (
    ALBUM_KIND_PHOTO_ESSAY,
    dedupe,
    infer_album_context,
    join_human,
)
from ._prompt_skill import (
    family_skill as _skill_family,
    section as _section,
    section_from as _section_from,
    travel_skill as _skill_travel,
)

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


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_describe_preamble(*, is_cover_page: bool = False) -> list[str]:
    lines: list[str] = []

    if is_cover_page:
        lines.extend(_section("Preamble Cover Page"))

    return lines


def _build_describe_context_hints(
    *,
    people_list: list[str],
    people_positions: dict[str, str] | None = None,
) -> list[str]:
    hint_lines: list[str] = []
    if people_list:
        if people_positions:
            entries = [
                (f"{name} ({people_positions[name]})" if name in people_positions else name) for name in people_list
            ]
            hint_lines.extend(_section("People Hint With Positions", people_hint=", ".join(entries)))
        else:
            hint_lines.extend(_section("People Hint", people_hint=join_human(people_list)))
    if not hint_lines:
        return []
    return ["Context hints (image-specific):"] + hint_lines


def _build_people_count_context_hints(
    *,
    people_list: list[str],
    people_positions: dict[str, str] | None = None,
) -> list[str]:
    hint_lines: list[str] = []
    if people_list:
        if people_positions:
            entries = [
                (f"{name} ({people_positions[name]})" if name in people_positions else name) for name in people_list
            ]
            hint_lines.extend(_section("People Count Hint With Positions", people_hint=", ".join(entries)))
        else:
            hint_lines.extend(_section("People Count Hint", people_hint=join_human(people_list)))
    if not hint_lines:
        return []
    return ["Context hints (image-specific):"] + hint_lines


def _build_local_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    is_cover_page: bool = False,
    people_positions: dict[str, str] | None = None,
    request_photo_regions: bool = True,
) -> str:
    people_list = dedupe(people)
    context = infer_album_context(
        image_path=source_path,
        ocr_text=ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    album_skill = _skill_travel() if context.kind == ALBUM_KIND_PHOTO_ESSAY else _skill_family()
    lines = _section_from(album_skill, "Preamble Describe")
    lines.extend(_build_describe_preamble(is_cover_page=is_cover_page))
    lines.extend(
        _build_describe_context_hints(
            people_list=people_list,
            people_positions=people_positions,
        )
    )
    if request_photo_regions:
        lines.extend(_section("Preamble Page Photo Regions Compact"))
        lines.extend(_section("Output Format – Describe Page (with photo regions)"))
    else:
        lines.extend(_section("Output Format – Describe (full caption)"))
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
    people_list = dedupe(people)
    lines = _section("Preamble People Count")
    lines.extend(
        _build_people_count_context_hints(
            people_list=people_list,
            people_positions=people_positions,
        )
    )
    lines.extend(_section("Output Format – People Count"))
    return "\n".join(lines)


def _build_location_prompt(
    *,
    is_cover_page: bool = False,
) -> str:
    lines = _section("Preamble Location")
    lines.extend(_build_describe_preamble(is_cover_page=is_cover_page))
    lines.extend(_section("Output Format – Location"))
    return "\n".join(lines)
