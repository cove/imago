from __future__ import annotations

from pathlib import Path

from ._caption_album import (
    ALBUM_KIND_PHOTO_ESSAY,
    AlbumContext,
    clean_text,
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


def _should_apply_album_prompt_rules(source_path: str | Path | None, album_context: AlbumContext) -> bool:
    if album_context.kind:
        return True
    if source_path is None:
        return False
    joined = " ".join(str(part or "").casefold() for part in Path(source_path).parts)
    return "photo albums" in joined or "cordell" in joined


def _build_describe_preamble(*, is_cover_page: bool = False) -> list[str]:
    lines: list[str] = []

    if is_cover_page:
        lines.extend(_section("Preamble Cover Page"))

    return lines


def _build_describe_context_hints(
    *,
    context: AlbumContext,
    source_path: str | Path | None,
    people_list: list[str],
    ocr_text: str,
    people_positions: dict[str, str] | None = None,
) -> list[str]:
    hint_lines: list[str] = []
    if _should_apply_album_prompt_rules(source_path, context):
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            hint_lines.extend(_section("Album Focus Hint", album_focus=context.focus))
    if people_list:
        if people_positions:
            entries = [
                (f"{name} ({people_positions[name]})" if name in people_positions else name) for name in people_list
            ]
            hint_lines.extend(_section("People Hint With Positions", people_hint=", ".join(entries)))
        else:
            hint_lines.extend(_section("People Hint", people_hint=join_human(people_list)))
    text = clean_text(ocr_text)
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        hint_lines.extend(_section("OCR Hint", ocr_snippet=snippet))
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


def _build_location_context_hints(
    *,
    context: AlbumContext,
    source_path: str | Path | None,
    ocr_text: str,
) -> list[str]:
    hint_lines: list[str] = []
    if _should_apply_album_prompt_rules(source_path, context):
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            hint_lines.extend(_section("Album Focus Hint", album_focus=context.focus))
    text = clean_text(ocr_text)
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        hint_lines.extend(_section("OCR Hint", ocr_snippet=snippet))
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
    request_photo_regions: bool = False,
) -> str:
    people_list = dedupe(people)
    text = clean_text(ocr_text)
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
            context=context,
            source_path=source_path,
            people_list=people_list,
            ocr_text=text,
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
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    is_cover_page: bool = False,
    people_positions: dict[str, str] | None = None,
) -> str:
    text = clean_text(ocr_text)
    context = infer_album_context(
        image_path=source_path,
        ocr_text=ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    lines = _section("Preamble Location")
    lines.extend(_build_describe_preamble(is_cover_page=is_cover_page))
    lines.extend(
        _build_location_context_hints(
            context=context,
            source_path=source_path,
            ocr_text=text,
        )
    )
    lines.extend(_section("Output Format – Location"))
    return "\n".join(lines)


def _build_describe_prompt(
    prompt_text: str,
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None,
    album_title: str,
    printed_album_title: str,
    is_cover_page: bool,
    people_positions: dict[str, str] | None = None,
    request_photo_regions: bool = False,
) -> str:
    return prompt_text or _build_local_prompt(
        people=people,
        objects=objects,
        ocr_text=ocr_text,
        source_path=source_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        is_cover_page=is_cover_page,
        people_positions=people_positions,
        request_photo_regions=request_photo_regions,
    )
