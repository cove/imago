from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ._caption_album import (
    ALBUM_KIND_PHOTO_ESSAY,
    AlbumContext,
    clean_text,
    dedupe,
    infer_album_context,
    join_human,
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
# Skill file loader
# ---------------------------------------------------------------------------

_SKILL_FILE = Path(__file__).parent.parent.parent / "skills" / "CORDELL_PHOTO_ALBUMS" / "SKILL.md"


def _parse_skill(path: Path) -> dict[str, list[str]]:
    """Parse a .skill file into {section_name: [lines]}."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            stripped = raw.rstrip()
            if stripped.startswith("## ") and not stripped.startswith("### "):
                current = stripped[3:].strip()
                sections.setdefault(current, [])
            elif current is not None and stripped.strip() and not stripped.strip().startswith("#"):
                sections[current].append(stripped.strip())
    return sections


@lru_cache(maxsize=1)
def _skill() -> dict[str, list[str]]:
    try:
        return _parse_skill(_SKILL_FILE)
    except FileNotFoundError:
        return {}


def _section(*names: str, **kwargs: str) -> list[str]:
    """Return lines from one or more named skill sections with {var} substitution.

    Lines where any substituted variable resolves to an empty string are dropped.
    """
    lines: list[str] = []
    skill = _skill()
    for name in names:
        for raw in skill.get(name, []):
            rendered, drop = _render_line(raw, kwargs)
            if not drop:
                lines.append(rendered)
    return lines


def _render_line(text: str, vars: dict[str, str]) -> tuple[str, bool]:
    """Substitute {var} patterns. Returns (rendered, drop) where drop=True if any var was empty."""
    drop = False

    def replace(m: re.Match) -> str:
        nonlocal drop
        val = vars.get(m.group(1), "")
        if not val:
            drop = True
        return val

    return re.sub(r"\{(\w+)\}", replace, text), drop


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


def _build_shared_prompt_rules(
    *,
    context: AlbumContext,
    source_path: str | Path | None,
    people_list: list[str],
    object_list: list[str],
    combined: bool = False,
    is_cover_page: bool = False,
    people_positions: dict[str, str] | None = None,
) -> list[str]:
    lines: list[str] = []

    if is_cover_page:
        lines.extend(_section("Preamble Cover Page"))

    if context.title:
        lines.extend(_section("Album Title Hint", album_title=context.title))

    if context.canonical_title and context.title and context.canonical_title.casefold() != context.title.casefold():
        lines.extend(_section("Canonical Title Hint", canonical_title=context.canonical_title))

    if _should_apply_album_prompt_rules(source_path, context):
        lines.extend(_section("Album Classification Rules (apply in this order)"))
        if context.label:
            lines.extend(_section("Album Classification Hint", album_label=context.label))
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            lines.extend(_section("Album Focus Hint", album_focus=context.focus))

    lines.extend(_section("Global Style & Behavior Rules (apply to every mode)"))
    lines.extend(_section("Text Handling & Correction Rules"))
    lines.extend(_section("Location Rules (strict)"))
    lines.extend(_section("People Rules"))

    if people_list:
        if people_positions:
            entries = [
                (f"{name} ({people_positions[name]})" if name in people_positions else name) for name in people_list
            ]
            lines.extend(_section("People Hint With Positions", people_hint=", ".join(entries)))
        else:
            lines.extend(_section("People Hint", people_hint=join_human(people_list)))

    if object_list:
        lines.extend(_section("Objects Hint", object_list=join_human(object_list)))

    return lines


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
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = infer_album_context(
        image_path=source_path,
        ocr_text=ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    lines = _section("Preamble Describe")
    lines.extend(
        _build_shared_prompt_rules(
            context=context,
            source_path=source_path,
            people_list=people_list,
            object_list=object_list,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
        )
    )
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.extend(_section("OCR Hint", ocr_snippet=snippet))
    if request_photo_regions:
        lines.extend(_section("Preamble Page Photo Regions"))
        lines.extend(_section("Output Format – Describe Page (with photo regions)"))
    else:
        lines.extend(_section("Output Format – Describe (full caption)"))
    return "\n".join(lines)


def _build_combined_local_prompt(
    *,
    people: list[str],
    objects: list[str],
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    is_cover_page: bool = False,
    people_positions: dict[str, str] | None = None,
) -> str:
    """Prompt that requests both OCR text and a caption in a single inference."""
    people_list = dedupe(people)
    object_list = dedupe(objects)
    context = infer_album_context(
        image_path=source_path,
        ocr_text="",
        allow_ocr=False,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    lines = _section("Preamble Combined")
    lines.extend(
        _build_shared_prompt_rules(
            context=context,
            source_path=source_path,
            people_list=people_list,
            object_list=object_list,
            combined=True,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
        )
    )
    lines.extend(_section("Output Format – Combined"))
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
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = infer_album_context(
        image_path=source_path,
        ocr_text=ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    lines = _section("Preamble People Count")
    if _should_apply_album_prompt_rules(source_path, context):
        lines.extend(_section("Album Classification Rules (apply in this order)"))
    lines.extend(_section("People Rules"))
    if people_list:
        if people_positions:
            entries = [
                (f"{name} ({people_positions[name]})" if name in people_positions else name) for name in people_list
            ]
            lines.extend(_section("People Count Hint With Positions", people_hint=", ".join(entries)))
        else:
            lines.extend(_section("People Count Hint", people_hint=join_human(people_list)))
    if object_list:
        lines.extend(_section("Objects Hint", object_list=join_human(object_list)))
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.extend(_section("OCR Hint", ocr_snippet=snippet))
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
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = infer_album_context(
        image_path=source_path,
        ocr_text=ocr_text,
        allow_ocr=True,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )
    lines = _section("Preamble Location")
    lines.extend(
        _build_shared_prompt_rules(
            context=context,
            source_path=source_path,
            people_list=people_list,
            object_list=object_list,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
        )
    )
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.extend(_section("OCR Hint", ocr_snippet=snippet))
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
