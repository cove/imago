from __future__ import annotations

from pathlib import Path

from ._caption_album import (
    ALBUM_KIND_FAMILY,
    ALBUM_KIND_PHOTO_ESSAY,
    AlbumContext,
    clean_text,
    dedupe,
    infer_album_context,
    join_human,
)


def _should_apply_album_prompt_rules(
    source_path: str | Path | None, album_context: AlbumContext
) -> bool:
    if album_context.kind:
        return True
    if source_path is None:
        return False
    joined = " ".join(str(part or "").casefold() for part in Path(source_path).parts)
    return "photo albums" in joined or "cordell" in joined


def build_template_caption(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> str:
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = album_context or AlbumContext()

    parts: list[str] = []
    subject_prefix = (
        f"This image from {context.title}" if context.title else "This photo"
    )
    if people_list and object_list:
        parts.append(
            f"{subject_prefix} shows {join_human(people_list)} with {join_human(object_list)} in view."
        )
    elif people_list:
        parts.append(f"{subject_prefix} shows {join_human(people_list)}.")
    elif object_list:
        parts.append(f"{subject_prefix} includes {join_human(object_list)}.")

    if text:
        snippet = text[:180].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible text reads: "{snippet}".')
    return " ".join(parts).strip()


def build_page_caption(
    *,
    photo_count: int,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> str:
    count = max(1, int(photo_count))
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = album_context or AlbumContext()

    if context.title and context.kind == ALBUM_KIND_FAMILY:
        parts = [
            f"This page from {context.title}, a Family Photo Album, contains {count} photo(s)."
        ]
    elif context.title and context.kind == ALBUM_KIND_PHOTO_ESSAY:
        parts = [
            f"This page from {context.title}, a Photo Essay, contains {count} photo(s)."
        ]
    elif context.title:
        parts = [f"This page from {context.title} contains {count} photo(s)."]
    elif context.kind == ALBUM_KIND_FAMILY:
        parts = [f"This Family Photo Album page contains {count} photo(s)."]
    elif context.kind == ALBUM_KIND_PHOTO_ESSAY:
        parts = [f"This Photo Essay page contains {count} photo(s)."]
    else:
        parts = [f"This album page contains {count} photo(s)."]
    if not context.title and context.kind == ALBUM_KIND_PHOTO_ESSAY and context.focus:
        parts.append(f"The album title suggests {context.focus}.")
    if people_list and object_list:
        parts.append(
            f"Across the page, it shows {join_human(people_list)} with {join_human(object_list)} in view."
        )
    elif people_list:
        parts.append(f"Across the page, it shows {join_human(people_list)}.")
    elif object_list:
        parts.append(
            f"Across the page, visible objects include {join_human(object_list)}."
        )

    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible text on the page reads: "{snippet}".')
    return " ".join(parts).strip()


def _build_shared_prompt_rules(
    *,
    context: AlbumContext,
    source_path: str | Path | None,
    people_list: list[str],
    object_list: list[str],
    combined: bool = False,
    is_cover_page: bool = False,
) -> list[str]:
    lines: list[str] = []
    if is_cover_page:
        lines.append("This image is an album cover or title page.")
    if context.title:
        lines.append(f"Album title hint: {context.title}.")
    if (
        context.canonical_title
        and context.title
        and context.canonical_title.casefold() != context.title.casefold()
    ):
        lines.append(f"Canonical album title hint: {context.canonical_title}.")
        lines.append(
            "When naming the album in the caption, prefer the printed cover title over the normalized title."
        )
    if _should_apply_album_prompt_rules(source_path, context):
        lines.append("Cordell Photo Albums rules:")
        lines.append(
            "- If the album is a family collection, describe it as a Family Photo Album."
        )
        lines.append(
            "- If the album title names a country or region, describe it as a Photo Essay."
        )
        lines.append(
            "- If the image is mostly a solid blue or white cover with title text naming a country, "
            "region, or family, describe it as the cover of the photo album book."
        )
        lines.append(
            "- Preserve visible book labels exactly as shown. Do not silently normalize them. "
            "If a label uses digit 1 characters for a Roman numeral volume, keep the visible label "
            "and note that it is a typo; for example, BOOK 11 is a typo for Book II (2)."
        )
        lines.append(
            "- When quoting any visible text, preserve the original text as shown."
        )
        if context.label:
            lines.append(f"Album classification hint: {context.label}.")
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            lines.append(f"Album focus hint: {context.focus}.")
    lines.append(
        "Use decisive language. Never hedge with appears, seems, likely, or maybe."
    )
    lines.append(
        "Never mention raw file names, folder names, or internal IDs such as B02, P01, Archive, or View."
    )
    if combined:
        lines.append(
            "When the visible text contains non-English characters, copy them exactly in the ocr_text field. "
            "In the caption, follow each non-English phrase with its English translation in parentheses — "
            "for example: '时间：上午8—11时 (Time: 8–11 AM)'."
        )
    else:
        lines.append(
            "If any visible text is not in English, preserve the original characters exactly in the caption, "
            "then add an English translation in parentheses immediately after each non-English phrase — "
            "for example: '敦煌历史文物展览 (Dunhuang Historical Relics Exhibition)'."
        )
    lines.append(
        "Text visible in the image should make sense with the photo subjects: "
        "if a word appears cut off at a scan edge, misspelled, or truncated, "
        "infer the correct word from what is visible in the photo "
        "(e.g., 'Chendo' on a sign next to panda or red panda photos → 'Chengdu', word cut off at scan edge). "
        "Apply this to all text, not just place names."
    )
    lines.append("Location rules:")
    lines.append(
        "- Infer location from OCR text only when evidence is high confidence."
    )
    lines.append(
        "- When location is clear, name the landmark, town, province, and country."
    )
    lines.append(
        "- When evidence is imprecise, give the best city, state or province, and country."
    )
    lines.append(
        "- When evidence is weak or conflicting, say the location is uncertain."
    )
    lines.append(
        "- Do not invent GPS coordinates unless explicitly visible in the image or OCR text."
    )
    lines.append(
        "- Correct misspelled, outdated, or truncated place names using context clues (album region, photo content); "
        "words may be cut off at scan edges — use visible photo subjects to complete them."
    )
    lines.append(
        "- Only use place names for well-known, widely documented locations (cities, provinces, landmarks); "
        "avoid inferring obscure townships or villages — if you cannot confidently name a specific city, "
        "fall back to province and country."
    )
    lines.append(
        'Hyphen-separated lowercase names in OCR text (e.g. "leslie-tommy-robert") '
        "list people left to right: Leslie, Tommy, Robert."
    )
    if people_list:
        lines.append(f"Known people: {join_human(people_list)}.")
    if object_list:
        lines.append(f"Detected objects: {join_human(object_list)}.")
    return lines


def _build_qwen_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    photo_count: int = 1,
    is_cover_page: bool = False,
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
    if photo_count > 1:
        lines = [
            f"This album page contains {photo_count} separate photos arranged as a collage or grid.",
            "Describe each photo individually: what it shows, who or what is in it, and where it is located.",
            "Do not blend locations or subjects from different photos into a single description.",
        ]
    elif photo_count == 0:
        lines = [
            "This is a scan of an album page that may contain one or more individual photos.",
            "If you see multiple distinct photos, describe each one separately with its own location.",
            "Do not blend subjects or locations from different photos into a single description.",
        ]
    else:
        lines = ["Describe this photo in detail"]
    lines.extend(
        _build_shared_prompt_rules(
            context=context,
            source_path=source_path,
            people_list=people_list,
            object_list=object_list,
            is_cover_page=is_cover_page,
        )
    )
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.append(f'OCR text hint: "{snippet}".')
    lines.append(
        "Output a JSON object only. No markdown, no labels, no text outside the JSON."
    )
    lines.append(
        'Use this exact schema: {"caption": "...", "location_name": "...",'
        ' "gps_latitude": "...", "gps_longitude": "...", "name_suggestions": [...]}'
    )
    lines.append(
        "caption: a detailed description of the photo using only declarative statements."
    )
    lines.append(
        "location_name: a concise geocoding query like 'Mogao Caves, Dunhuang, Gansu, China', "
        "or empty string."
    )
    lines.append(
        "gps_latitude / gps_longitude: decimal degree strings only if exact coordinates are "
        "explicitly visible in the image or OCR text, otherwise empty strings."
    )
    lines.append(
        "name_suggestions: an array of objects with 'name', 'confidence', 'source', and 'context' fields. "
        "Extract names from visible text, labels, or contextual clues. "
        "Include names that appear in signs, documents, clothing, or other visible elements. "
        "Set confidence between 0.0 and 1.0 based on clarity and context. "
        "Set source to 'visible_text', 'contextual_clue', or 'label'. "
        "Set context to describe where the name was found."
    )
    return "\n".join(lines)


def _build_combined_qwen_prompt(
    *,
    people: list[str],
    objects: list[str],
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    photo_count: int = 1,
    is_cover_page: bool = False,
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
    if photo_count == 0:
        lines = [
            "This is a scan of an album page that may contain one or more individual photos. Do both tasks:",
            "1. Extract all visible text exactly as it appears. If there is none, write nothing.",
            "2. Write one sentence per photo; if multiple distinct photos are visible, describe each separately.",
        ]
    else:
        lines = [
            "Analyze this photo. Do both tasks:",
            "1. Extract all visible text exactly as it appears. If there is none, write nothing.",
            "2. Write one sentence describing the scene.",
        ]
    lines.extend(
        _build_shared_prompt_rules(
            context=context,
            source_path=source_path,
            people_list=people_list,
            object_list=object_list,
            combined=True,
            is_cover_page=is_cover_page,
        )
    )
    lines.append(
        "Output a JSON object only. No markdown, no labels, no text outside the JSON."
    )
    lines.append(
        'Use this exact schema: {"ocr_text": "...", "caption": "...", "location_name": "...",'
        ' "gps_latitude": "...", "gps_longitude": "...", "name_suggestions": [...]}'
    )
    lines.append(
        "ocr_text: all visible text in the image exactly as shown, or empty string if none."
    )
    lines.append(
        "caption: one sentence describing the scene using only declarative statements."
    )
    lines.append(
        "location_name: a concise geocoding query, or empty string if unknown."
    )
    lines.append(
        "gps_latitude / gps_longitude: decimal degree strings only if explicitly visible, "
        "otherwise empty strings."
    )
    lines.append(
        "name_suggestions: an array of objects with 'name', 'confidence', 'source', and 'context' fields. "
        "Extract names from visible text, labels, or contextual clues. "
        "Include names that appear in signs, documents, clothing, or other visible elements. "
        "Set confidence between 0.0 and 1.0 based on clarity and context. "
        "Set source to 'visible_text', 'contextual_clue', or 'label'. "
        "Set context to describe where the name was found."
    )
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
    photo_count: int,
    is_cover_page: bool,
) -> str:
    return prompt_text or _build_qwen_prompt(
        people=people,
        objects=objects,
        ocr_text=ocr_text,
        source_path=source_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        photo_count=photo_count,
        is_cover_page=is_cover_page,
    )
