from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..naming import parse_album_filename
from .model_store import HF_MODEL_CACHE_DIR
from .ai_ocr import (
    DEFAULT_QWEN_OCR_MAX_IMAGE_EDGE,
    DEFAULT_QWEN_OCR_MAX_NEW_TOKENS,
    _normalize_ocr_text,
)


DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LEGACY_QWEN_CAPTION_MODEL_ALIASES = {
    "qwen/qwen3.5-4b": DEFAULT_QWEN_CAPTION_MODEL,
}
DEFAULT_QWEN_AUTO_MAX_PIXELS = 786_432
DEFAULT_LMSTUDIO_MAX_NEW_TOKENS = 256
DEFAULT_LMSTUDIO_BASE_URL = "http://192.168.4.72:1234/v1"
DEFAULT_LMSTUDIO_TIMEOUT_SECONDS = 180.0
DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE = 2048
LMSTUDIO_VISION_MODEL_HINTS = (
    "vl",
    "vision",
    "llava",
    "minicpm",
    "moondream",
    "pixtral",
    "internvl",
    "phi-3.5-vision",
    "phi-4-multimodal",
    "qvq",
)
QWEN_ATTN_IMPLEMENTATIONS = {"auto", "sdpa", "flash_attention_2", "eager"}
ALBUM_KIND_FAMILY = "family_photo_album"
ALBUM_KIND_PHOTO_ESSAY = "photo_essay"
_ALBUM_REGION_HINTS = (
    ("eastern europe", "Eastern Europe"),
    ("south america", "South America"),
    ("panama canal", "Panama Canal"),
    ("china", "China"),
    ("egypt", "Egypt"),
    ("england", "England"),
    ("europe", "Europe"),
    ("italy", "Italy"),
    ("morocco", "Morocco"),
    ("mexico", "Mexico"),
    ("orient", "Orient"),
    ("panama", "Panama"),
    ("portugal", "Portugal"),
    ("russia", "Russia"),
    ("spain", "Spain"),
)


@dataclass(frozen=True)
class AlbumContext:
    kind: str = ""
    label: str = ""
    focus: str = ""
    title: str = ""
    canonical_title: str = ""
    printed_title: str = ""


def clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def join_human(values: list[str]) -> str:
    clean = [str(item or "").strip() for item in values if str(item or "").strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split_camel_case(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))


def _humanize_hint_text(value: str) -> str:
    return clean_text(_split_camel_case(value).replace("_", " ").replace("-", " "))


def _normalized_hint_text(value: str) -> str:
    return _humanize_hint_text(value).casefold()


def _extract_collection_hint(image_path: str | Path | None) -> str:
    if image_path is None:
        return ""
    path = Path(image_path)
    if path.name:
        collection, _year, _book, _page = parse_album_filename(path.name)
        if collection != "Unknown":
            return _humanize_hint_text(collection)
    for candidate in (path.parent.name, path.parent.parent.name if path.parent != path else ""):
        text = str(candidate or "").strip()
        if not text:
            continue
        if text.lower().startswith("imago-page-"):
            continue
        for suffix in ("_Archive", "_View"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        collection, _year, _book, _page = parse_album_filename(text)
        if collection != "Unknown":
            return _humanize_hint_text(collection)
        match = re.search(r"(?P<collection>.+?)_\d{4}(?:-\d{4})?_B", text, flags=re.IGNORECASE)
        if match:
            return _humanize_hint_text(match.group("collection"))
        if text.casefold() not in {"photo albums", "photoalbums"}:
            return _humanize_hint_text(text)
    return ""


def _int_to_roman(value: int) -> str:
    number = int(value)
    if number <= 0:
        return ""
    numerals = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    parts: list[str] = []
    remaining = number
    for arabic, roman in numerals:
        while remaining >= arabic:
            parts.append(roman)
            remaining -= arabic
    return "".join(parts)


def _romanize_book_token(value: str) -> str:
    token = str(value or "").strip().upper()
    if not token:
        return ""
    normalized = token.replace("1", "I")
    if normalized and set(normalized) == {"I"}:
        return normalized
    if token.isdigit():
        return _int_to_roman(int(token))
    if re.fullmatch(r"[IVXLCDM]+", token):
        return token
    return token


def _humanize_album_title_text(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"[A-Z0-9&'().,/ -]+", text):
        words: list[str] = []
        for token in text.split():
            words.append(token.capitalize() if token.isalpha() else token)
        return " ".join(words)
    return text


def _extract_cover_title_text(ocr_text: str) -> str:
    lines = [clean_text(line) for line in str(ocr_text or "").splitlines() if clean_text(line)]
    if not lines:
        return ""
    title_parts: list[str] = []
    for line in lines:
        before_year = re.split(r"\b(?:19|20)\d{2}(?:-\d{4})?\b", line, maxsplit=1)[0]
        if clean_text(before_year) and before_year != line:
            title_parts.append(clean_text(before_year))
            break
        before_book = re.split(r"\bBOOK\b", line, maxsplit=1, flags=re.IGNORECASE)[0]
        if clean_text(before_book) and before_book != line:
            title_parts.append(clean_text(before_book))
            break
        if re.search(r"\bBOOK\b", line, flags=re.IGNORECASE):
            break
        if re.fullmatch(r"(?:19|20)\d{2}(?:-\d{4})?", line):
            break
        title_parts.append(line)
    return _humanize_album_title_text(" ".join(title_parts))


def _extract_book_label(ocr_text: str) -> str:
    match = re.search(
        r"\bBOOK\s+([1I]{1,6}|\d{1,3}|[IVXLCDM]+)\b",
        clean_text(ocr_text),
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    return str(match.group(1) or "").strip().upper()


def infer_printed_album_title(
    *,
    ocr_text: str = "",
    fallback_title: str = "",
) -> str:
    book_label = _extract_book_label(ocr_text)
    title_text = _extract_cover_title_text(ocr_text) if book_label else ""
    if title_text and book_label:
        return f"{title_text} Book {book_label}"
    if title_text:
        return title_text
    return clean_text(fallback_title)


def infer_album_title(
    *,
    image_path: str | Path | None = None,
    ocr_text: str = "",
    fallback_title: str = "",
    source_text: str = "",
) -> str:
    book_label = _extract_book_label(ocr_text)
    title_text = _extract_cover_title_text(ocr_text) if book_label else ""
    if title_text and book_label:
        book_display = _romanize_book_token(book_label)
        if book_display:
            return f"{title_text} Book {book_display}"
        return title_text
    fallback = clean_text(fallback_title)
    if image_path is None:
        return fallback
    path = Path(image_path)
    collection, _year, book, _page = parse_album_filename(path.name)
    collection_hint = _extract_collection_hint(path)
    source_name = str(source_text or "").split(";", 1)[0].strip()
    if source_name:
        source_collection, _source_year, source_book, _source_page = parse_album_filename(source_name)
        if collection == "Unknown" and source_collection != "Unknown":
            collection = source_collection
        if source_book and source_book != "00" and (not book or book == "00"):
            book = source_book
        if not collection_hint and source_collection != "Unknown":
            collection_hint = _humanize_hint_text(source_collection)
    if not collection_hint:
        return fallback
    book_display = _romanize_book_token(book)
    derived_title = f"{collection_hint} Book {book_display}" if book_display else collection_hint
    if fallback:
        fallback_has_book = bool(re.search(r"\bBook\b", fallback, flags=re.IGNORECASE))
        derived_has_book = bool(re.search(r"\bBook\b", derived_title, flags=re.IGNORECASE))
        if fallback_has_book != derived_has_book:
            return fallback if fallback_has_book else derived_title
        if len(fallback.split()) >= len(derived_title.split()):
            return fallback
    return derived_title or fallback


def _find_region_hints(*values: str) -> list[str]:
    haystack = " ".join(_normalized_hint_text(value) for value in values if str(value or "").strip())
    if not haystack:
        return []
    matches: list[str] = []
    seen: set[str] = set()
    for needle, label in _ALBUM_REGION_HINTS:
        pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
        if not re.search(pattern, haystack):
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        matches.append(label)
    return matches


def infer_album_context(
    *,
    image_path: str | Path | None = None,
    ocr_text: str = "",
    allow_ocr: bool = True,
    album_title: str = "",
    printed_album_title: str = "",
) -> AlbumContext:
    collection_hint = _extract_collection_hint(image_path)
    path_hint = _humanize_hint_text(str(image_path or ""))
    canonical_title = infer_album_title(
        image_path=image_path,
        ocr_text=ocr_text if allow_ocr else "",
        fallback_title=album_title,
    )
    printed_title = infer_printed_album_title(
        ocr_text=ocr_text if allow_ocr else "",
        fallback_title=printed_album_title,
    )
    title_hint = printed_title or canonical_title
    signals = [title_hint, canonical_title, printed_title, collection_hint, path_hint]
    if allow_ocr:
        signals.append(ocr_text)
    normalized = " ".join(_normalized_hint_text(value) for value in signals if str(value or "").strip())
    if not normalized:
        return AlbumContext(title=title_hint, canonical_title=canonical_title, printed_title=printed_title)
    if re.search(r"(?<![a-z])family(?![a-z])", normalized):
        return AlbumContext(
            kind=ALBUM_KIND_FAMILY,
            label="Family Photo Album",
            focus="Family",
            title=title_hint,
            canonical_title=canonical_title,
            printed_title=printed_title,
        )
    region_hints = _find_region_hints(*signals)
    if region_hints:
        return AlbumContext(
            kind=ALBUM_KIND_PHOTO_ESSAY,
            label="Photo Essay",
            focus=join_human(region_hints),
            title=title_hint,
            canonical_title=canonical_title,
            printed_title=printed_title,
        )
    return AlbumContext(title=title_hint, canonical_title=canonical_title, printed_title=printed_title)


def _should_apply_album_prompt_rules(source_path: str | Path | None, album_context: AlbumContext) -> bool:
    if album_context.kind:
        return True
    if source_path is None:
        return False
    joined = " ".join(str(part or "").casefold() for part in Path(source_path).parts)
    return "photo albums" in joined or "cordell" in joined


def _looks_like_uniform_cover_color(image_path: str | Path) -> bool:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
    except Exception:
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return False

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > 512:
        scale = 512.0 / float(longest)
        resized = cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    pixels = resized.reshape(-1, 3).astype("float32")
    if pixels.size == 0:
        return False

    mean = pixels.mean(axis=0)
    delta = np.abs(pixels - mean)
    uniform_ratio = float((delta.max(axis=1) <= 45.0).mean())
    blue_dominant = bool(mean[0] >= 95.0 and mean[0] >= mean[1] + 18.0 and mean[0] >= mean[2] + 18.0)
    white_dominant = bool(float(mean.min()) >= 170.0)
    return uniform_ratio >= 0.72 and (blue_dominant or white_dominant)


def looks_like_album_cover(
    image_path: str | Path,
    *,
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> bool:
    text = clean_text(ocr_text)
    if not text:
        return False
    context = album_context or infer_album_context(image_path=image_path, ocr_text=text, allow_ocr=True)
    if not context.kind:
        return False
    return _looks_like_uniform_cover_color(image_path)


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
    subject_prefix = f"This image from {context.title}" if context.title else "This photo"
    if people_list and object_list:
        parts.append(f"{subject_prefix} shows {join_human(people_list)} with {join_human(object_list)} in view.")
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
        parts = [f"This page from {context.title}, a Family Photo Album, contains {count} photo(s)."]
    elif context.title and context.kind == ALBUM_KIND_PHOTO_ESSAY:
        parts = [f"This page from {context.title}, a Photo Essay, contains {count} photo(s)."]
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
        parts.append(f"Across the page, it shows {join_human(people_list)} with {join_human(object_list)} in view.")
    elif people_list:
        parts.append(f"Across the page, it shows {join_human(people_list)}.")
    elif object_list:
        parts.append(f"Across the page, visible objects include {join_human(object_list)}.")

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
    if context.canonical_title and context.title and context.canonical_title.casefold() != context.title.casefold():
        lines.append(f"Canonical album title hint: {context.canonical_title}.")
        lines.append("When naming the album in the caption, prefer the printed cover title over the normalized title.")
    if _should_apply_album_prompt_rules(source_path, context):
        lines.append("Cordell Photo Albums rules:")
        lines.append("- If the album is a family collection, describe it as a Family Photo Album.")
        lines.append("- If the album title names a country or region, describe it as a Photo Essay.")
        lines.append(
            "- If the image is mostly a solid blue or white cover with title text naming a country, region, or family, describe it as the cover of the photo album book."
        )
        lines.append(
            "- Preserve visible book labels exactly as shown. Do not silently normalize them. If a label uses digit 1 characters for a Roman numeral volume, keep the visible label and note that it is a typo; for example, BOOK 11 is a typo for Book II (2)."
        )
        lines.append("- When quoting any visible text, preserve the original text as shown.")
        if context.label:
            lines.append(f"Album classification hint: {context.label}.")
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            lines.append(f"Album focus hint: {context.focus}.")
    lines.append("Use decisive language. Never hedge with appears, seems, likely, or maybe.")
    lines.append("Never mention raw file names, folder names, or internal IDs such as B02, P01, Archive, or View.")
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
    lines.append("- Infer location from OCR text only when evidence is high confidence.")
    lines.append("- When location is clear, name the landmark, town, province, and country.")
    lines.append("- When evidence is imprecise, give the best city, state or province, and country.")
    lines.append("- When evidence is weak or conflicting, say the location is uncertain.")
    lines.append("- Do not invent GPS coordinates unless explicitly visible in the image or OCR text.")
    lines.append(
        "- Correct misspelled, outdated, or truncated place names using context clues (album region, photo content); "
        "words may be cut off at scan edges — use visible photo subjects to complete them."
    )
    lines.append(
        "- Only use place names for well-known, widely documented locations (cities, provinces, landmarks); "
        "avoid inferring obscure townships or villages — if you cannot confidently name a specific city, fall back to province and country."
    )
    lines.append(
        'Hyphen-separated lowercase names in OCR text (e.g. "leslie-tommy-robert") list people left to right: Leslie, Tommy, Robert.'
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
    lines.extend(_build_shared_prompt_rules(
        context=context,
        source_path=source_path,
        people_list=people_list,
        object_list=object_list,
        is_cover_page=is_cover_page,
    ))
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.append(f'OCR text hint: "{snippet}".')
    lines.append("Output a JSON object only. No markdown, no labels, no text outside the JSON.")
    lines.append('Use this exact schema: {"caption": "...", "location_name": "...", "gps_latitude": "...", "gps_longitude": "..."}')
    lines.append("caption: a detailed description of the photo using only declarative statements.")
    lines.append("location_name: a concise geocoding query like 'Mogao Caves, Dunhuang, Gansu, China', or empty string.")
    lines.append("gps_latitude / gps_longitude: decimal degree strings only if exact coordinates are explicitly visible in the image or OCR text, otherwise empty strings.")
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
    lines.extend(_build_shared_prompt_rules(
        context=context,
        source_path=source_path,
        people_list=people_list,
        object_list=object_list,
        combined=True,
        is_cover_page=is_cover_page,
    ))
    lines.append("Output a JSON object only. No markdown, no labels, no text outside the JSON.")
    lines.append('Use this exact schema: {"ocr_text": "...", "caption": "...", "location_name": "...", "gps_latitude": "...", "gps_longitude": "..."}')
    lines.append("ocr_text: all visible text in the image exactly as shown, or empty string if none.")
    lines.append("caption: one sentence describing the scene using only declarative statements.")
    lines.append("location_name: a concise geocoding query, or empty string if unknown.")
    lines.append("gps_latitude / gps_longitude: decimal degree strings only if explicitly visible, otherwise empty strings.")
    return "\n".join(lines)



def resolve_caption_model(engine: str, model_name: str) -> str:
    normalized = str(engine or "").strip().lower()
    if normalized == "blip":
        normalized = "qwen"
    text = str(model_name or "").strip()
    if text and normalized == "qwen":
        alias = LEGACY_QWEN_CAPTION_MODEL_ALIASES.get(text.casefold())
        if alias:
            return alias
    if text:
        return text
    if normalized == "qwen":
        return DEFAULT_QWEN_CAPTION_MODEL
    return ""


def normalize_qwen_attn_implementation(value: str, default: str = "auto") -> str:
    text = str(value or "").strip().lower()
    if text in QWEN_ATTN_IMPLEMENTATIONS:
        return text
    fallback = str(default or "auto").strip().lower()
    if fallback in QWEN_ATTN_IMPLEMENTATIONS:
        return fallback
    return "auto"


def _resize_caption_image(image, max_image_edge: int):
    if int(max_image_edge) <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= int(max_image_edge):
        return image
    scale = float(max_image_edge) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resampling = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resampling is None:
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        except Exception:  # pragma: no cover - Pillow always present in runtime
            resampling = 1
    return image.resize(new_size, resampling)


def normalize_lmstudio_base_url(value: str, default: str = DEFAULT_LMSTUDIO_BASE_URL) -> str:
    text = str(value or "").strip() or str(default or DEFAULT_LMSTUDIO_BASE_URL)
    text = text.rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


def _resolve_local_hf_snapshot(model_name: str) -> Path | None:
    text = str(model_name or "").strip()
    if "/" not in text:
        return None
    repo_dir = HF_MODEL_CACHE_DIR / f"models--{text.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    for snapshot in sorted(repo_dir.iterdir()):
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").exists() and (
            (snapshot / "preprocessor_config.json").exists() or (snapshot / "processor_config.json").exists()
        ):
            return snapshot
    return None


def _load_qwen_transformers():
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except Exception as exc:
        raise RuntimeError(
            "Qwen captioning requires a compatible transformers/torch install."
        ) from exc

    return torch, AutoProcessor, AutoModelForImageTextToText


def _build_data_url(image_path: str | Path, max_image_edge: int) -> str:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    path = Path(image_path)
    image = Image.open(str(path)).convert("RGB")
    try:
        working_image = _resize_caption_image(image, max_image_edge)
        buffer = io.BytesIO()
        if path.suffix.lower() == ".png":
            mime = "image/png"
            working_image.save(buffer, format="PNG")
        else:
            mime = "image/jpeg"
            working_image.save(buffer, format="JPEG", quality=95)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{data}"
    finally:
        if "working_image" in locals() and working_image is not image:
            working_image.close()
        image.close()


def _decode_lmstudio_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _lmstudio_caption_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "caption_payload",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "gps_latitude": {"type": "string"},
                    "gps_longitude": {"type": "string"},
                    "location_name": {"type": "string"},
                },
                "required": ["caption", "gps_latitude", "gps_longitude", "location_name"],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_error_preview(value: str, *, limit: int = 180) -> str:
    text = clean_text(value)
    if not text:
        return "<empty>"
    snippet = text[: max(1, int(limit))]
    if len(text) > len(snippet):
        snippet += "..."
    return snippet


def _extract_structured_json_payload(text: str) -> dict[str, object] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None



def _format_decimal_coordinate(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_dms_coordinate(value: str) -> float | None:
    match = re.search(
        r"(?P<deg>\d{1,3})\s*[°º]\s*(?P<min>\d{1,2})\s*[′']\s*(?P<sec>\d{1,2}(?:\.\d+)?)?\s*[″\"]?\s*(?P<hem>[NSEW])",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    deg = float(match.group("deg"))
    minutes = float(match.group("min"))
    seconds = float(match.group("sec") or 0.0)
    hemisphere = str(match.group("hem") or "").upper()
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _parse_decimal_coordinate(value: str, *, positive_hemisphere: str, negative_hemisphere: str) -> float | None:
    text = clean_text(value).replace("−", "-")
    if not text:
        return None
    match = re.fullmatch(
        rf"(?P<number>[+-]?\d{{1,3}}(?:\.\d+)?)\s*(?P<hem>[{positive_hemisphere}{negative_hemisphere}])?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    decimal = float(match.group("number"))
    hemisphere = str(match.group("hem") or "").upper()
    if hemisphere == negative_hemisphere.upper():
        decimal = -abs(decimal)
    elif hemisphere == positive_hemisphere.upper():
        decimal = abs(decimal)
    return decimal


def _normalize_gps_value(value: str, *, axis: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    decimal: float | None = None
    if any(symbol in text for symbol in ("°", "º", "′", "″", "'")):
        decimal = _parse_dms_coordinate(text)
    if decimal is None:
        if axis == "lat":
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="N", negative_hemisphere="S")
        else:
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="E", negative_hemisphere="W")
    if decimal is None:
        return ""
    return _format_decimal_coordinate(decimal)



def _parse_lmstudio_structured_caption_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> tuple[str, list[tuple[str, str]], str, str, str]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured caption content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    # Strip <think>...</think> blocks produced by reasoning models so that intermediate
    # JSON objects inside the thinking block are not mistaken for the structured response.
    # If stripping empties the text, keep the original so _extract_structured_json_payload
    # can still find JSON embedded inside an unclosed <think> block.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        text = stripped
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(text)
        if payload is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid structured caption JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not isinstance(payload, dict):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio returned structured caption JSON that is not an object; raw={preview!r}.{finish_note}"
        )
    caption = payload.get("caption")
    if not isinstance(caption, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio structured caption JSON is missing a caption string; raw={preview!r}.{finish_note}"
        )
    gps_latitude = _normalize_gps_value(str(payload.get("gps_latitude") or ""), axis="lat")
    gps_longitude = _normalize_gps_value(str(payload.get("gps_longitude") or ""), axis="lon")
    location_name = clean_text(payload.get("location_name"))
    return caption, gps_latitude, gps_longitude, location_name


@dataclass(frozen=True)
class CaptionDetails:
    text: str
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""

    def __str__(self) -> str:
        return self.text

    def __contains__(self, item: object) -> bool:
        return str(item or "") in self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CaptionDetails):
            return (
                self.text == other.text
                and self.gps_latitude == other.gps_latitude
                and self.gps_longitude == other.gps_longitude
                and self.location_name == other.location_name
            )
        if isinstance(other, str):
            return self.text == other
        return False


_CAPTION_REASONING_MARKERS = (
    "the user wants",
    "analyze the input data",
    "visual analysis",
    "synthesize the visual content",
    "filename hint",
    "folder hint",
    "album title hint",
    "ocr/text in image",
    "ocr text hint",
    "detected objects",
    "album classification hint",
    "album focus hint",
    "cordell photo albums rules",
    "based on the rules",
    "i need to",
    "i should",
)


def _looks_like_reasoning_or_prompt_echo(value: str) -> bool:
    """Return True if the text looks like a model reasoning trace or prompt echo rather than a caption."""
    text = clean_text(value)
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _CAPTION_REASONING_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if text.startswith(("**1.", "1.", "* **filename:**", "- **filename:**")):
        return True
    if re.search(r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:filename|folder|ocr/text in image|detected objects)\s*:", lowered):
        return True
    if re.search(r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:album classification hint|album focus hint)\s*:", lowered):
        return True
    return False


def _parse_qwen_json_output(raw: str) -> CaptionDetails:
    """Parse structured JSON output from a Qwen model inference, with plain-text fallback."""
    text = str(raw or "").strip()
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        text = stripped
    payload = _extract_structured_json_payload(text)
    if payload is not None:
        caption = payload.get("caption")
        if isinstance(caption, str) and caption.strip():
            gps_latitude = _normalize_gps_value(str(payload.get("gps_latitude") or ""), axis="lat")
            gps_longitude = _normalize_gps_value(str(payload.get("gps_longitude") or ""), axis="lon")
            location_name = clean_text(payload.get("location_name"))
            return CaptionDetails(
                text=clean_text(caption),
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
                location_name=location_name,
            )
    return CaptionDetails(text=clean_text(text))


def _parse_qwen_combined_json_output(raw: str) -> tuple[str, str]:
    """Parse structured JSON output from a combined OCR+caption Qwen inference.
    Returns (ocr_text, caption_text).
    """
    text = str(raw or "").strip()
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        text = stripped
    payload = _extract_structured_json_payload(text)
    if payload is not None:
        ocr_text = _normalize_ocr_text(str(payload.get("ocr_text") or ""))
        caption = payload.get("caption")
        if isinstance(caption, str) and caption.strip():
            return ocr_text, clean_text(caption)
    return "", clean_text(text)


def _parse_lmstudio_structured_caption(
    value: object,
    *,
    finish_reason: str = "",
) -> CaptionDetails:
    caption, gps_latitude, gps_longitude, location_name = _parse_lmstudio_structured_caption_payload(
        value,
        finish_reason=finish_reason,
    )
    return CaptionDetails(
        text=clean_text(caption),
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_name=location_name,
    )


def _lmstudio_request_json(url: str, *, payload: dict | None = None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or f"HTTP {exc.code}"
        raise RuntimeError(f"LM Studio request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _lmstudio_stream_tokens(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = list(chunk.get("choices") or [])
                if not choices:
                    continue
                delta = dict(choices[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    yield str(content)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or f"HTTP {exc.code}"
        raise RuntimeError(f"LM Studio request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _select_lmstudio_model(base_url: str, requested_model: str, timeout: float) -> str:
    text = str(requested_model or "").strip()
    if text:
        return text
    payload = _lmstudio_request_json(f"{base_url}/models", timeout=timeout)
    model_ids = [
        str(row.get("id") or "").strip()
        for row in list(payload.get("data") or [])
        if str(row.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError("LM Studio did not return any models. Load a model or pass --caption-model.")
    for model_id in model_ids:
        lowered = model_id.casefold()
        if any(hint in lowered for hint in LMSTUDIO_VISION_MODEL_HINTS):
            return model_id
    return model_ids[0]


class LMStudioCaptioner:
    def __init__(
        self,
        *,
        model_name: str = "",
        prompt_text: str = "",
        max_new_tokens: int = DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
        temperature: float = 0.2,
        base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        timeout_seconds: float = DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
        max_image_edge: int = 0,
        stream: bool = False,
    ):
        self.model_name = str(model_name or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.base_url = normalize_lmstudio_base_url(base_url)
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_image_edge = max(0, int(max_image_edge))
        self.stream = bool(stream)
        self._resolved_model_name = ""

    def _resolve_model_name(self) -> str:
        if self._resolved_model_name:
            return self._resolved_model_name
        self._resolved_model_name = _select_lmstudio_model(
            self.base_url,
            self.model_name,
            self.timeout_seconds,
        )
        return self._resolved_model_name

    def describe(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> CaptionDetails:
        prompt = self.prompt_text or _build_qwen_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a photo caption writer. "
                        "Return only valid JSON matching the response_format schema. "
                        "Put the final caption text in the caption field. "
                        "If any visible text is not in English, add an English translation in parentheses directly after each non-English phrase in the caption — for example: '敦煌历史文物展览 (Dunhuang Historical Relics Exhibition)'. "
                        "If the location is known confidently enough for online geocoding, set location_name to a concise English geocoding query such as 'Mogao Caves, Dunhuang, Gansu, China'. "
                        "Only set gps_latitude and gps_longitude when exact coordinates are explicitly visible in the image or OCR text. "
                        "If the exact GPS is not explicitly known, set both GPS fields to empty strings. "
                        "If no confident geocoding query is available, set location_name to an empty string. "
                        "Never mention raw filenames, folder names, or internal ids. "
                        "Do not include reasoning or extra fields."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "response_format": _lmstudio_caption_response_format(),
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "stream": self.stream,
        }
        if self.stream:
            print(f"  Running LM Studio model ({self._resolve_model_name()})...", end="", flush=True)
            tokens: list[str] = []
            for token in _lmstudio_stream_tokens(
                f"{self.base_url}/chat/completions",
                payload,
                self.timeout_seconds,
            ):
                tokens.append(token)
            print(f"\r\033[K", end="", flush=True)
            return _parse_lmstudio_structured_caption(
                "".join(tokens),
            )
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        return _parse_lmstudio_structured_caption(
            message.get("content"),
            finish_reason=str(choices[0].get("finish_reason") or ""),
        )


class QwenLocalCaptioner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_CAPTION_MODEL,
        prompt_text: str = "",
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        attn_implementation: str = "auto",
        min_pixels: int = 0,
        max_pixels: int = 0,
        max_image_edge: int = 0,
        stream: bool = False,
    ):
        self.model_name = str(model_name or DEFAULT_QWEN_CAPTION_MODEL).strip() or DEFAULT_QWEN_CAPTION_MODEL
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.attn_implementation = normalize_qwen_attn_implementation(attn_implementation)
        self.min_pixels = max(0, int(min_pixels))
        self.max_pixels = max(0, int(max_pixels))
        self.max_image_edge = max(0, int(max_image_edge))
        self.stream = bool(stream)
        self._processor = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        torch, AutoProcessor, AutoModelForImageTextToText = _load_qwen_transformers()

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        local_snapshot = _resolve_local_hf_snapshot(self.model_name)
        model_ref = str(local_snapshot) if local_snapshot is not None else self.model_name
        local_files_only = local_snapshot is not None
        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if self.min_pixels > 0:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        processor_kwargs["max_pixels"] = (
            int(self.max_pixels) if self.max_pixels > 0 else int(DEFAULT_QWEN_AUTO_MAX_PIXELS)
        )
        self._processor = AutoProcessor.from_pretrained(
            model_ref,
            **processor_kwargs,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        resolved_attn = "auto"
        if self.attn_implementation != "auto":
            if self.attn_implementation == "flash_attention_2" and not torch.cuda.is_available():
                resolved_attn = "auto"
            else:
                resolved_attn = self.attn_implementation
                load_kwargs["attn_implementation"] = resolved_attn
        # Prefer dtype over torch_dtype to avoid deprecation warnings on newer transformers.
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_ref,
                dtype="auto",
                **load_kwargs,
            )
        except TypeError:
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_ref,
                torch_dtype="auto",
                **load_kwargs,
            )
        self._torch = torch

    def describe(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> CaptionDetails:
        self._ensure_loaded()
        prompt = self.prompt_text or _build_qwen_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        return _parse_qwen_json_output(self._infer_raw(image_path, prompt))

    def _infer_raw(self, image_path: str | Path, prompt: str, max_new_tokens: int | None = None) -> str:
        """Run a single inference pass and return the raw decoded string."""
        from PIL import Image  # pylint: disable=import-outside-toplevel

        max_tokens = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        image = Image.open(str(image_path)).convert("RGB")
        try:
            working_image = _resize_caption_image(image, int(DEFAULT_QWEN_OCR_MAX_IMAGE_EDGE))
            if hasattr(self._processor, "apply_chat_template"):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                try:
                    prompt_text = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    prompt_text = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            else:
                prompt_text = prompt
            inputs = self._processor(text=[prompt_text], images=[working_image], padding=True, return_tensors="pt")
            device = getattr(self._model, "device", None)
            if device is not None:
                for key, value in list(inputs.items()):
                    if hasattr(value, "to"):
                        inputs[key] = value.to(device)
            do_sample = self.temperature > 0
            kwargs: dict = {"max_new_tokens": max_tokens, "do_sample": do_sample}
            if do_sample:
                kwargs["temperature"] = self.temperature
                kwargs["top_p"] = 0.9
            if self.stream:
                import threading  # pylint: disable=import-outside-toplevel
                from transformers import TextIteratorStreamer  # pylint: disable=import-outside-toplevel

                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                kwargs["streamer"] = streamer
                thread = threading.Thread(target=self._model.generate, kwargs={**inputs, **kwargs}, daemon=True)
                tokens: list[str] = []
                with self._torch.inference_mode():
                    thread.start()
                    for token in streamer:
                        tokens.append(token)
                        partial = "".join(tokens)
                        display = partial[-120:] if len(partial) > 120 else partial
                        print(f"\r  {display}", end="", flush=True)
                    thread.join()
                print(f"\r\033[K", end="", flush=True)
                return "".join(tokens)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **kwargs)
            input_ids = inputs.get("input_ids")
            if hasattr(generated_ids, "shape") and input_ids is not None and hasattr(input_ids, "shape"):
                prompt_tokens = int(input_ids.shape[-1])
                generated_ids = generated_ids[:, prompt_tokens:]
            decoded = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
            return decoded[0] if decoded else ""
        finally:
            if "working_image" in locals() and working_image is not image:
                working_image.close()
            image.close()

    def describe_combined(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> tuple[str, str]:
        """Single inference that returns (ocr_text, caption)."""
        self._ensure_loaded()
        prompt = _build_combined_qwen_prompt(
            people=people,
            objects=objects,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        max_tokens = self.max_new_tokens + DEFAULT_QWEN_OCR_MAX_NEW_TOKENS
        raw = self._infer_raw(image_path, prompt, max_new_tokens=max_tokens)
        return _parse_qwen_combined_json_output(raw)


@dataclass
class CaptionOutput:
    text: str
    engine: str
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""
    fallback: bool = False
    error: str = ""


class CaptionEngine:
    def __init__(
        self,
        *,
        engine: str = "qwen",
        model_name: str = "",
        caption_prompt: str = "",
        max_tokens: int = 96,
        temperature: float = 0.2,
        qwen_attn_implementation: str = "auto",
        qwen_min_pixels: int = 0,
        qwen_max_pixels: int = 0,
        lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        max_image_edge: int = 0,
        fallback_to_template: bool = True,
        stream: bool = False,
    ):
        normalized = str(engine or "qwen").strip().lower()
        if normalized == "blip":
            normalized = "qwen"
        if normalized not in {"none", "template", "qwen", "lmstudio"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self.fallback_to_template = bool(fallback_to_template)
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._caption_prompt = str(caption_prompt or "").strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._qwen_attn_implementation = normalize_qwen_attn_implementation(qwen_attn_implementation)
        self._qwen_min_pixels = max(0, int(qwen_min_pixels))
        self._qwen_max_pixels = max(0, int(qwen_max_pixels))
        self._lmstudio_base_url = normalize_lmstudio_base_url(lmstudio_base_url)
        self._max_image_edge = max(0, int(max_image_edge))
        self._stream = bool(stream)

    def _ensure_captioner(self) -> None:
        if self._captioner is not None:
            return
        if self.engine == "lmstudio":
            self._captioner = LMStudioCaptioner(
                model_name=self._model_name,
                prompt_text=self._caption_prompt,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                base_url=self._lmstudio_base_url,
                max_image_edge=self._max_image_edge,
                stream=self._stream,
            )
        else:
            self._captioner = QwenLocalCaptioner(
                model_name=self._model_name,
                prompt_text=self._caption_prompt,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                attn_implementation=self._qwen_attn_implementation,
                min_pixels=self._qwen_min_pixels,
                max_pixels=self._qwen_max_pixels,
                max_image_edge=self._max_image_edge,
                stream=self._stream,
            )

    def generate(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> CaptionOutput:
        context = infer_album_context(
            image_path=source_path or image_path,
            ocr_text=ocr_text,
            allow_ocr=True,
            album_title=album_title,
            printed_album_title=printed_album_title,
        )
        template = build_template_caption(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            album_context=context,
        )
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        if self.engine == "template":
            return CaptionOutput(
                text=template,
                engine="template",
                gps_latitude="",
                gps_longitude="",
                location_name="",
            )
        self._ensure_captioner()
        try:
            caption = self._captioner.describe(
                image_path=image_path,
                people=people,
                objects=objects,
                ocr_text=ocr_text,
                source_path=source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=photo_count,
                is_cover_page=is_cover_page,
            )
            if caption.text:
                return CaptionOutput(
                    text=caption.text,
                    engine=self.engine,
                    gps_latitude=caption.gps_latitude,
                    gps_longitude=caption.gps_longitude,
                    location_name=caption.location_name,
                )
            if not self.fallback_to_template:
                return CaptionOutput(
                    text="",
                    engine=self.engine,
                    fallback=True,
                    error=f"{self.engine.upper()} returned empty output.",
                )
            return CaptionOutput(
                text=template,
                engine="template",
                gps_latitude="",
                gps_longitude="",
                location_name="",
                fallback=True,
                error=f"{self.engine.upper()} returned empty output.",
            )
        except Exception as exc:
            if not self.fallback_to_template:
                raise
            return CaptionOutput(text=template, engine="template", fallback=True, error=str(exc))

    def generate_combined(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> tuple[CaptionOutput, str]:
        """Single Qwen inference for both OCR and caption. Returns (CaptionOutput, ocr_text).
        Only valid when engine == 'qwen'. Falls back to empty ocr_text on error."""
        if self.engine != "qwen":
            return CaptionOutput(text="", engine=self.engine, fallback=True, error="generate_combined requires qwen engine"), ""
        self._ensure_captioner()
        try:
            ocr_text, caption = self._captioner.describe_combined(
                image_path=image_path,
                people=people,
                objects=objects,
                source_path=source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=photo_count,
                is_cover_page=is_cover_page,
            )
            if caption:
                return CaptionOutput(
                    text=caption,
                    engine=self.engine,
                    gps_latitude="",
                    gps_longitude="",
                    location_name="",
                ), ocr_text
            template = build_template_caption(
                people=people,
                objects=[],
                ocr_text=ocr_text,
                album_context=infer_album_context(
                    image_path=source_path or image_path,
                    ocr_text=ocr_text,
                    allow_ocr=True,
                    album_title=album_title,
                    printed_album_title=printed_album_title,
                ),
            )
            return CaptionOutput(
                text=template, engine="template", fallback=True,
                error="Qwen combined returned empty description.",
            ), ocr_text
        except Exception as exc:
            return CaptionOutput(text="", engine=self.engine, fallback=True, error=str(exc)), ""
