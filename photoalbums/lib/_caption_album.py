from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..naming import parse_album_filename

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
    for candidate in (
        path.parent.name,
        path.parent.parent.name if path.parent != path else "",
    ):
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
        match = re.search(
            r"(?P<collection>.+?)_\d{4}(?:-\d{4})?_B", text, flags=re.IGNORECASE
        )
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
    lines = [
        clean_text(line)
        for line in str(ocr_text or "").splitlines()
        if clean_text(line)
    ]
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
        source_collection, _source_year, source_book, _source_page = (
            parse_album_filename(source_name)
        )
        if collection == "Unknown" and source_collection != "Unknown":
            collection = source_collection
        if source_book and source_book != "00" and (not book or book == "00"):
            book = source_book
        if not collection_hint and source_collection != "Unknown":
            collection_hint = _humanize_hint_text(source_collection)
    if not collection_hint:
        return fallback
    book_display = _romanize_book_token(book)
    derived_title = (
        f"{collection_hint} Book {book_display}" if book_display else collection_hint
    )
    if fallback:
        fallback_has_book = bool(re.search(r"\bBook\b", fallback, flags=re.IGNORECASE))
        derived_has_book = bool(
            re.search(r"\bBook\b", derived_title, flags=re.IGNORECASE)
        )
        if fallback_has_book != derived_has_book:
            return fallback if fallback_has_book else derived_title
        if len(fallback.split()) >= len(derived_title.split()):
            return fallback
    return derived_title or fallback


def _find_region_hints(*values: str) -> list[str]:
    haystack = " ".join(
        _normalized_hint_text(value) for value in values if str(value or "").strip()
    )
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
    normalized = " ".join(
        _normalized_hint_text(value) for value in signals if str(value or "").strip()
    )
    if not normalized:
        return AlbumContext(
            title=title_hint,
            canonical_title=canonical_title,
            printed_title=printed_title,
        )
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
    return AlbumContext(
        title=title_hint, canonical_title=canonical_title, printed_title=printed_title
    )


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
    blue_dominant = bool(
        mean[0] >= 95.0 and mean[0] >= mean[1] + 18.0 and mean[0] >= mean[2] + 18.0
    )
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
    context = album_context or infer_album_context(
        image_path=image_path, ocr_text=text, allow_ocr=True
    )
    if not context.kind:
        return False
    return _looks_like_uniform_cover_color(image_path)
