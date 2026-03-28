from __future__ import annotations

import re
from typing import Final

ELLIPSIS_BOOK: Final[str] = "\u2026"
LEGACY_ELLIPSIS_BOOK: Final[str] = "\u00c3\u00a2\u00cb\u2020\u00a6"
UNKNOWN_BOOK_TOKENS: Final[set[str]] = {ELLIPSIS_BOOK, LEGACY_ELLIPSIS_BOOK}

_BOOK_TOKEN = rf"(?:\d{{2}}|{re.escape(ELLIPSIS_BOOK)}|{re.escape(LEGACY_ELLIPSIS_BOOK)})"

SCAN_TIFF_RE = re.compile(
    rf"^(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d{{2}})_S(?P<scan>\d{{2}})\.tif$",
    re.IGNORECASE,
)

SCAN_NAME_RE = re.compile(
    rf"(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d+)_S(?P<scan>\d+)",
    re.IGNORECASE,
)

DERIVED_NAME_RE = re.compile(
    rf"(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d+)_D(?P<derived>\d{{1,2}})_(?P<iter>\d{{1,2}})",
    re.IGNORECASE,
)

BASE_PAGE_NAME_RE = re.compile(
    rf"(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d+)",
    re.IGNORECASE,
)

PAGE_SCAN_RE = re.compile(r"_P(?P<page>\d+)_S(?P<scan>\d+)", re.IGNORECASE)

# Matches the stem of a single-scan view page: …_P##_V
VIEW_PAGE_RE = re.compile(r"_P\d+_V$", re.IGNORECASE)

# Matches the stem of a reconstructed (stitched) view page: …_P##_VC
VIEW_RECON_RE = re.compile(r"_P\d+_VC$", re.IGNORECASE)

# Legacy suffixes kept for transition-period recognition
VIEW_STITCHED_LEGACY_RE = re.compile(r"_P\d+_stitched$", re.IGNORECASE)
VIEW_RECON_LEGACY_RE = re.compile(r"_P\d+_VR$", re.IGNORECASE)


def parse_album_filename(
    filename: str,
    default: tuple[str, str, str, str] = ("Unknown", "Unknown", "00", "00"),
) -> tuple[str, str, str, str]:
    for pattern in (SCAN_NAME_RE, DERIVED_NAME_RE, BASE_PAGE_NAME_RE):
        match = pattern.search(filename)
        if match:
            return (
                str(match.group("collection")),
                str(match.group("year")),
                str(match.group("book")),
                str(match.group("page")),
            )
    return default


def format_book_display(book: str) -> str:
    raw = str(book or "").strip()
    if not raw:
        return "00"
    if raw in UNKNOWN_BOOK_TOKENS:
        return ELLIPSIS_BOOK
    try:
        return f"{int(raw):02d}"
    except ValueError:
        return raw
