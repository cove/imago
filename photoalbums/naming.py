from __future__ import annotations

import re
from pathlib import Path
from typing import Final

ELLIPSIS_BOOK: Final[str] = "\u2026"
LEGACY_ELLIPSIS_BOOK: Final[str] = "\u00c3\u00a2\u00cb\u2020\u00a6"
UNKNOWN_BOOK_TOKENS: Final[set[str]] = {ELLIPSIS_BOOK, LEGACY_ELLIPSIS_BOOK}
ALBUM_DIR_SUFFIX_ARCHIVE: Final[str] = "_Archive"
ALBUM_DIR_SUFFIX_PAGES: Final[str] = "_Pages"
ALBUM_DIR_SUFFIX_PHOTOS: Final[str] = "_Photos"
ALBUM_DIR_SUFFIXES: Final[tuple[str, str, str]] = (
    ALBUM_DIR_SUFFIX_ARCHIVE,
    ALBUM_DIR_SUFFIX_PAGES,
    ALBUM_DIR_SUFFIX_PHOTOS,
)

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
    rf"(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d+)_D(?P<derived>\d{{1,2}})-(?P<iter>\d{{1,2}})",
    re.IGNORECASE,
)

BASE_PAGE_NAME_RE = re.compile(
    rf"(?P<collection>[^_]+)_(?P<year>\d{{4}}(?:-\d{{4}})?)_B(?P<book>{_BOOK_TOKEN})_P(?P<page>\d+)",
    re.IGNORECASE,
)

PAGE_SCAN_RE = re.compile(r"_P(?P<page>\d+)_S(?P<scan>\d+)", re.IGNORECASE)

# Matches the stem of any view page: …_P##_V
VIEW_PAGE_RE = re.compile(r"_P\d+_V$", re.IGNORECASE)

# Matches the stem of a view derived image: …_D##-##_V
DERIVED_VIEW_RE = re.compile(r"_D\d{1,2}-\d{1,2}_V$", re.IGNORECASE)

# Legacy suffixes kept for transition-period recognition
VIEW_STITCHED_LEGACY_RE = re.compile(r"_P\d+_stitched$", re.IGNORECASE)
VIEW_RECON_LEGACY_RE = re.compile(r"_P\d+_VR$", re.IGNORECASE)
VIEW_VC_LEGACY_RE = re.compile(r"_P\d+_VC$", re.IGNORECASE)


def album_dir_suffix(path: str | Path) -> str:
    name = Path(path).name
    for suffix in ALBUM_DIR_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def album_dir_base_name(path: str | Path) -> str:
    name = Path(path).name
    suffix = album_dir_suffix(path)
    if not suffix:
        raise ValueError(f"Path is not an album directory with a canonical suffix: {path}")
    return name[: -len(suffix)]


def album_sibling_dir(path: str | Path, suffix: str) -> Path:
    base_path = Path(path)
    if suffix not in ALBUM_DIR_SUFFIXES:
        raise ValueError(f"Unsupported album directory suffix: {suffix}")
    return base_path.parent / f"{album_dir_base_name(base_path)}{suffix}"


def is_archive_dir(path: str | Path) -> bool:
    return album_dir_suffix(path) == ALBUM_DIR_SUFFIX_ARCHIVE


def is_pages_dir(path: str | Path) -> bool:
    return album_dir_suffix(path) == ALBUM_DIR_SUFFIX_PAGES


def is_photos_dir(path: str | Path) -> bool:
    return album_dir_suffix(path) == ALBUM_DIR_SUFFIX_PHOTOS


def archive_dir_for_album_dir(path: str | Path) -> Path:
    return album_sibling_dir(path, ALBUM_DIR_SUFFIX_ARCHIVE)


def pages_dir_for_album_dir(path: str | Path) -> Path:
    return album_sibling_dir(path, ALBUM_DIR_SUFFIX_PAGES)


def photos_dir_for_album_dir(path: str | Path) -> Path:
    return album_sibling_dir(path, ALBUM_DIR_SUFFIX_PHOTOS)


def parse_album_filename(
    filename: str,
    default: tuple[str, str, str, str] = ("Unknown", "Unknown", "00", "00"),
) -> tuple[str, str, str, str]:
    stem = Path(filename).stem
    for suffix in ("_stitched", "_VR", "_VC", "_V"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for pattern in (SCAN_NAME_RE, DERIVED_NAME_RE, BASE_PAGE_NAME_RE):
        match = pattern.fullmatch(stem)
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
