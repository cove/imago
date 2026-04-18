from __future__ import annotations

from pathlib import Path

from ..naming import (
    ALBUM_DIR_SUFFIX_ARCHIVE,
    ALBUM_DIR_SUFFIX_PAGES,
    BASE_PAGE_NAME_RE,
    DERIVED_NAME_RE,
    SCAN_NAME_RE,
    album_sibling_dir,
    album_dir_suffix,
    parse_album_filename,
)
from .ai_caption import clean_text
from .xmp_sidecar import read_ai_sidecar_state


def _album_identity_key(image_path: Path) -> str:
    collection, year, book, _page = parse_album_filename(image_path.name)
    if collection != "Unknown":
        return f"{collection}_{year}_B{book}".casefold()
    parent = image_path.parent
    if album_dir_suffix(parent):
        return str(album_sibling_dir(parent, ALBUM_DIR_SUFFIX_ARCHIVE).resolve()).casefold()
    return str(parent.resolve()).casefold()


def _album_directory_candidates(image_path: Path) -> list[Path]:
    out: list[Path] = [image_path.parent]
    if not album_dir_suffix(image_path.parent):
        return out
    for suffix in (ALBUM_DIR_SUFFIX_ARCHIVE, ALBUM_DIR_SUFFIX_PAGES):
        candidate = album_sibling_dir(image_path.parent, suffix)
        if candidate in out or not candidate.is_dir():
            continue
        out.append(candidate)
    return out


def _iter_album_cover_sidecars(image_path: Path):
    collection, year, book, _page = parse_album_filename(image_path.name)
    target_prefix = ""
    if collection != "Unknown":
        target_prefix = f"{collection}_{year}_B{book}_".casefold()
    seen: set[str] = set()
    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for folder in _album_directory_candidates(image_path):
        for sidecar_path in sorted(folder.glob("*.xmp")):
            match = _cover_sidecar_match(sidecar_path)
            if match is None:
                continue
            if target_prefix and not Path(sidecar_path).stem.casefold().startswith(target_prefix):
                continue
            sidecar_key = str(sidecar_path.resolve()).casefold()
            if sidecar_key in seen:
                continue
            seen.add(sidecar_key)
            page_rank = int(match.group("page"))
            scan_match = _scan_name_match(sidecar_path)
            kind_rank = 1 if scan_match is not None else 0
            candidates.append(((page_rank, kind_rank, sidecar_path.name.casefold()), sidecar_path))
    for _sort_key, sidecar_path in sorted(candidates, key=lambda item: item[0]):
        yield sidecar_path


def _iter_album_p01_sidecars(image_path: Path):
    for sidecar_path in _iter_album_cover_sidecars(image_path):
        match = _cover_sidecar_match(sidecar_path)
        if match is None:
            continue
        if int(match.group("page")) == 1:
            yield sidecar_path


def _scan_name_match(path: str | Path):
    return SCAN_NAME_RE.fullmatch(Path(path).stem)


def _derived_name_match(path: str | Path):
    return DERIVED_NAME_RE.fullmatch(Path(path).stem)


def _base_page_name_match(path: str | Path):
    return BASE_PAGE_NAME_RE.fullmatch(Path(path).stem)


def _title_page_scan_match(path: str | Path):
    match = _scan_name_match(path)
    if match is None:
        return None
    try:
        page_number = int(match.group("page"))
        scan_number = int(match.group("scan"))
    except (ValueError, IndexError):
        return None
    if page_number == 1 and scan_number == 1:
        return match
    return None


def _title_page_base_match(path: str | Path):
    match = _base_page_name_match(path)
    if match is None:
        return None
    try:
        page_number = int(match.group("page"))
    except (ValueError, IndexError):
        return None
    if page_number == 1:
        return match
    return None


def _title_page_match(path: str | Path):
    return _title_page_scan_match(path) or _title_page_base_match(path)


def _cover_sidecar_match(path: str | Path):
    return _title_page_match(path)


def _title_page_dependency_sort_key(path: Path) -> tuple[int, int, int, int, str]:
    _, _, _, page = parse_album_filename(path.name)
    try:
        page_number = int(str(page or "").strip())
    except ValueError:
        page_number = 999
    scan_match = _scan_name_match(path)
    derived_match = _derived_name_match(path)
    if scan_match is not None:
        kind_rank = 0
        item_number = int(scan_match.group("scan"))
    elif derived_match is None:
        kind_rank = 1
        item_number = 0
    else:
        kind_rank = 2
        item_number = int(derived_match.group("derived"))
    return (
        page_number,
        kind_rank,
        item_number,
        len(path.name),
        path.name.casefold(),
    )


def _is_album_title_source_candidate(image_path: Path) -> bool:
    return _title_page_match(image_path) is not None


def _iter_album_title_page_images(image_path: Path, extensions: set[str]):
    seen: set[str] = set()
    album_key = _album_identity_key(image_path)
    candidates: list[Path] = []
    for folder in _album_directory_candidates(image_path):
        try:
            rows = list(folder.iterdir())
        except FileNotFoundError:
            continue
        for candidate in rows:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in extensions:
                continue
            if _album_identity_key(candidate) != album_key:
                continue
            if _title_page_match(candidate) is None:
                continue
            key = str(candidate.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    for candidate in sorted(candidates, key=_title_page_dependency_sort_key):
        yield candidate


def _resolve_album_title_dependencies(image_path: Path, extensions: set[str]) -> list[Path]:
    if _is_album_title_source_candidate(image_path):
        return []
    if _album_title_valid_in_sidecars(image_path):
        return []
    current_key = str(image_path.resolve()).casefold()
    candidates = [
        path
        for path in _iter_album_title_page_images(image_path, extensions)
        if str(path.resolve()).casefold() != current_key
    ]
    source_candidates = [path for path in candidates if _is_album_title_source_candidate(path)]
    return source_candidates or candidates


def _expand_album_title_dependencies(files: list[Path], extensions: set[str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    for image_path in files:
        for dependency in _resolve_album_title_dependencies(image_path, extensions):
            dep_key = str(dependency.resolve()).casefold()
            if dep_key in seen:
                continue
            seen.add(dep_key)
            expanded.append(dependency)
        image_key = str(image_path.resolve()).casefold()
        if image_key in seen:
            continue
        seen.add(image_key)
        expanded.append(image_path)
    return expanded


def _read_album_title_from_sidecar_iter(sidecar_iter) -> str:
    for sidecar_path in sidecar_iter:
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        album_title = str(state.get("album_title") or "").strip()
        if album_title:
            return album_title
    return ""


def _resolve_album_title_from_sidecars(image_path: Path) -> str:
    """Read album title from the P01 XMP sidecar. Returns '' if not yet processed."""
    return _read_album_title_from_sidecar_iter(_iter_album_p01_sidecars(image_path))


def _album_title_valid_in_sidecars(image_path: Path) -> bool:
    """Return True only if the P01 sidecar exists and its album_title matches its ocr_text.

    A mismatch means the title was set by the AI caption (not OCR) and needs to be
    reprocessed so the OCR-authoritative value is written.
    """
    for sidecar_path in _iter_album_p01_sidecars(image_path):
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            continue
        album_title = str(state.get("album_title") or "").strip()
        ocr_text = str(state.get("ocr_text") or "").strip()
        if album_title and ocr_text and album_title == ocr_text:
            return True
    return False


def _resolve_album_title_hint(image_path: Path) -> str:
    return _resolve_album_title_from_sidecars(image_path)


def _resolve_album_printed_title_from_sidecars(image_path: Path) -> str:
    return _read_album_title_from_sidecar_iter(_iter_album_cover_sidecars(image_path))


def _resolve_album_printed_title_hint(image_path: Path, printed_title_cache: dict[str, str]) -> str:
    key = _album_identity_key(image_path)
    cached = str(printed_title_cache.get(key) or "").strip()
    if cached:
        return cached
    title = _resolve_album_printed_title_from_sidecars(image_path)
    if title:
        printed_title_cache[key] = title
    return title


def _store_album_printed_title_hint(image_path: Path, printed_title_cache: dict[str, str], title: str) -> str:
    value = str(title or "").strip()
    if value:
        printed_title_cache[_album_identity_key(image_path)] = value
    return value


def _looks_like_album_title_page(image_path: Path) -> bool:
    return _title_page_match(image_path) is not None


def _require_album_title_for_title_page(
    *,
    image_path: Path,
    album_title: str,
    context: str,
) -> str:
    value = clean_text(str(album_title or ""))
    if value:
        return value
    if _is_album_title_source_candidate(image_path):
        raise RuntimeError(f"Missing album title for title page during {context}: {image_path}")
    return ""


def _resolve_title_page_album_title(
    *,
    image_path: Path,
    album_title: str,
    ocr_text: str,
) -> str:
    value = clean_text(str(album_title or ""))
    if value:
        return value
    if _is_album_title_source_candidate(image_path):
        return clean_text(str(ocr_text or ""))
    return ""
