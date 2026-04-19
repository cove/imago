from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .ai_index_scan import _build_dc_source, _page_scan_filenames
from .ai_render_settings import find_archive_dir_for_image
from .ai_sidecar_state import _effective_sidecar_album_title
from .caption_layout_migration import _write_tree
from .xmp_sidecar import DC_NS, IMAGO_NS, _get_rdf_desc, _set_simple_text, read_ai_sidecar_state
from ..naming import DERIVED_VIEW_RE, is_photos_dir, parse_album_filename


def _iter_target_sidecars(photos_root: str | Path, album_id: str = "", page: str | None = None) -> list[Path]:
    root = Path(photos_root)
    album_filter = str(album_id or "").casefold()
    page_filter = f"{int(page):02d}" if str(page or "").strip().isdigit() else ""
    targets: list[Path] = []
    for sidecar_path in sorted(root.rglob("*.xmp")):
        if not is_photos_dir(sidecar_path.parent):
            continue
        if not DERIVED_VIEW_RE.search(sidecar_path.stem):
            continue
        if album_filter and album_filter not in sidecar_path.parent.name.casefold():
            continue
        if page_filter:
            _, _, _, page_str = parse_album_filename(sidecar_path.name)
            if page_str != page_filter:
                continue
        targets.append(sidecar_path)
    return targets


def _resolve_expected_crop_metadata(sidecar_path: str | Path) -> tuple[str, str, str, str]:
    path = Path(sidecar_path)
    state = read_ai_sidecar_state(path)
    if not isinstance(state, dict):
        raise ValueError(f"Could not parse XMP sidecar: {path}")
    image_path = path.with_suffix(".jpg")
    if not image_path.is_file():
        raise FileNotFoundError(f"Companion JPG was not found: {image_path}")
    scan_filenames = _page_scan_filenames(image_path)
    if not scan_filenames:
        raise ValueError(f"No archive scan filenames were found for: {image_path}")
    expected_album_title = _effective_sidecar_album_title(image_path, state)
    if not expected_album_title:
        archive_dir = find_archive_dir_for_image(image_path)
        if archive_dir is not None and archive_dir.is_dir():
            archive_state = read_ai_sidecar_state((archive_dir / scan_filenames[0]).with_suffix(".xmp"))
            if isinstance(archive_state, dict):
                expected_album_title = _effective_sidecar_album_title(image_path, archive_state)
    expected_source_text = _build_dc_source(expected_album_title, image_path, scan_filenames)
    current_album_title = str(state.get("album_title") or "").strip()
    current_source_text = str(state.get("source_text") or "").strip()
    return expected_album_title, expected_source_text, current_album_title, current_source_text


def crop_sidecar_needs_source_repair(sidecar_path: str | Path) -> bool:
    expected_album_title, expected_source_text, current_album_title, current_source_text = _resolve_expected_crop_metadata(
        sidecar_path
    )
    return current_album_title != expected_album_title or current_source_text != expected_source_text


def repair_crop_sidecar_source(sidecar_path: str | Path) -> bool:
    path = Path(sidecar_path)
    expected_album_title, expected_source_text, current_album_title, current_source_text = _resolve_expected_crop_metadata(
        path
    )
    if current_album_title == expected_album_title and current_source_text == expected_source_text:
        return False
    tree = ET.parse(path)
    desc = _get_rdf_desc(tree)
    if desc is None:
        raise ValueError(f"rdf:Description missing from XMP sidecar: {path}")
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", expected_album_title)
    _set_simple_text(desc, f"{{{DC_NS}}}source", expected_source_text)
    ET.indent(tree, space="  ")
    _write_tree(path, tree)
    return True


def find_crop_sidecars_needing_source_repair(
    photos_root: str | Path,
    *,
    album_id: str = "",
    page: str | None = None,
) -> list[Path]:
    matches: list[Path] = []
    for sidecar_path in _iter_target_sidecars(photos_root, album_id=album_id, page=page):
        if crop_sidecar_needs_source_repair(sidecar_path):
            matches.append(sidecar_path)
    return matches


def repair_album_crop_sources(
    photos_root: str | Path,
    *,
    album_id: str = "",
    page: str | None = None,
) -> dict[str, int]:
    root = Path(photos_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo albums root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

    files_scanned = 0
    files_changed = 0
    for sidecar_path in _iter_target_sidecars(root, album_id=album_id, page=page):
        files_scanned += 1
        if repair_crop_sidecar_source(sidecar_path):
            files_changed += 1
    return {
        "files_scanned": files_scanned,
        "files_changed": files_changed,
    }
