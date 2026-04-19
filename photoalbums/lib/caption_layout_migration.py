from __future__ import annotations

import os
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

from .xmp_sidecar import (
    DC_NS,
    DESCRIPTION_ROLE_CROP,
    DESCRIPTION_ROLE_PLAIN,
    DESCRIPTION_ROLE_PAGE,
    IMAGO_NS,
    MWGRS_NS,
    _RDF_ALT,
    _RDF_BAG,
    _RDF_LI,
    _description_role_for_sidecar_path,
    _get_alt_text,
    _get_description_value,
    _get_rdf_desc,
    _page_description_summary,
    _set_alt_text,
    _set_simple_text,
)
from .xmpmm_provenance import read_derived_from
from ..naming import DERIVED_NAME_RE, is_photos_dir, pages_dir_for_album_dir, parse_album_filename

_DESCRIPTION_TAG = f"{{{DC_NS}}}description"
_OCR_TEXT_TAG = f"{{{IMAGO_NS}}}OCRText"
_PARENT_OCR_TEXT_TAG = f"{{{IMAGO_NS}}}ParentOCRText"
_SCENE_TEXT_TAG = f"{{{IMAGO_NS}}}SceneText"
_LEGACY_DESCRIPTION_LANGS = {"x-caption", "x-author", "x-scene"}
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _first_nonempty(*values: str) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _load_desc(xmp_path: str | Path) -> ET.Element | None:
    path = Path(xmp_path)
    if not path.is_file():
        return None
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    return _get_rdf_desc(tree)


def _description_lang_values(desc: ET.Element) -> dict[str, str]:
    field = desc.find(_DESCRIPTION_TAG)
    if field is None:
        return {}
    alt = field.find(_RDF_ALT)
    if alt is None:
        return {}
    values: dict[str, str] = {}
    for item in alt.findall(_RDF_LI):
        lang = str(item.get(_XML_LANG) or "").strip()
        text = str(item.text or "").strip()
        if lang and text:
            values[lang] = text
    return values


def _set_description(desc: ET.Element, value: str) -> bool:
    desired = str(value or "").strip()
    current = _description_lang_values(desc)
    desired_map = {"x-default": desired} if desired else {}
    if current == desired_map:
        return False
    _set_alt_text(desc, _DESCRIPTION_TAG, desired)
    return True


def _set_simple_field(desc: ET.Element, tag: str, value: str) -> bool:
    desired = str(value or "").strip()
    current = str(desc.findtext(tag, default="") or "").strip()
    if current == desired:
        return False
    _set_simple_text(desc, tag, desired)
    return True


def _write_tree(path: Path, tree: ET.ElementTree) -> None:
    was_readonly = False
    try:
        file_attributes = int(path.stat().st_file_attributes)
    except (AttributeError, OSError, ValueError):
        file_attributes = 0
    if file_attributes & getattr(stat, "FILE_ATTRIBUTE_READONLY", 0):
        was_readonly = True
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    try:
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
    finally:
        if was_readonly:
            os.chmod(path, stat.S_IREAD)


def _crop_region_index(sidecar_path: Path) -> int | None:
    stem = sidecar_path.stem
    if stem.endswith("_V"):
        stem = stem[:-2]
    match = DERIVED_NAME_RE.fullmatch(stem)
    if match is None:
        return None
    return int(match.group("derived")) - 1


def _resolve_parent_page_sidecar(crop_sidecar_path: Path) -> Path | None:
    candidates: list[Path] = []
    seen: set[Path] = set()
    derived_from = read_derived_from(crop_sidecar_path)
    source_path = str(derived_from.get("source_path") or "").strip()
    if source_path:
        source = Path(source_path)
        if source.is_absolute():
            candidates.append(source.with_suffix(".xmp"))
        else:
            candidates.append((crop_sidecar_path.parent / source).with_suffix(".xmp"))
            if source.name:
                candidates.append((crop_sidecar_path.parent / source.name).with_suffix(".xmp"))
    if is_photos_dir(crop_sidecar_path.parent):
        collection, year, book, page = parse_album_filename(crop_sidecar_path.stem)
        page_token = f"{int(page):02d}" if str(page).isdigit() else str(page)
        candidates.append(pages_dir_for_album_dir(crop_sidecar_path.parent) / f"{collection}_{year}_B{book}_P{page_token}_V.xmp")
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return candidate
    return None


def _read_parent_region_caption(page_sidecar_path: Path | None, region_index: int | None) -> str:
    if page_sidecar_path is None or region_index is None:
        return ""
    desc = _load_desc(page_sidecar_path)
    if desc is None:
        return ""
    region_info = desc.find(f"{{{MWGRS_NS}}}RegionInfo")
    if region_info is None:
        return ""
    region_list = region_info.find(f"{{{MWGRS_NS}}}RegionList")
    if region_list is None:
        return ""
    bag = region_list.find(_RDF_BAG)
    if bag is None:
        return ""
    items = bag.findall(_RDF_LI)
    if region_index < 0 or region_index >= len(items):
        return ""
    return str(items[region_index].get(f"{{{MWGRS_NS}}}Name") or "").strip()


def _read_page_summary(page_sidecar_path: Path | None) -> str:
    if page_sidecar_path is None:
        return ""
    desc = _load_desc(page_sidecar_path)
    if desc is None:
        return ""
    ocr_text = str(desc.findtext(_OCR_TEXT_TAG, default="") or "").strip()
    scene_text = str(desc.findtext(_SCENE_TEXT_TAG, default="") or "").strip()
    return _page_description_summary(ocr_text, scene_text) or _get_description_value(desc)


def _read_page_ocr_text(page_sidecar_path: Path | None) -> str:
    if page_sidecar_path is None:
        return ""
    desc = _load_desc(page_sidecar_path)
    if desc is None:
        return ""
    return str(desc.findtext(_OCR_TEXT_TAG, default="") or "").strip()


def _logical_crop_description(
    *,
    current_default: str,
    inherited_parent_ocr: str,
    parent_page_description: str,
) -> str:
    logical = str(current_default or "").strip()
    if not logical:
        return ""
    if logical == str(inherited_parent_ocr or "").strip():
        return ""
    if logical == str(parent_page_description or "").strip():
        return ""
    return logical


def sidecar_needs_caption_layout_migration(sidecar_path: str | Path) -> bool:
    path = Path(sidecar_path)
    desc = _load_desc(path)
    if desc is None:
        return False
    langs = _description_lang_values(desc)
    if _LEGACY_DESCRIPTION_LANGS.intersection(langs):
        return True
    if _description_role_for_sidecar_path(path) == DESCRIPTION_ROLE_CROP:
        if str(desc.findtext(_OCR_TEXT_TAG, default="") or "").strip() and not str(
            desc.findtext(_PARENT_OCR_TEXT_TAG, default="") or ""
        ).strip():
            return True
    return False


def migrate_sidecar_caption_layout(sidecar_path: str | Path) -> bool:
    path = Path(sidecar_path)
    if not path.is_file():
        return False
    role = _description_role_for_sidecar_path(path)
    if role not in {DESCRIPTION_ROLE_PAGE, DESCRIPTION_ROLE_CROP, DESCRIPTION_ROLE_PLAIN}:
        return False
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return False
    desc = _get_rdf_desc(tree)
    if desc is None:
        return False

    changed = False
    if role == DESCRIPTION_ROLE_PAGE:
        ocr_text = str(desc.findtext(_OCR_TEXT_TAG, default="") or "").strip()
        scene_text = str(desc.findtext(_SCENE_TEXT_TAG, default="") or "").strip()
        page_summary = _page_description_summary(ocr_text, scene_text) or _get_description_value(desc)
        changed |= _set_description(desc, page_summary)
    elif role == DESCRIPTION_ROLE_CROP:
        page_sidecar_path = _resolve_parent_page_sidecar(path)
        page_description = _read_page_summary(page_sidecar_path)
        page_ocr_text = _read_page_ocr_text(page_sidecar_path)
        region_caption = _read_parent_region_caption(page_sidecar_path, _crop_region_index(path))
        legacy_caption = _get_alt_text(desc, _DESCRIPTION_TAG, prefer_lang="x-caption", fallback_to_any=False)
        current_default = _get_alt_text(desc, _DESCRIPTION_TAG, prefer_lang="x-default", fallback_to_any=False)
        existing_parent_ocr = str(desc.findtext(_PARENT_OCR_TEXT_TAG, default="") or "").strip()
        legacy_parent_ocr = str(desc.findtext(_OCR_TEXT_TAG, default="") or "").strip()
        logical_description = _logical_crop_description(
            current_default=current_default,
            inherited_parent_ocr=existing_parent_ocr or legacy_parent_ocr,
            parent_page_description=page_description,
        )
        caption = _first_nonempty(region_caption, legacy_caption, logical_description, page_description)
        parent_ocr_text = _first_nonempty(existing_parent_ocr, legacy_parent_ocr, page_ocr_text)
        changed |= _set_description(desc, caption)
        changed |= _set_simple_field(desc, _PARENT_OCR_TEXT_TAG, parent_ocr_text)
        if legacy_parent_ocr and (not existing_parent_ocr or legacy_parent_ocr == existing_parent_ocr):
            changed |= _set_simple_field(desc, _OCR_TEXT_TAG, "")
    else:
        plain_description = _first_nonempty(
            _get_alt_text(desc, _DESCRIPTION_TAG, prefer_lang="x-caption", fallback_to_any=False),
            _get_alt_text(desc, _DESCRIPTION_TAG, prefer_lang="x-default", fallback_to_any=False),
        )
        changed |= _set_description(desc, plain_description)

    if not changed:
        return False
    ET.indent(tree, space="  ")
    _write_tree(path, tree)
    return True


def find_sidecars_with_legacy_caption_layout(photos_root: str | Path) -> list[Path]:
    root = Path(photos_root)
    matches: list[Path] = []
    for sidecar_path in sorted(root.rglob("*.xmp")):
        if sidecar_needs_caption_layout_migration(sidecar_path):
            matches.append(sidecar_path)
    return matches


def migrate_album_caption_layout(photos_root: str | Path) -> dict[str, int]:
    root = Path(photos_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo albums root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

    files_scanned = 0
    files_changed = 0
    for sidecar_path in sorted(root.rglob("*.xmp")):
        files_scanned += 1
        if migrate_sidecar_caption_layout(sidecar_path):
            files_changed += 1
    return {
        "files_scanned": files_scanned,
        "files_changed": files_changed,
    }
