from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib.ai_photo_crops import resolve_region_caption
from photoalbums.lib.caption_layout_migration import (
    _crop_region_index,
    _resolve_parent_page_sidecar,
    _set_description,
    _write_tree,
)
from photoalbums.lib.xmp_sidecar import _get_rdf_desc, read_ai_sidecar_state, read_region_list
from photoalbums.naming import DERIVED_VIEW_RE, is_photos_dir, parse_album_filename
from photoalbums.scripts._repair_args import build_repair_parser

_PLACEHOLDER_RE = re.compile(r"photo_\d+$", re.IGNORECASE)
_CROP_VIEW_RE = re.compile(r"_D\d{1,2}-00_V$", re.IGNORECASE)


def _iter_target_sidecars(photos_root: Path, album_filter: str, page_filter: str) -> list[Path]:
    targets: list[Path] = []
    for sidecar_path in sorted(photos_root.rglob("*.xmp")):
        if not is_photos_dir(sidecar_path.parent):
            continue
        if not DERIVED_VIEW_RE.search(sidecar_path.stem):
            continue
        if not _CROP_VIEW_RE.search(sidecar_path.stem):
            continue
        if album_filter and album_filter not in sidecar_path.parent.name.casefold():
            continue
        if page_filter:
            _, _, _, page = parse_album_filename(sidecar_path.name)
            if page != page_filter:
                continue
        targets.append(sidecar_path)
    return targets


def _expected_caption(crop_sidecar_path: Path) -> tuple[str, str] | None:
    crop_state = read_ai_sidecar_state(crop_sidecar_path)
    if not isinstance(crop_state, dict):
        raise ValueError(f"Could not parse crop sidecar: {crop_sidecar_path}")
    page_sidecar_path = _resolve_parent_page_sidecar(crop_sidecar_path)
    if page_sidecar_path is None:
        raise ValueError(f"Could not resolve parent page sidecar for crop: {crop_sidecar_path}")
    region_index = _crop_region_index(crop_sidecar_path)
    if region_index is None:
        raise ValueError(f"Could not resolve crop region index for: {crop_sidecar_path}")
    regions = read_region_list(page_sidecar_path, 1, 1)
    if region_index >= len(regions):
        return None
    region = regions[region_index]
    raw_region_caption = str(region.get("caption") or "").strip()
    if not _PLACEHOLDER_RE.fullmatch(raw_region_caption):
        return None
    page_state = read_ai_sidecar_state(page_sidecar_path)
    if not isinstance(page_state, dict):
        raise ValueError(f"Could not parse parent page sidecar: {page_sidecar_path}")
    page_description = str(page_state.get("description") or "").strip()
    expected = resolve_region_caption(
        raw_region_caption,
        str(region.get("caption_hint") or "").strip(),
        page_description,
    )
    current = str(crop_state.get("description") or "").strip()
    if current not in {"", raw_region_caption, page_description} and current != expected:
        return None
    return current, expected


def _repair_sidecar(crop_sidecar_path: Path, *, dry_run: bool) -> str:
    result = _expected_caption(crop_sidecar_path)
    if result is None:
        return "skip"
    current, expected = result
    if current == expected:
        return "skip"
    if dry_run:
        return "would_fix"
    tree = ET.parse(crop_sidecar_path)
    desc = _get_rdf_desc(tree)
    if desc is None:
        raise ValueError(f"rdf:Description missing from crop sidecar: {crop_sidecar_path}")
    changed = _set_description(desc, expected)
    if not changed:
        return "skip"
    ET.indent(tree, space="  ")
    _write_tree(crop_sidecar_path, tree)
    return "fixed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_repair_parser(
        description="Repair crop captions when parent MWG-RS Name only contains placeholder photo_# text.",
        default_photos_root=str(PHOTO_ALBUMS_DIR),
        include_page=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")

    album_filter = str(args.album or "").casefold()
    page_filter = f"{int(args.page):02d}" if str(args.page or "").strip().isdigit() else ""

    fixed = 0
    skipped = 0
    would_fix = 0
    failures = 0

    for crop_sidecar_path in _iter_target_sidecars(photos_root, album_filter, page_filter):
        try:
            outcome = _repair_sidecar(crop_sidecar_path, dry_run=not bool(args.run))
        except Exception as exc:
            failures += 1
            print(f"FAIL  {crop_sidecar_path}: {exc}")
            continue
        if outcome == "fixed":
            fixed += 1
            print(f"FIX   {crop_sidecar_path}")
        elif outcome == "would_fix":
            would_fix += 1
            print(f"PLAN  {crop_sidecar_path}")
        else:
            skipped += 1

    if args.run:
        print(f"done fixed={fixed} skipped={skipped} failures={failures}")
    else:
        print(f"dry-run would_fix={would_fix} skipped={skipped} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
