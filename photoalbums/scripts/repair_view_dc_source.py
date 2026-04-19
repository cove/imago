from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib.ai_index import _build_dc_source, _page_scan_filenames
from photoalbums.lib.xmp_sidecar import DC_NS, _get_rdf_desc, _set_simple_text, read_ai_sidecar_state
from photoalbums.scripts._repair_args import build_repair_parser


def _iter_target_sidecars(photos_root: Path, album_filter: str) -> list[Path]:
    targets: list[Path] = []
    filter_text = str(album_filter or "").casefold()
    for sidecar_path in sorted(photos_root.rglob("*.xmp")):
        if sidecar_path.suffix.lower() != ".xmp":
            continue
        if not sidecar_path.parent.name.endswith("_Pages"):
            continue
        if not (sidecar_path.name.endswith("_V.xmp") or sidecar_path.name.endswith("_VC.xmp")):
            continue
        if filter_text and filter_text not in sidecar_path.parent.name.casefold():
            continue
        targets.append(sidecar_path)
    return targets


def _expected_dc_source(sidecar_path: Path) -> tuple[str, str]:
    state = read_ai_sidecar_state(sidecar_path)
    if not isinstance(state, dict):
        raise ValueError(f"Could not parse XMP sidecar: {sidecar_path}")
    image_path = sidecar_path.with_suffix(".jpg")
    if not image_path.is_file():
        raise ValueError(f"Companion JPG was not found: {image_path}")
    scan_filenames = _page_scan_filenames(image_path)
    if not scan_filenames:
        raise ValueError(f"No archive scan filenames were found for: {image_path}")
    current_source = str(state.get("source_text") or "").strip()
    album_title = str(state.get("album_title") or "").strip()
    if not album_title and " Page " in current_source:
        album_title = current_source.split(" Page ", 1)[0].strip()
    return _build_dc_source(album_title, image_path, scan_filenames), current_source


def _repair_sidecar(sidecar_path: Path, *, dry_run: bool) -> str:
    expected_source, current_source = _expected_dc_source(sidecar_path)
    if current_source == expected_source:
        return "skip"
    if dry_run:
        return "would_fix"
    tree = ET.parse(sidecar_path)
    desc = _get_rdf_desc(tree)
    if desc is None:
        raise ValueError(f"rdf:Description missing from XMP sidecar: {sidecar_path}")
    _set_simple_text(desc, f"{{{DC_NS}}}source", expected_source)
    ET.indent(tree, space="  ")
    tree.write(sidecar_path, encoding="utf-8", xml_declaration=True)
    return "fixed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_repair_parser(
        description="Repair stale dc:source values in view XMP sidecars without rerunning AI.",
        default_photos_root=str(PHOTO_ALBUMS_DIR),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")

    fixed = 0
    skipped = 0
    would_fix = 0
    failures = 0

    for sidecar_path in _iter_target_sidecars(photos_root, str(args.album or "")):
        try:
            outcome = _repair_sidecar(sidecar_path, dry_run=not bool(args.run))
        except Exception as exc:
            failures += 1
            print(f"FAIL  {sidecar_path}: {exc}")
            continue
        if outcome == "fixed":
            fixed += 1
            print(f"FIX   {sidecar_path}")
        elif outcome == "would_fix":
            would_fix += 1
            print(f"PLAN  {sidecar_path}")
        else:
            skipped += 1

    if args.run:
        print(f"done fixed={fixed} skipped={skipped} failures={failures}")
    else:
        print(f"dry-run would_fix={would_fix} skipped={skipped} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
