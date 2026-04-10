from __future__ import annotations

import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _call_main(func) -> int:
    try:
        result = func()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        print(code)
        return 1
    if isinstance(result, int):
        return int(result)
    return 0


def run_ai_index(argv: list[str]) -> int:
    from .lib import ai_index

    return int(ai_index.run(argv) or 0)


def run_apply_metadata() -> int:
    import apply_metadata

    return _call_main(apply_metadata.main)


def run_create_metadata_tsv() -> int:
    import create_metadata_tsv

    return _call_main(create_metadata_tsv.main)


def run_metadata_map(*, paths: list[str], port: int) -> int:
    from . import map_server

    try:
        map_server.run_server(paths, port=port)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def run_compress_tiff() -> int:
    import compress_tiff

    return _call_main(compress_tiff.main)


def run_render() -> int:
    import stitch_oversized_pages

    return _call_main(stitch_oversized_pages.main)


def run_ctm(argv: list[str]) -> int:
    from .lib import ai_ctm_restoration
    from .lib.ai_render_settings import find_archive_dir_for_image
    from .stitch_oversized_pages import list_archive_dirs, list_page_scans, _require_primary_scan

    if not argv:
        print("Error: missing CTM command")
        return 2
    command = str(argv[0]).strip().lower()
    if command not in {"generate", "review"}:
        print(f"Error: unknown CTM command: {command}")
        return 2
    args = list(argv[1:])
    force = False
    album_id = ""
    page = ""
    photos_root = "."
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--force":
            force = True
            index += 1
            continue
        if token == "--album-id" and index + 1 < len(args):
            album_id = args[index + 1]
            index += 2
            continue
        if token == "--page" and index + 1 < len(args):
            page = args[index + 1]
            index += 2
            continue
        if token == "--photos-root" and index + 1 < len(args):
            photos_root = args[index + 1]
            index += 2
            continue
        print(f"Error: unknown argument: {token}")
        return 2

    archives = [Path(path) for path in list_archive_dirs(photos_root)]
    selected = [path for path in archives if not album_id or path.name == f"{album_id}_Archive"]
    if not selected:
        print(f"Error: no archive matched album_id={album_id!r}")
        return 1

    matched: list[Path] = []
    for archive in selected:
        for group in list_page_scans(archive):
            primary = Path(_require_primary_scan(group))
            if page and f"P{int(page):02d}" not in primary.name:
                continue
            matched.append(primary)
    if not matched:
        print("Error: no matching archive scan pages found")
        return 1

    if command == "generate":
        for scan in matched:
            archive_sidecar, result = ai_ctm_restoration.generate_and_store_ctm(scan, force=force)
            print(json.dumps({"image": scan.name, "archive_xmp": str(archive_sidecar), **result.to_dict()}, ensure_ascii=False))
        return 0

    for scan in matched:
        state = ai_ctm_restoration.read_ctm_from_archive_xmp(scan.with_suffix(".xmp"))
        print(json.dumps({"image": scan.name, "archive_xmp": str(scan.with_suffix('.xmp')), "ctm": state}, ensure_ascii=False))
    return 0


def run_stitch_validate() -> int:
    import stitch_oversized_pages_validate

    return _call_main(stitch_oversized_pages_validate.main)


def run_watch_incoming() -> int:
    import incoming_scans_watcher

    return _call_main(incoming_scans_watcher.main)


def run_checksum_tree(*, base_dir: str, verify: bool) -> int:
    import sha3_tree_hashes

    argv = [str(base_dir)]
    if verify:
        argv.append("--verify")
    return int(sha3_tree_hashes.run(argv) or 0)


def run_detect_view_regions(*, album_id: str, photos_root: str, page: str | None, force: bool) -> int:
    from pathlib import Path
    from .lib.ai_view_regions import detect_regions, RegionWithCaption
    from .lib.xmp_sidecar import write_region_list
    from .lib.ai_view_regions import _image_dimensions, associate_captions

    root = Path(photos_root)
    album_id_lower = album_id.casefold()

    view_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.endswith("_View") and album_id_lower in d.name.casefold())
    if not view_dirs:
        print(f"No _View directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    errors = 0
    for view_dir in view_dirs:
        if page is not None:
            page_padded = str(page).zfill(2)
            candidates = sorted(view_dir.glob(f"*_P{page_padded}_V.jpg"))
        else:
            candidates = sorted(view_dir.glob("*_V.jpg"))

        for view_path in candidates:
            xmp_path = view_path.with_suffix(".xmp")
            print(f"Processing {view_path.name}...")
            try:
                img_w, img_h = _image_dimensions(view_path)
                regions = detect_regions(view_path, force=force)
                if not regions:
                    print(f"  No regions detected or model unavailable; skipping XMP write.")
                    continue
                captions: list[dict] = []  # Future: extract from existing XMP description
                regions_with_captions = associate_captions(regions, captions, img_w)
                write_region_list(xmp_path, regions_with_captions, img_w, img_h)
                print(f"  Wrote {len(regions)} region(s) to {xmp_path.name}")
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
