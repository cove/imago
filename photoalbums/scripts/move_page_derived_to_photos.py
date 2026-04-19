from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.naming import DERIVED_VIEW_RE, is_pages_dir, photos_dir_for_album_dir

_DERIVED_MEDIA_SUFFIXES = {".jpg", ".xmp", ".mp4", ".mov", ".avi", ".m4v", ".pdf"}


@dataclass(frozen=True)
class MoveOperation:
    source: Path
    target: Path


def _matches_filters(path: Path, *, album_filter: str, page_filter: str) -> bool:
    if album_filter and album_filter not in path.parent.name.casefold():
        return False
    if page_filter and f"_P{page_filter}_" not in path.name.casefold():
        return False
    return True


def _iter_move_operations(photos_root: Path, *, album_filter: str = "", page: str = "") -> list[MoveOperation]:
    page_filter = f"{int(page):02d}" if str(page or "").strip().isdigit() else ""
    filter_text = str(album_filter or "").casefold()
    operations: list[MoveOperation] = []

    for pages_dir in sorted(path for path in photos_root.iterdir() if path.is_dir() and is_pages_dir(path)):
        photos_dir = photos_dir_for_album_dir(pages_dir)
        for path in sorted(pages_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _DERIVED_MEDIA_SUFFIXES:
                continue
            if path.suffix.lower() in {".jpg", ".xmp"}:
                if DERIVED_VIEW_RE.search(path.stem) is None:
                    continue
            elif "_D" not in path.stem:
                continue
            if not _matches_filters(path, album_filter=filter_text, page_filter=page_filter.casefold()):
                continue
            operations.append(MoveOperation(source=path, target=photos_dir / path.name))

    return operations


def _validate_move_operations(operations: list[MoveOperation]) -> None:
    seen_targets: set[Path] = set()
    for operation in operations:
        if operation.target in seen_targets:
            raise FileExistsError(f"Duplicate move target planned: {operation.target}")
        seen_targets.add(operation.target)
        if operation.target.exists():
            raise FileExistsError(f"Target already exists: {operation.target}")


def _execute_move_operations(operations: list[MoveOperation], *, dry_run: bool) -> tuple[int, int]:
    moved = 0
    skipped = 0
    for operation in operations:
        if dry_run:
            print(f"PLAN  {operation.source} -> {operation.target}")
            continue
        operation.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(operation.source), str(operation.target))
        moved += 1
        print(f"MOVE  {operation.source} -> {operation.target}")
    if dry_run:
        skipped = len(operations)
    return moved, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move misrouted derived JPG/XMP/media files from *_Pages to sibling *_Photos directories."
    )
    parser.add_argument(
        "--photos-root",
        default=str(PHOTO_ALBUMS_DIR),
        help="Photo Albums root directory.",
    )
    parser.add_argument(
        "--album",
        default="",
        help="Optional substring filter against the parent _Pages directory name.",
    )
    parser.add_argument(
        "--page",
        default="",
        help="Optional page number filter; omit for all pages.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the move in place. Omit for a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")

    operations = _iter_move_operations(
        photos_root,
        album_filter=str(args.album or ""),
        page=str(args.page or ""),
    )
    _validate_move_operations(operations)
    moved, planned = _execute_move_operations(operations, dry_run=not bool(args.run))

    if args.run:
        print(f"done moved={moved} failures=0")
    else:
        print(f"dry-run planned={planned} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
