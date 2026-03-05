#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".m2ts",
    ".mts",
}


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "album"


def detect_media_type(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def iter_media_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")
    for path in iterator:
        try:
            is_file = path.is_file()
        except OSError:
            continue
        if not is_file:
            continue
        if detect_media_type(path) is None:
            continue
        yield path


def collect_view_dirs(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for view_dir in sorted(root.rglob("*_View")):
            if not view_dir.is_dir():
                continue
            key = str(view_dir.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(view_dir.resolve())
    return out


def collect_named_dirs(
    roots: Iterable[Path], names: Iterable[str], max_depth: int = 5, include_root: bool = False
) -> List[Path]:
    wanted = {n.lower() for n in names}
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        candidates = [root] if include_root else []
        try:
            candidates.extend(p for p in root.rglob("*") if p.is_dir())
        except OSError:
            pass
        root_parts = len(root.parts)
        for path in candidates:
            try:
                if len(path.parts) - root_parts > max_depth:
                    continue
            except Exception:
                continue
            if path.name.lower() not in wanted:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(path.resolve())
    return sorted(out, key=lambda p: str(p).lower())


def make_item(item_id: str, media_type: str, path: Path, root: Path) -> Dict[str, str]:
    title = path.stem.replace("_", " ").strip() or path.name
    try:
        rel = path.relative_to(root)
        caption = str(rel)
    except ValueError:
        caption = path.name
    return {
        "id": item_id,
        "type": media_type,
        "source": "local",
        "title": title,
        "caption": caption,
        "path": str(path.resolve()),
    }


def build_album(root: Path, recursive: bool, max_items: int, album_index: int) -> Tuple[Dict, int]:
    files = sorted(iter_media_files(root, recursive), key=lambda p: str(p).lower())
    if max_items > 0:
        files = files[:max_items]

    album_id = f"{slugify(root.name)}-{album_index:02d}"
    album = {
        "id": album_id,
        "title": root.name or f"Album {album_index}",
        "items": [],
    }

    item_count = 0
    for i, media_path in enumerate(files, start=1):
        media_type = detect_media_type(media_path)
        if media_type is None:
            continue
        item_id = f"{album_id}-{i:05d}"
        album["items"].append(make_item(item_id, media_type, media_path, root))
        item_count += 1
    return album, item_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan local media folders and generate viewer/gallery.json.")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root folder to scan (repeat for multiple albums).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "gallery.json"),
        help="Output gallery JSON path (default: viewer/gallery.json).",
    )
    parser.add_argument(
        "--max-items-per-album",
        type=int,
        default=0,
        help="Limit items per album; 0 means no limit.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan top-level files in each root folder.",
    )
    parser.add_argument(
        "--imago-layout",
        action="store_true",
        help="Auto-scan project conventions: videos from VHS Clips/ + Videos/, photos from *_View folders.",
    )
    parser.add_argument(
        "--videos-root",
        action="append",
        default=[],
        help="Additional root to search for video folders (used by --imago-layout).",
    )
    parser.add_argument(
        "--photos-root",
        action="append",
        default=[],
        help="Additional root to search for *_View folders (used by --imago-layout).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recursive = not args.no_recursive
    max_items = max(args.max_items_per_album, 0)
    if not args.imago_layout and not args.root:
        raise SystemExit("Provide at least one --root, or use --imago-layout.")

    albums: List[Dict] = []
    total_items = 0

    if args.imago_layout:
        home = Path.home()
        inferred_video_candidates = [home / "Videos"]
        inferred_photo_candidates = [
            home / "OneDrive" / "Cordell, Leslie & Audrey" / "Photo Albums",
            home / "Library" / "CloudStorage" / "OneDrive-Personal" / "Cordell, Leslie & Audrey" / "Photo Albums",
        ]

        explicit_video_roots = [Path(p).expanduser().resolve() for p in args.videos_root]
        explicit_photo_roots = [Path(p).expanduser().resolve() for p in args.photos_root]

        video_search_roots: List[Path] = []
        seen_video_search = set()
        for cand in [*explicit_video_roots, *inferred_video_candidates]:
            if not cand.exists() or not cand.is_dir():
                continue
            key = str(cand.resolve()).lower()
            if key in seen_video_search:
                continue
            seen_video_search.add(key)
            video_search_roots.append(cand.resolve())
        video_roots = collect_named_dirs(
            video_search_roots, names=["VHS Clips", "Videos"], max_depth=5, include_root=False
        )

        photo_search_roots: List[Path] = []
        seen_photo = set()
        for cand in [*explicit_photo_roots, *inferred_photo_candidates]:
            if not cand.exists() or not cand.is_dir():
                continue
            key = str(cand.resolve()).lower()
            if key in seen_photo:
                continue
            seen_photo.add(key)
            photo_search_roots.append(cand.resolve())

        album_idx = 1
        for root in video_roots:
            album, count = build_album(root, recursive=True, max_items=max_items, album_index=album_idx)
            # Keep only video files for explicit video roots.
            album["items"] = [it for it in album["items"] if it.get("type") == "video"]
            count = len(album["items"])
            albums.append(album)
            total_items += count
            print(f"[scan_media] video root {root} -> {count} item(s)")
            album_idx += 1

        view_dirs = collect_view_dirs(photo_search_roots)
        for view_dir in view_dirs:
            album, count = build_album(view_dir, recursive=True, max_items=max_items, album_index=album_idx)
            album["items"] = [it for it in album["items"] if it.get("type") == "image"]
            count = len(album["items"])
            albums.append(album)
            total_items += count
            print(f"[scan_media] photo view {view_dir} -> {count} item(s)")
            album_idx += 1

        if not video_roots:
            print("[scan_media] note: no video roots found (expected VHS Clips/ and/or Videos).")
        if not photo_search_roots:
            print("[scan_media] note: no photo roots found for *_View discovery.")
        if photo_search_roots and not view_dirs:
            print("[scan_media] note: no *_View folders found under photo roots.")
    else:
        roots: List[Path] = [Path(raw).expanduser().resolve() for raw in args.root]
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"Root does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Root is not a directory: {root}")

        for idx, root in enumerate(roots, start=1):
            album, count = build_album(root, recursive=recursive, max_items=max_items, album_index=idx)
            albums.append(album)
            total_items += count
            print(f"[scan_media] {root} -> {count} item(s)")

    payload = {"albums": albums}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[scan_media] wrote {len(albums)} album(s), {total_items} total item(s) to {output}")
    if total_items == 0:
        print("[scan_media] warning: no media files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
