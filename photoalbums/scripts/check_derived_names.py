"""
Utilities for managing D##-## derived image files in an _Archive folder.

Subcommands
-----------
check <archive_dir>
    Compare D##-##.png files against their _Photos _V.jpg counterparts using ORB
    feature matching. Files that look different get flagged for renaming.

place <archive_dir> <colorized_dir>
    Match colorized PNGs (no album naming) against existing Archive D##-## images
    using ORB. Best match determines the page and D-number; the colorized file is
    copied in as the next unused iteration (e.g. D04-01 → D04-02).

Usage
-----
    uv run photoalbums/scripts/check_derived_names.py check <archive_dir> [options]
    uv run photoalbums/scripts/check_derived_names.py place <archive_dir> <colorized_dir> [options]

Common options
    --threshold F   ORB good-match ratio for "same image" (default: 0.10)
    --execute       Apply changes (default: dry-run)
    --no-preview    Skip viu image previews
    --width N       viu display width in columns (default: 60)

check-only options
    --today         Only check files modified today
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from photoalbums.naming import (
    DERIVED_NAME_RE,
    album_sibling_dir,
    is_archive_dir,
)
from photoalbums import naming as _naming

PHOTOS_SUFFIX = _naming.ALBUM_DIR_SUFFIX_PHOTOS
ARCHIVE_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_archive_hash_index(archive_dir: Path) -> dict[str, Path]:
    """Map sha256 → archive file path for all image files in the archive."""
    index: dict[str, Path] = {}
    for f in archive_dir.iterdir():
        if f.suffix.lower() in ARCHIVE_IMAGE_EXTS:
            index[_file_hash(f)] = f
    return index


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_gray_thumb(path: Path, size: int = 800) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = size / max(h, w)
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _orb_descriptors(img: np.ndarray) -> tuple[list, np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def _orb_similarity(
    kp_a: list, des_a: np.ndarray | None,
    kp_b: list, des_b: np.ndarray | None,
) -> float:
    """Return ratio of good ORB matches to total keypoints (0.0–1.0)."""
    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    good = [m for m in matches if m.distance < 50]
    return len(good) / max(len(kp_a), len(kp_b))


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _next_d_number(archive_dir: Path, collection: str, year: str, book: str, page: str) -> int:
    """Next D number not used in either Archive or Photos for this page."""
    photos_dir = album_sibling_dir(archive_dir, PHOTOS_SUFFIX)
    used: set[int] = set()
    for search_dir in (archive_dir, photos_dir):
        if not search_dir.is_dir():
            continue
        for f in search_dir.iterdir():
            m = DERIVED_NAME_RE.search(f.name)
            if m and _page_key(m) == (collection, year, book, page):
                used.add(int(m.group("derived")))
    return max(used, default=0) + 1


def _next_iter_number(
    archive_dir: Path, collection: str, year: str, book: str, page: str, d_num: int
) -> int:
    """Next iteration number for a given D## on this page, scanning Archive only."""
    used: set[int] = set()
    for f in archive_dir.iterdir():
        m = DERIVED_NAME_RE.search(f.name)
        if (
            m
            and _page_key(m) == (collection, year, book, page)
            and int(m.group("derived")) == d_num
        ):
            used.add(int(m.group("iter")))
    return max(used, default=0) + 1


def _page_key(m: re.Match[str]) -> tuple[str, str, str, str]:
    return (m.group("collection"), m.group("year"), m.group("book"), m.group("page"))


def _build_stem(m: re.Match[str], d_num: int, iter_num: int) -> str:
    collection = m.group("collection")
    year = m.group("year")
    book = m.group("book")
    page = m.group("page")
    book_fmt = f"B{int(book):02d}" if book.isdigit() else f"B{book}"
    return f"{collection}_{year}_{book_fmt}_P{int(page):02d}_D{d_num:02d}-{iter_num:02d}"


# ---------------------------------------------------------------------------
# viu preview
# ---------------------------------------------------------------------------

def _viu(path: Path, label: str, width: int) -> None:
    print(f"  {label}: {path.name}")
    viu = shutil.which("viu")
    if viu is None:
        print("  (viu not found — install with: cargo install viu)")
        return
    subprocess.run([viu, "-w", str(width), str(path)], check=False)


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------

def _is_today(path: Path) -> bool:
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    return mtime == date.today()


def _rename_archive_file(
    archive_file: Path, m: re.Match[str], new_d: int, execute: bool
) -> Path:
    # New D## = completely different photo, so iteration resets to 01
    new_stem = _build_stem(m, new_d, 1)
    new_path = archive_file.parent / (new_stem + archive_file.suffix)
    if execute:
        archive_file.rename(new_path)
        xmp = archive_file.with_suffix(".xmp")
        if xmp.exists():
            xmp.rename(archive_file.parent / (new_stem + ".xmp"))
    return new_path


def cmd_check(args: argparse.Namespace) -> None:
    archive_dir: Path = args.archive_dir.expanduser().resolve()
    if not is_archive_dir(archive_dir):
        print(f"WARNING: {archive_dir} does not look like an _Archive directory", file=sys.stderr)

    photos_dir = album_sibling_dir(archive_dir, PHOTOS_SUFFIX)

    candidates = sorted(
        f for f in archive_dir.glob("*_D*-*.png") if DERIVED_NAME_RE.search(f.name)
    )
    if args.today:
        candidates = [f for f in candidates if _is_today(f)]

    if not candidates:
        print("No matching files found.")
        return

    col_w = max(len(f.name) for f in candidates)
    header = f"{'Archive file':<{col_w}}  {'Photos match?':<14}  {'Similarity':>10}  Status"
    print(header)
    print("-" * len(header))

    for archive_file in candidates:
        m = DERIVED_NAME_RE.search(archive_file.name)
        assert m

        photos_file = photos_dir / (archive_file.stem + "_V.jpg")

        if not photos_file.exists():
            print(f"{archive_file.name:<{col_w}}  {'NO PHOTOS':<14}  {'N/A':>10}  (skip)")
            continue

        img_a = _load_gray_thumb(archive_file)
        img_b = _load_gray_thumb(photos_file)
        if img_a is None or img_b is None:
            print(f"{archive_file.name:<{col_w}}  {'LOAD ERROR':<14}  {'N/A':>10}  (skip)")
            continue

        kp_a, des_a = _orb_descriptors(img_a)
        kp_b, des_b = _orb_descriptors(img_b)
        sim = _orb_similarity(kp_a, des_a, kp_b, des_b)

        if sim >= args.threshold:
            print(f"{archive_file.name:<{col_w}}  {'yes':<14}  {sim:>10.4f}  OK")
        else:
            new_d = _next_d_number(
                archive_dir, m.group("collection"), m.group("year"), m.group("book"), m.group("page")
            )
            new_path = _rename_archive_file(archive_file, m, new_d, args.execute)
            action = "RENAMED" if args.execute else "WOULD RENAME"
            print(f"{archive_file.name:<{col_w}}  {'yes':<14}  {sim:>10.4f}  {action} → {new_path.name}")
            if not args.no_preview:
                _viu(new_path if args.execute else archive_file, "Archive (new)", args.width)
                _viu(photos_file, "Photos (existing)", args.width)
                print()


# ---------------------------------------------------------------------------
# Subcommand: place
# ---------------------------------------------------------------------------

def cmd_place(args: argparse.Namespace) -> None:
    archive_dir: Path = args.archive_dir.expanduser().resolve()
    colorized_dir: Path = args.colorized_dir.expanduser().resolve()

    if not is_archive_dir(archive_dir):
        print(f"WARNING: {archive_dir} does not look like an _Archive directory", file=sys.stderr)
    if not colorized_dir.is_dir():
        print(f"Error: {colorized_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Build content hash index so we can detect already-copied files instantly
    print("Hashing archive images…")
    archive_hashes = _build_archive_hash_index(archive_dir)

    # Build ORB descriptor library for all Archive D##-## images
    print("Indexing archive images…")
    archive_entries: list[tuple[Path, re.Match[str], list, np.ndarray | None]] = []
    for f in sorted(archive_dir.iterdir()):
        if f.suffix.lower() not in ARCHIVE_IMAGE_EXTS:
            continue
        m = DERIVED_NAME_RE.search(f.name)
        if not m:
            continue
        img = _load_gray_thumb(f)
        if img is None:
            continue
        kp, des = _orb_descriptors(img)
        archive_entries.append((f, m, kp, des))
    print(f"  Indexed {len(archive_entries)} archive derived images.\n")

    colorized_files = sorted(
        f for f in colorized_dir.iterdir() if f.suffix.lower() == ".png"
    )
    if not colorized_files:
        print("No PNG files found in colorized directory.")
        return

    col_w = max(len(f.name) for f in colorized_files)

    # Track iter numbers allocated this run to avoid collisions between files
    # key: (collection, year, book, page, d_num) → next iter to assign
    allocated_iters: dict[tuple[str, str, str, str, int], int] = {}

    for color_file in colorized_files:
        # Check if this exact file is already in the archive
        color_hash = _file_hash(color_file)
        if color_hash in archive_hashes:
            existing = archive_hashes[color_hash]
            already_name = f"already copied - {color_file.name}"
            print(f"{color_file.name:<{col_w}}  ALREADY IN ARCHIVE ({existing.name}) — renaming source")
            if args.execute:
                color_file.rename(color_file.parent / already_name)
            else:
                print(f"  WOULD RENAME source → {already_name}")
            continue

        img_c = _load_gray_thumb(color_file)
        if img_c is None:
            print(f"{color_file.name:<{col_w}}  LOAD ERROR")
            continue

        kp_c, des_c = _orb_descriptors(img_c)

        # Find best matching archive image
        best_sim = 0.0
        best_entry: tuple[Path, re.Match[str], list, np.ndarray | None] | None = None
        for arch_file, arch_m, kp_a, des_a in archive_entries:
            sim = _orb_similarity(kp_c, des_c, kp_a, des_a)
            if sim > best_sim:
                best_sim = sim
                best_entry = (arch_file, arch_m, kp_a, des_a)

        if best_entry is None or best_sim < args.threshold:
            print(f"{color_file.name:<{col_w}}  NO MATCH  (best={best_sim:.4f})")
            continue

        arch_file, arch_m, _, _ = best_entry
        collection, year, book, page = _page_key(arch_m)
        d_num = int(arch_m.group("derived"))
        alloc_key = (collection, year, book, page, d_num)

        if alloc_key not in allocated_iters:
            allocated_iters[alloc_key] = _next_iter_number(
                archive_dir, collection, year, book, page, d_num
            )
        iter_num = allocated_iters[alloc_key]
        allocated_iters[alloc_key] = iter_num + 1

        new_stem = _build_stem(arch_m, d_num, iter_num)
        dest = archive_dir / (new_stem + ".png")

        action = "COPIED" if args.execute else "WOULD COPY"
        print(f"{color_file.name:<{col_w}}  sim={best_sim:.4f}  {action} → {dest.name}")
        print(f"  matched: {arch_file.name}")

        if not args.no_preview:
            _viu(color_file, "Colorized (source)", args.width)
            _viu(arch_file, "Archive match", args.width)
            print()

        if args.execute:
            shutil.copy2(color_file, dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared options
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--threshold", type=float, default=0.10, help="ORB match ratio threshold (default: 0.10)")
        p.add_argument("--execute", action="store_true", help="Apply changes (default: dry-run)")
        p.add_argument("--no-preview", action="store_true", help="Skip viu previews")
        p.add_argument("--width", type=int, default=60, help="viu width in columns (default: 60)")

    # check subcommand
    p_check = sub.add_parser("check", help="Check archive PNGs against Photos counterparts")
    p_check.add_argument("archive_dir", type=Path)
    p_check.add_argument("--today", action="store_true", help="Only check files modified today")
    _add_common(p_check)
    p_check.set_defaults(func=cmd_check)

    # place subcommand
    p_place = sub.add_parser("place", help="Place colorized PNGs into Archive by ORB matching")
    p_place.add_argument("archive_dir", type=Path)
    p_place.add_argument("colorized_dir", type=Path)
    _add_common(p_place)
    p_place.set_defaults(func=cmd_place)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
