"""Delete stale crop files that are duplicates of crops on another page.

A re-pagination/re-scan can leave crops stranded under the wrong page number: the page
itself was re-rendered (and now has no RegionList, or different content), while the old
crops linger and are byte-identical copies of crops that already live -- correctly, with
a RegionList -- on a neighbouring page.

This tool flags an orphan-page crop for deletion ONLY when it is an exact (MD5) or
near-identical (perceptual-hash) duplicate of a crop on a DIFFERENT page whose sidecar
has a real mwg-rs:RegionList (the canonical home). Crops with no canonical duplicate are
never touched -- they are reported for separate handling.

Dry-run by default; pass --run to delete (crop .jpg + its .xmp sidecar).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.scripts.restore_orphan_regions import _has_region_list

_CROP_RE = re.compile(r"^(?P<stem>.+_P\d+)_D\d+-\d+_V\.jpg$", re.IGNORECASE)
_PHASH_NEAR = 2  # max Hamming distance (out of 256) to treat as the same image


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _phash(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((16, 16))
    arr = np.asarray(im, dtype=float)
    return (arr > arr.mean()).flatten()


def _page_xmp_for_crop(crop: Path) -> Path | None:
    m = _CROP_RE.match(crop.name)
    if not m:
        return None
    photos_dir = crop.parent
    if not photos_dir.name.endswith("_Photos"):
        return None
    pages_dir = photos_dir.parent / (photos_dir.name[: -len("_Photos")] + "_Pages")
    return pages_dir / f"{m.group('stem')}_V.xmp"


class _Crop:
    __slots__ = ("path", "page_stem", "canonical", "md5", "phash")

    def __init__(self, path: Path, page_stem: str, canonical: bool):
        self.path = path
        self.page_stem = page_stem
        self.canonical = canonical
        self.md5 = _md5(path)
        self.phash = _phash(path)


def _collect(root: Path, album_filter: str) -> list[_Crop]:
    album_filter = album_filter.casefold()
    region_cache: dict[Path, bool] = {}
    crops: list[_Crop] = []
    for crop in sorted(root.rglob("*_D*_V.jpg")):
        if not crop.parent.name.endswith("_Photos"):
            continue
        if album_filter and album_filter not in crop.parent.name.casefold():
            continue
        page_xmp = _page_xmp_for_crop(crop)
        if page_xmp is None:
            continue
        if page_xmp not in region_cache:
            region_cache[page_xmp] = _has_region_list(page_xmp)
        m = _CROP_RE.match(crop.name)
        crops.append(_Crop(crop, m.group("stem"), canonical=region_cache[page_xmp]))
    return crops


def _find_canonical_dup(orphan: _Crop, canonicals: list[_Crop]) -> _Crop | None:
    # Exact byte match first.
    for c in canonicals:
        if c.md5 == orphan.md5 and c.page_stem != orphan.page_stem:
            return c
    # Then near-identical image (re-encoded copy).
    for c in canonicals:
        if c.page_stem == orphan.page_stem:
            continue
        if int((orphan.phash != c.phash).sum()) <= _PHASH_NEAR:
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR))
    parser.add_argument("--album", default="", help="Substring filter on the _Photos dir name.")
    parser.add_argument("--run", action="store_true", help="Delete duplicates. Omit for a dry run.")
    args = parser.parse_args(argv)

    root = Path(args.photos_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {root}")

    crops = _collect(root, str(args.album or ""))
    canonicals = [c for c in crops if c.canonical]
    orphans = [c for c in crops if not c.canonical]
    print(f"scanned {len(crops)} crops: {len(canonicals)} canonical, {len(orphans)} on pages without a RegionList\n")

    to_delete: list[tuple[_Crop, _Crop]] = []
    unique_orphans: list[_Crop] = []
    for orphan in orphans:
        dup = _find_canonical_dup(orphan, canonicals)
        if dup is not None:
            to_delete.append((orphan, dup))
        else:
            unique_orphans.append(orphan)

    for orphan, dup in to_delete:
        kind = "exact" if orphan.md5 == dup.md5 else "near"
        print(f"DUP   {orphan.path.name}  ==({kind})==>  {dup.path.name}")

    if unique_orphans:
        print(f"\n{len(unique_orphans)} orphan crop(s) with NO canonical duplicate (left untouched):")
        for o in unique_orphans:
            print(f"  KEEP  {o.path.name}")

    deleted = 0
    if args.run:
        for orphan, _ in to_delete:
            orphan.path.unlink()
            sidecar = orphan.path.with_suffix(".xmp")
            if sidecar.is_file():
                sidecar.unlink()
            deleted += 1
        print(f"\ndone: deleted {deleted} duplicate crop(s) (+ sidecars); {len(unique_orphans)} unique orphan(s) kept")
    else:
        print(f"\ndry-run: would delete {len(to_delete)} duplicate crop(s); {len(unique_orphans)} unique orphan(s) kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
