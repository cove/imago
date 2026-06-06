"""Reconstruct a page's missing mwg-rs:RegionList from the crops already on disk.

Some pages lost their RegionList while their crop JPEGs survived: a ``no_regions``
re-detection (or a post-crop sidecar rewrite) erased the region markup but left the
``_D##-##_V.jpg`` crops in the ``_Photos`` sibling. Neither the page sidecar nor the
crop sidecars retain the source bounding boxes, so this tool recovers them by locating
each crop inside the page image via normalized cross-correlation, then rewrites the
page RegionList with the recovered geometry.

Only crops that are genuine sub-images of the page can be placed. Crops larger than the
page (rotated/whole-page reprocesses, typical of cover ``*_P01`` pages) or that match
with low confidence are reported and skipped, never guessed.

Dry-run by default; pass --run to write.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib.ai_view_regions import (
    RegionResult,
    RegionWithCaption,
    validate_region_set,
)
from photoalbums.lib.xmp_sidecar import write_region_list
from photoalbums.scripts._repair_args import build_repair_parser

# Page filename: <album>_P<NN>_V.jpg ; crop: <album>_P<NN>_D<dd>-<ii>_V.jpg
_CROP_RE = re.compile(r"_D(\d+)-(\d+)_V\.jpg$", re.IGNORECASE)
# Longest edge (px) the page is downscaled to before matching. Keeps 60MP pages fast;
# coordinates are scaled back up afterwards (sub-pixel error is negligible for regions).
_MATCH_LONGEST_EDGE = 2000
# TM_CCOEFF_NORMED score below which we refuse to place a crop. Exact sub-images score
# ~0.95+; this leaves a wide safety margin while rejecting non-matches.
_MIN_SCORE = 0.80


def _has_region_list(xmp_path: Path) -> bool:
    try:
        return "mwg-rs:RegionList" in xmp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _base_crops_for_page(photos_dir: Path, page_stem: str) -> dict[int, Path]:
    """Return {base_D_number: crop_path}, one per base photo (prefer the -00 iteration)."""
    chosen: dict[int, tuple[int, Path]] = {}
    prefix = f"{page_stem}_D"
    for crop in photos_dir.glob(f"{prefix}*_V.jpg"):
        m = _CROP_RE.search(crop.name)
        if not m:
            continue
        base = int(m.group(1))
        iteration = int(m.group(2))
        # Lowest iteration wins (the -00 original crop where present).
        if base not in chosen or iteration < chosen[base][0]:
            chosen[base] = (iteration, crop)
    return {base: path for base, (_, path) in sorted(chosen.items())}


def _match_crop(page_gray: np.ndarray, crop_path: Path) -> tuple[int, int, int, int, float] | None:
    """Locate ``crop_path`` inside ``page_gray``. Returns (x, y, w, h, score) or None."""
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        return None
    page_h, page_w = page_gray.shape[:2]
    crop_h, crop_w = crop.shape[:2]
    if crop_w > page_w or crop_h > page_h:
        return None  # not a sub-image of this page

    scale = min(1.0, _MATCH_LONGEST_EDGE / float(max(page_h, page_w)))
    if scale < 1.0:
        pg = cv2.resize(page_gray, (round(page_w * scale), round(page_h * scale)), interpolation=cv2.INTER_AREA)
        cg = cv2.resize(crop, (max(1, round(crop_w * scale)), max(1, round(crop_h * scale))), interpolation=cv2.INTER_AREA)
    else:
        pg, cg = page_gray, crop
    if cg.shape[0] > pg.shape[0] or cg.shape[1] > pg.shape[1]:
        return None

    result = cv2.matchTemplate(pg, cg, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x = int(round(max_loc[0] / scale))
    y = int(round(max_loc[1] / scale))
    x = max(0, min(x, page_w - 1))
    y = max(0, min(y, page_h - 1))
    w = min(crop_w, page_w - x)
    h = min(crop_h, page_h - y)
    return x, y, w, h, float(max_val)


def _restore_page(view_jpg: Path, *, dry_run: bool) -> str:
    xmp_path = view_jpg.with_suffix(".xmp")
    photos_dir = view_jpg.parent.parent / (view_jpg.parent.name[: -len("_Pages")] + "_Photos")
    page_stem = view_jpg.stem  # e.g. England_1983_B01_P51_V
    page_stem = page_stem[:-2] if page_stem.endswith("_V") else page_stem
    if not photos_dir.is_dir():
        return "skip"

    base_crops = _base_crops_for_page(photos_dir, page_stem)
    if not base_crops:
        return "skip"

    page = cv2.imread(str(view_jpg), cv2.IMREAD_GRAYSCALE)
    if page is None:
        print(f"FAIL  {view_jpg.name}: page image unreadable")
        return "fail"
    page_h, page_w = page.shape[:2]

    candidates: list[RegionResult] = []
    unmatched: list[tuple[int, str]] = []
    for index, (base, crop_path) in enumerate(base_crops.items()):
        match = _match_crop(page, crop_path)
        if match is None or match[4] < _MIN_SCORE:
            score = "n/a" if match is None else f"{match[4]:.3f}"
            unmatched.append((base, score))
            continue
        x, y, w, h, score = match
        candidates.append(
            RegionResult(index=index, x=x, y=y, width=w, height=h, confidence=score, photo_number=base)
        )

    validation = validate_region_set(candidates, img_w=page_w, img_h=page_h)
    kept = validation.kept
    dropped = {f.region_index: f.reason for f in validation.failures}

    print(f"PAGE  {view_jpg.name}  page={page_w}x{page_h}  crops={len(base_crops)} matched={len(kept)}")
    for region in candidates:
        flag = f"  DROPPED({dropped[region.index]})" if region.index in dropped else ""
        print(f"      D{region.photo_number:02d} score={region.confidence:.3f} "
              f"box=({region.x},{region.y},{region.width},{region.height}){flag}")
    for base, score in unmatched:
        print(f"      D{base:02d} score={score}  UNMATCHED (crop not a sub-image / low confidence)")

    if not kept or len(kept) != len(base_crops):
        # Partial reconstructions are unsafe: writing fewer regions than crops would
        # silently mis-link captions/crops. Require a complete, confident match.
        print(f"      -> NOT WRITTEN (need all {len(base_crops)} crops matched)")
        return "skip"

    if dry_run:
        print("      -> would write RegionList")
        return "would_fix"

    regions_with_captions = [RegionWithCaption(region, "") for region in kept]
    write_region_list(xmp_path, regions_with_captions, page_w, page_h)
    print(f"      -> wrote {len(kept)} region(s)")
    return "fixed"


def _iter_orphan_pages(photos_root: Path, album_filter: str, exclude: str) -> list[Path]:
    album_filter = str(album_filter or "").casefold()
    exclude = str(exclude or "").casefold()
    pages: list[Path] = []
    for view_jpg in sorted(photos_root.rglob("*_V.jpg")):
        if not view_jpg.parent.name.endswith("_Pages"):
            continue
        if album_filter and album_filter not in view_jpg.parent.name.casefold():
            continue
        if exclude and exclude in view_jpg.parent.name.casefold():
            continue
        if _has_region_list(view_jpg.with_suffix(".xmp")):
            continue
        pages.append(view_jpg)
    return pages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_repair_parser(
        description="Rebuild missing page mwg-rs:RegionList from existing crops via template matching.",
        default_photos_root=str(PHOTO_ALBUMS_DIR),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Substring to exclude from album directory names (e.g. 'TravelPostCards').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")

    dry_run = not bool(args.run)
    counts = {"fixed": 0, "would_fix": 0, "skip": 0, "fail": 0}
    for view_jpg in _iter_orphan_pages(photos_root, str(args.album or ""), str(args.exclude or "")):
        outcome = _restore_page(view_jpg, dry_run=dry_run)
        counts[outcome] = counts.get(outcome, 0) + 1

    print()
    if dry_run:
        print(f"dry-run would_fix={counts['would_fix']} skip={counts['skip']} fail={counts['fail']}")
    else:
        print(f"done fixed={counts['fixed']} skip={counts['skip']} fail={counts['fail']}")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
