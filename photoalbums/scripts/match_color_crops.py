"""match_color_crops.py — Match colorized crops to archive scans and migrate to Archive.

For each file in _Color directories:
1. SIFT-match (grayscale) against archive TIFs for the same book.
2. Pick the best match by RANSAC inlier count.
3. Derive _P## from the matched scan; assign _D##-01 in spatial order,
   reusing an existing D## if the bounding box overlaps (IoU >= 0.7).
4. Copy/convert to PNG, rename to _P##_D##-01_C.png, move to _Archive.
5. Preserve the source file's read-only state on the canonical output.

Unmatched files are written to a report; nothing is moved without a confident match.

Usage
-----
  # Dry run:
  uv run python photoalbums/scripts/match_color_crops.py

  # Execute:
  uv run python photoalbums/scripts/match_color_crops.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import shutil
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    cv2 = None
    np = None
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from naming import DERIVED_NAME_RE, parse_album_filename

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHOTO_ALBUMS_ROOT = Path(r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums")
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / ".tmp" / "match_color_crops_manifest.json"
REPORT_PATH = REPO_ROOT / ".tmp" / "match_color_crops_report.json"

MIN_INLIERS = 10
IOU_THRESHOLD = 0.7
MAX_DIM = 2000  # resize long edge to this before SIFT

_SCAN_TIF_RE = re.compile(r"_S\d+\.tif$", re.IGNORECASE)
_PAGE_FROM_SCAN_RE = re.compile(r"_P(\d+)_S", re.IGNORECASE)
_CANONICAL_CROP_RE = re.compile(r"^(?P<prefix>.+_P\d+)_D(?P<derived>\d+)-01_C\.png$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_cv2() -> None:
    if cv2 is None or np is None or Image is None:
        raise RuntimeError("cv2, numpy, and pillow are required for this script.")


def _load_gray(path: Path, max_dim: int = MAX_DIM) -> tuple:
    """Load image as grayscale, resize to max_dim on longest edge.
    Returns (gray_array, scale_factor)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        with Image.open(path) as pil:
            img = np.array(pil.convert("L"))
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def _bbox_rect(poly: "np.ndarray") -> tuple[int, int, int, int]:
    """Axis-aligned bounding rect from a 4-point polygon. Returns (x, y, w, h)."""
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    return x, y, w, h


def _iou(r1: tuple, r2: tuple) -> float:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _spatial_key(bbox: "np.ndarray") -> tuple[float, float]:
    """Sort key: top-to-bottom, left-to-right by bbox centre."""
    x, y, w, h = _bbox_rect(bbox)
    return y + h / 2, x + w / 2


# ---------------------------------------------------------------------------
# SIFT matching
# ---------------------------------------------------------------------------


def _compute_descriptors(sift, gray: "np.ndarray") -> tuple:
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def _match_query_to_target(
    matcher,
    qkp,
    qdes,
    tkp,
    tdes,
    tscale: float,
    query_shape: tuple,
    min_inliers: int,
) -> tuple["np.ndarray", int] | None:
    """Match query descriptors against target. Returns (bbox_in_orig_target, inliers) or None."""
    if qdes is None or tdes is None or len(qkp) < 4 or len(tkp) < 4:
        return None
    try:
        raw = matcher.knnMatch(qdes, tdes, k=2)
    except cv2.error:
        return None
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or np.linalg.det(H[:2, :2]) <= 0:
        return None
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return None
    qh, qw = query_shape[:2]
    corners = np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2)
    bbox_scaled = cv2.perspectiveTransform(corners, H)
    return bbox_scaled / tscale, inliers


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_color_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dp = Path(dirpath)
        if dp.name.endswith("_Color"):
            print(f"Found color directory: {dp}")
            for fname in sorted(filenames):
                if Path(fname).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    files.append(dp / fname)
    return files


def _sibling_dir(color_dir: Path, suffix: str) -> Path:
    base = color_dir.name[: -len("_Color")]
    return color_dir.parent / f"{base}{suffix}"


def _existing_d_crops(archive_dir: Path, page: int) -> dict[tuple[int, int], Path]:
    """Return {(d1, d2): path} for _D##-##_C archive crops on a given page."""
    result: dict[tuple[int, int], Path] = {}
    if not archive_dir.is_dir():
        return result
    for f in archive_dir.iterdir():
        if f.suffix.lower() not in {".png"}:
            continue
        m = DERIVED_NAME_RE.search(f.stem)
        if m and int(m.group("page")) == page:
            result[(int(m.group("derived")), int(m.group("iter")))] = f
    return result


def _preserve_read_only(source: Path, dest: Path) -> None:
    dest.chmod(source.stat().st_mode)


def _remove_legacy_archive_variants(dest: Path) -> None:
    match = _CANONICAL_CROP_RE.match(dest.name)
    if not match:
        return
    legacy_re = re.compile(
        rf"^{re.escape(match.group('prefix'))}_D{re.escape(match.group('derived'))}_\d+\.(jpg|png|xmp)$",
        re.IGNORECASE,
    )
    for sibling in dest.parent.iterdir():
        if not sibling.is_file() or not legacy_re.match(sibling.name):
            continue
        sibling.chmod(sibling.stat().st_mode | stat.S_IWRITE)
        sibling.unlink()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def build_plan(root: Path, min_inliers: int = MIN_INLIERS) -> tuple[list[dict], list[str]]:
    _require_cv2()

    color_files = discover_color_files(root)
    print(f"Discovered {len(color_files)} color image files.")
    if not color_files:
        return [], []

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    plan: list[dict] = []
    unmatched: list[str] = []

    # Group color files by archive dir
    by_archive: dict[Path, list[Path]] = {}
    for cf in color_files:
        archive = _sibling_dir(cf.parent, "_Archive")
        if not archive.is_dir():
            print(f"Skipping {cf.name}: archive directory missing for {cf.parent.name}")
            unmatched.append(f"no archive dir: {cf}")
            continue
        by_archive.setdefault(archive, []).append(cf)

    for archive_dir, crops in sorted(by_archive.items()):
        tifs = sorted(f for f in archive_dir.iterdir() if _SCAN_TIF_RE.search(f.name))
        print(f"\nArchive {archive_dir.name}: {len(crops)} color crops, {len(tifs)} scan TIFs")
        if not tifs:
            for cf in crops:
                unmatched.append(f"no archive TIFs in {archive_dir.name}: {cf.name}")
            continue

        # Pre-compute TIF descriptors (cached per archive dir)
        tif_cache: list[dict] = []
        for tif in tifs:
            m = _PAGE_FROM_SCAN_RE.search(tif.name)
            if not m:
                continue
            try:
                gray, scale = _load_gray(tif)
            except Exception as exc:
                print(f"  WARN: could not load {tif.name}: {exc}", file=sys.stderr)
                continue
            kp, des = _compute_descriptors(sift, gray)
            tif_cache.append({"path": tif, "page": int(m.group(1)), "gray": gray, "scale": scale, "kp": kp, "des": des})
        print(f"  Prepared {len(tif_cache)} scan descriptors")

        # Match each color crop
        matched_items: list[dict] = []
        for crop in crops:
            print(f"  Matching {crop.name} ...")
            try:
                qgray, _ = _load_gray(crop)
            except Exception as exc:
                print(f"    load error: {exc}")
                unmatched.append(f"load error {crop.name}: {exc}")
                continue
            qkp, qdes = _compute_descriptors(sift, qgray)

            best: dict | None = None
            for td in tif_cache:
                result = _match_query_to_target(
                    matcher, qkp, qdes, td["kp"], td["des"], td["scale"], qgray.shape, min_inliers
                )
                if result and (best is None or result[1] > best["inliers"]):
                    best = {
                        "crop": crop,
                        "scan": td["path"],
                        "page": td["page"],
                        "bbox": result[0],
                        "inliers": result[1],
                    }

            if best is None:
                print("    no confident match")
                unmatched.append(f"no confident match: {crop.name}")
            else:
                print(f"    matched {best['scan'].name} page {best['page']:02d} with {best['inliers']} inliers")
                matched_items.append(best)

        # Assign D## numbers per page
        by_page: dict[int, list[dict]] = {}
        for item in matched_items:
            by_page.setdefault(item["page"], []).append(item)

        for page, items in sorted(by_page.items()):
            print(f"  Assigning D numbers for page {page:02d} ({len(items)} matched crops)")
            existing = _existing_d_crops(archive_dir, page)
            max_d1 = max((k[0] for k in existing), default=0)

            # Get bboxes of existing crops (best-effort)
            existing_bboxes: dict[tuple[int, int], tuple] = {}
            for key, epath in existing.items():
                try:
                    egray, _ = _load_gray(epath)
                    ekp, edes = _compute_descriptors(sift, egray)
                    for td in tif_cache:
                        if td["page"] != page:
                            continue
                        result = _match_query_to_target(
                            matcher, ekp, edes, td["kp"], td["des"], td["scale"], egray.shape, min_inliers=6
                        )
                        if result:
                            existing_bboxes[key] = _bbox_rect(result[0])
                            break
                except Exception:
                    pass

            items.sort(key=lambda i: _spatial_key(i["bbox"]))
            next_d1 = max_d1 + 1

            for item in items:
                item_rect = _bbox_rect(item["bbox"])
                d1 = next(
                    (k[0] for k, r in existing_bboxes.items() if _iou(item_rect, r) >= IOU_THRESHOLD),
                    None,
                )
                if d1 is None:
                    d1 = next_d1
                    next_d1 += 1

                collection, year, book, _ = parse_album_filename(item["scan"].name)
                new_name = f"{collection}_{year}_B{book}_P{page:02d}_D{d1:02d}-01_C.png"
                plan.append(
                    {
                        "crop": str(item["crop"]),
                        "scan": str(item["scan"]),
                        "page": page,
                        "d1": d1,
                        "inliers": item["inliers"],
                        "new_path": str(archive_dir / new_name),
                    }
                )
                print(f"    planned {item['crop'].name} -> {new_name}")

    return plan, unmatched


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_plan(plan: list[dict]) -> list[dict]:
    results: list[dict] = []
    for entry in plan:
        crop = Path(entry["crop"])
        dest = Path(entry["new_path"])
        result = {**entry, "status": "ok", "error": ""}
        print(f"Migrating {crop.name} -> {dest.name}")
        try:
            if not crop.exists():
                result["status"] = "skipped_missing"
                print("  skipped: source missing")
            elif dest.exists():
                result["status"] = "skipped_conflict"
                result["error"] = f"Target exists: {dest}"
                print(f"  skipped: target exists {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if crop.suffix.lower() in {".jpg", ".jpeg"}:
                    with Image.open(crop) as img:
                        img.save(dest, "PNG")
                else:
                    shutil.copy2(crop, dest)
                _preserve_read_only(crop, dest)
                _remove_legacy_archive_variants(dest)
                print("  wrote PNG")
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"  error: {exc}")
        results.append(result)
    return results


def backup_color_dirs(plan: list[dict]) -> list[Path]:
    color_dirs = sorted({Path(entry["crop"]).parent for entry in plan})
    backups: list[Path] = []
    for color_dir in color_dirs:
        backup_dir = color_dir.with_name(f"{color_dir.name}.bak")
        if backup_dir.exists():
            raise RuntimeError(f"backup already exists: {backup_dir}")
        print(f"Backing up {color_dir} -> {backup_dir}")
        shutil.copytree(color_dir, backup_dir)
        backups.append(backup_dir)
    return backups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PHOTO_ALBUMS_ROOT))
    parser.add_argument("--run", action="store_true", help="Execute (default is dry-run)")
    parser.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root} ...")
    plan, unmatched = build_plan(root, min_inliers=args.min_inliers)

    print(f"\nMatched: {len(plan)}  Unmatched: {len(unmatched)}")
    if unmatched:
        print("Unmatched:")
        for u in unmatched:
            print(f"  {u}")

    manifest = {"root": str(root), "plan": plan, "unmatched": unmatched}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    REPORT_PATH.write_text(json.dumps({"unmatched": unmatched}, indent=2, ensure_ascii=False))
    print(f"Manifest: {MANIFEST_PATH}")

    if not args.run:
        print("\nDRY RUN — pass --run to execute.")
        for entry in plan[:15]:
            print(f"  [{entry['inliers']} inliers] {Path(entry['crop']).name}  ->  {Path(entry['new_path']).name}")
        if len(plan) > 15:
            print(f"  ... and {len(plan) - 15} more")
        return 0

    color_dir_count = len({Path(entry["crop"]).parent for entry in plan})
    print(f"\nBacking up {color_dir_count} _Color directories ...")
    backups = backup_color_dirs(plan)
    print(f"Created {len(backups)} backups.")

    print(f"\nExecuting {len(plan)} migrations ...")
    results = execute_plan(plan)
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = [r for r in results if r["status"] not in {"ok", "skipped_missing"}]
    print(f"Done: {ok}/{len(results)} ok")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e['status']}: {e['crop']} — {e['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
