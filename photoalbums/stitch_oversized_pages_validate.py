import os

try:
    import cv2
except Exception:
    cv2 = None

from common import (
    PHOTO_ALBUMS_DIR,
    count_totals,
    dir_created_ts,
    file_created_ts,
    list_archive_dirs,
    list_page_scan_groups,
)
from naming import SCAN_TIFF_RE, parse_album_filename

try:
    from stitch_oversized_pages import build_stitched_image
except Exception:
    from .stitch_oversized_pages import build_stitched_image

NEW_NAME_RE = SCAN_TIFF_RE


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("cv2 is required to validate images.")


def validate_single(tif_path: str) -> None:
    _require_cv2()
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Could not read image")

    if img.ndim == 2:
        _ = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        _ = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def validate_stitch(files, stitcher_factory=None) -> None:
    print("Validating stitch:", [os.path.basename(f) for f in files])
    build_stitched_image(files, stitcher_factory=stitcher_factory)


def main() -> None:
    success = failures = 0
    failed = []

    archive_dirs = list_archive_dirs(PHOTO_ALBUMS_DIR)
    archive_dirs.sort(key=dir_created_ts, reverse=True)

    print("Counting total pages and scans per book...")
    totals = count_totals(archive_dirs, NEW_NAME_RE, parse_album_filename)

    for key, data in totals.items():
        total_scans = sum(data["page_scans"].values())
        print(f"{key}: {data['total_pages']} pages, {total_scans} total scans")
    print()

    for archive in archive_dirs:
        groups = list_page_scan_groups(archive, NEW_NAME_RE)
        groups.sort(key=lambda g: max(file_created_ts(p) for p in g), reverse=True)

        for group in groups:
            try:
                if len(group) > 1:
                    validate_stitch(group)
                else:
                    validate_single(group[0])
                success += 1
            except Exception as exc:
                failures += 1
                failed.append(group)
                print("Error:", exc)

    print("\n===== SUMMARY =====")
    print("Processed:", success + failures)
    print("Successful:", success)
    print("Failed:", failures)
    for group in failed:
        print(" -", ", ".join(os.path.basename(x) for x in group))


if __name__ == "__main__":
    main()
