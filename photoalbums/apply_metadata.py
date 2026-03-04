import re
import subprocess
from pathlib import Path

from common import (
    CREATOR,
    PAGE_SCAN_RE,
    PHOTO_ALBUMS_DIR,
    count_totals,
    file_modified_ts,
    list_archive_dirs,
    parse_filename,
)

NEW_NAME_RE = re.compile(
    r"^[A-Z]{2,}_\d{4}(?:-\d{4})?_B(?:\d{2}|âˆ…)_P\d{2}_S\d{2}\.tif$",
    re.IGNORECASE,
)

FILENAME_RE = re.compile(
    r"(?P<collection>[A-Z]+)_(?P<year>\d{4}(?:-\d{4})?)_B(?P<book>\d{2}|âˆ…)_P(?P<page>\d+)_S\d+",
    re.IGNORECASE,
)


def parse_album_filename(filename: str):
    return parse_filename(filename, FILENAME_RE)


def format_book_display(book: str) -> str:
    return book if book == "âˆ…" else f"{int(book):02d}"


def build_header(
    collection: str,
    year: str,
    book: str,
    page: int,
    total_pages: int,
    scan_num: int,
    total_scans: int,
) -> str:
    return (
        f"{collection} ({year}) - Book {format_book_display(book)}, "
        f"Page {page:02d} of {total_pages:02d}, "
        f"Scan S{scan_num:02d} of {total_scans} total"
    )


def get_tif_tag(tif_path: Path, tag: str) -> str | None:
    try:
        result = subprocess.run(
            ["exiftool", f"-{tag}", "-s3", str(tif_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def update_tif_metadata(tif_path: Path, header_text: str) -> bool:
    current_desc = get_tif_tag(tif_path, "XMP-dc:Description")
    current_creator = get_tif_tag(tif_path, "XMP-dc:Creator")

    creator_needs_fix = bool(current_creator and current_creator.count(CREATOR) > 1)

    if current_desc == header_text and not creator_needs_fix and current_creator == CREATOR:
        return False

    if creator_needs_fix:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-XMP-dc:Creator=", str(tif_path)],
            check=True,
        )

    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            f"-XMP-dc:Creator={CREATOR}",
            f"-XMP-dc:Description={header_text}",
            str(tif_path),
        ],
        check=True,
    )

    return True


def main() -> None:
    updated = skipped = failures = 0

    archive_dirs = list_archive_dirs(PHOTO_ALBUMS_DIR)

    print("Counting total pages and scans per book...")
    totals = count_totals(archive_dirs, NEW_NAME_RE, parse_album_filename)

    for key, data in totals.items():
        total_scans = sum(data["page_scans"].values())
        print(f"{key}: {data['total_pages']} pages, {total_scans} total scans")
    print()

    all_tifs: list[Path] = []
    for archive in archive_dirs:
        for entry in archive.iterdir():
            if entry.is_file() and NEW_NAME_RE.fullmatch(entry.name):
                all_tifs.append(entry)

    all_tifs.sort(key=file_modified_ts, reverse=True)

    for tif_path in all_tifs:
        collection, year, book, page = parse_album_filename(tif_path.name)
        key = f"{collection}_{year}_B{book}"
        total_pages = totals.get(key, {}).get("total_pages", 0)
        page_num = int(page)
        total_scans_for_page = totals.get(key, {}).get("page_scans", {}).get(page_num, 1)

        scan_match = PAGE_SCAN_RE.search(tif_path.name)
        scan_num = int(scan_match.group("scan")) if scan_match else 1

        header = build_header(
            collection,
            year,
            book,
            int(page),
            total_pages,
            scan_num,
            total_scans_for_page,
        )

        try:
            if update_tif_metadata(tif_path, header):
                print(f"Updated TIFF metadata for {tif_path.name}")
                updated += 1
            else:
                print(f"TIFF metadata already current for {tif_path.name}")
                skipped += 1
        except Exception as exc:
            failures += 1
            print(f"Warning: Could not update TIFF metadata for {tif_path.name}: {exc}")

    print("\n===== SUMMARY =====")
    print("Updated:", updated)
    print("Skipped:", skipped)
    print("Failed:", failures)


if __name__ == "__main__":
    main()
