from pathlib import Path

from common import (
    CREATOR,
    PAGE_SCAN_RE,
    PHOTO_ALBUMS_DIR,
    count_totals,
    file_modified_ts,
    list_archive_dirs,
)
from exiftool_utils import read_tag, write_tags
from naming import SCAN_TIFF_RE, format_book_display, parse_album_filename

NEW_NAME_RE = SCAN_TIFF_RE


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
    return read_tag(tif_path, tag)


def update_tif_metadata(tif_path: Path, header_text: str) -> bool:
    current_desc = get_tif_tag(tif_path, "XMP-dc:Description")
    current_creator = get_tif_tag(tif_path, "XMP-dc:Creator")

    creator_needs_fix = bool(current_creator and current_creator.count(CREATOR) > 1)

    if (
        current_desc == header_text
        and not creator_needs_fix
        and current_creator == CREATOR
    ):
        return False

    if creator_needs_fix:
        write_tags(
            tif_path,
            clear_tags=["XMP-dc:Creator"],
        )

    write_tags(
        tif_path,
        set_tags={
            "XMP-dc:Creator": CREATOR,
            "XMP-dc:Description": header_text,
        },
    )

    return True


def apply_metadata_to_archives(base_dir: Path = PHOTO_ALBUMS_DIR) -> dict[str, int]:
    updated = skipped = failures = 0

    archive_dirs = list_archive_dirs(base_dir)

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
        total_scans_for_page = (
            totals.get(key, {}).get("page_scans", {}).get(page_num, 1)
        )

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
    return {"updated": updated, "skipped": skipped, "failed": failures}


def main() -> None:
    apply_metadata_to_archives(PHOTO_ALBUMS_DIR)


if __name__ == "__main__":
    main()
