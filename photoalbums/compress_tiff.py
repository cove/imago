from pathlib import Path

from common import (
    PHOTO_ALBUMS_DIR,
    configure_imagemagick,
    file_created_ts,
    list_archive_dirs,
    process_tiff_in_place,
    tiff_needs_conversion,
)


def convert_directory(base_dir: Path) -> None:
    archive_dirs = list_archive_dirs(base_dir)

    all_tiffs: list[Path] = []
    for archive in archive_dirs:
        all_tiffs.extend(archive.rglob("*.tif"))

    all_tiffs.sort(key=file_created_ts, reverse=True)

    for tiff_path in all_tiffs:
        if not tiff_needs_conversion(tiff_path):
            print(f"Skipped (already correct): {tiff_path}")
            continue
        if process_tiff_in_place(tiff_path, log_error=print):
            print(f"Processed and replaced: {tiff_path}")


def main() -> None:
    configure_imagemagick()
    convert_directory(PHOTO_ALBUMS_DIR)


if __name__ == "__main__":
    main()
