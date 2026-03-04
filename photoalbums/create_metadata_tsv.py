#!/usr/bin/env python3
"""
Extract metadata from image and video files into a TSV.
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from common import CREATOR, PHOTO_ALBUMS_DIR

EXIFTOOL_FIELDS = [
    "IFD0:ImageDescription",
    "XMP-iptcExt:ArtworkContentDescription",
    "XMP-iptcExt:PersonInImage",
    "XMP-dc:Subject",
    "XMP-dc:Description",
    "XMP-dc:Creator",
    "XMP-photoshop:CaptionWriter",
    "XMP-photoshop:City",
    "XMP-photoshop:State",
    "XMP-photoshop:Country",
    "XMP-iptcCore:Location",
    "XMP-iptcCore:CountryCode",
    "XMP-iptcCore:SubjectCode",
    "XMP-iptcCore:ExtDescrAccessibility",
    "XMP-lr:HierarchicalSubject",
    "IPTC:CodedCharacterSet",
    "IPTC:ApplicationRecordVersion",
    "IPTC:Keywords",
    "IPTC:City",
    "IPTC:Sub-location",
    "IPTC:Province-State",
    "IPTC:Country-PrimaryLocationCode",
    "IPTC:Country-PrimaryLocationName",
    "IPTC:Caption-Abstract",
    "IPTC:Writer-Editor",
]

OUTPUT_FILE = "metadata.tsv"
FILE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".mp4"}

SCAN_FILENAME_RE = re.compile(
    r"(?P<collection>[^_]+)_(?P<year>\d{4}(?:-\d{4})?)_B(?P<book>\d{2})_P(?P<page>\d+)_S(?P<scan>\d+)",
    re.IGNORECASE,
)

DERIVED_FILENAME_RE = re.compile(
    r"(?P<collection>[^_]+)_(?P<year>\d{4}(?:-\d{4})?)_B(?P<book>\d{2})_P(?P<page>\d+)_D(?P<derived>\d{1,2})_(?P<iter>\d{1,2})",
    re.IGNORECASE,
)


def extract_metadata(file_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-json",
                "-a",
                "-G1",
                "-struct",
                "-charset",
                "UTF8",
                *[f"-{tag}" for tag in EXIFTOOL_FIELDS],
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        metadata = json.loads(result.stdout)
        return metadata[0] if metadata else {}
    except subprocess.CalledProcessError as exc:
        print(f"Error reading {file_path}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"Error parsing metadata for {file_path}: {exc}")
        return {}


def find_files(base_dir: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in base_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in extensions
            and any(parent.name.endswith("_Archive") for parent in path.parents)
            and not any(parent.name.endswith("_View") for parent in path.parents)
        ):
            files.append(path)
    return files


def format_book_display(book: str) -> str:
    return f"{int(book):02d}"


def build_scan_description(
    collection: str,
    year: str,
    book: str,
    page: int,
    scan_num: int,
    total_scans: int,
) -> str:
    return (
        f"{collection} ({year}) - Book {format_book_display(book)}, "
        f"Page {page:02d}, Scan S{scan_num:02d} of {total_scans} total"
    )


def build_derived_description(
    collection: str,
    year: str,
    book: str,
    page: int,
    derived_code: str,
    iter_num: int,
    total_iters: int,
) -> str:
    return (
        f"{collection} ({year}) - Book {format_book_display(book)}, "
        f"Page {page:02d}, Derived D{derived_code}_{iter_num:02d} of {total_iters} total"
    )


PHRASE_JOINERS = {
    "North",
    "South",
    "East",
    "West",
    "Northern",
    "Southern",
    "Eastern",
    "Western",
    "Central",
    "Middle",
    "New",
    "United",
    "Panama",
}


def split_camel_words(value: str) -> str:
    if not value:
        return value
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return value.strip()


def split_camel_keywords(value: str) -> list[str]:
    if not value:
        return []
    words = split_camel_words(value).split()
    keywords: list[str] = []
    idx = 0
    while idx < len(words):
        if idx + 1 < len(words) and words[idx] in PHRASE_JOINERS:
            keywords.append(f"{words[idx]} {words[idx + 1]}")
            idx += 2
        else:
            keywords.append(words[idx])
            idx += 1
    return keywords


def merge_keyword(values, keyword: str) -> list[str]:
    items: list[str] = []
    if isinstance(values, list):
        items.extend(values)
    elif isinstance(values, str) and values:
        items.append(values)
    if keyword and keyword not in items:
        items.append(keyword)
    return items


def merge_keywords(values, keywords: list[str]) -> list[str]:
    items = merge_keyword(values, "")
    for keyword in keywords:
        if keyword and keyword not in items:
            items.append(keyword)
    return items


def build_scan_totals(files: list[Path]) -> dict[Path, dict[int, int]]:
    totals: dict[Path, dict[int, int]] = {}
    for file_path in files:
        match = SCAN_FILENAME_RE.search(file_path.name)
        if not match:
            continue
        page_num = int(match.group("page"))
        scan_num = int(match.group("scan"))
        by_page = totals.setdefault(file_path.parent, {})
        by_page[page_num] = max(by_page.get(page_num, 0), scan_num)
    return totals


def build_derived_totals(files: list[Path]) -> dict[Path, dict[int, dict[str, int]]]:
    totals: dict[Path, dict[int, dict[str, int]]] = {}
    for file_path in files:
        match = DERIVED_FILENAME_RE.search(file_path.name)
        if not match:
            continue
        page_num = int(match.group("page"))
        derived_code = match.group("derived")
        iter_num = int(match.group("iter"))
        by_page = totals.setdefault(file_path.parent, {})
        by_code = by_page.setdefault(page_num, {})
        by_code[derived_code] = max(by_code.get(derived_code, 0), iter_num)
    return totals


def collect_all_metadata(
    files: list[Path],
    scan_totals_by_dir: dict[Path, dict[int, int]],
    derived_totals_by_dir: dict[Path, dict[int, dict[str, int]]],
    progress: bool = True,
) -> list[dict]:
    all_metadata = []
    keyword_fields_found = set()

    total = len(files)
    for idx, file_path in enumerate(files, 1):
        if progress:
            print(f"Processing {idx}/{total}: {file_path.name}")

        metadata = extract_metadata(file_path)
        if not metadata:
            continue

        try:
            rel_path = file_path.relative_to(PHOTO_ALBUMS_DIR)
            metadata["FilePath"] = str(rel_path)
        except ValueError:
            metadata["FilePath"] = str(file_path)

        scan_match = SCAN_FILENAME_RE.search(file_path.name)
        derived_match = DERIVED_FILENAME_RE.search(file_path.name)
        if scan_match:
            page_num = int(scan_match.group("page"))
            scan_num = int(scan_match.group("scan"))
            total_scans = scan_totals_by_dir.get(file_path.parent, {}).get(page_num, scan_num)
            metadata["XMP-dc:Description"] = build_scan_description(
                scan_match.group("collection"),
                scan_match.group("year"),
                scan_match.group("book"),
                page_num,
                scan_num,
                total_scans,
            )
            keywords = split_camel_keywords(scan_match.group("collection"))
            metadata["IPTC:Keywords"] = merge_keywords(metadata.get("IPTC:Keywords"), keywords)
        elif derived_match:
            page_num = int(derived_match.group("page"))
            iter_num = int(derived_match.group("iter"))
            derived_code = derived_match.group("derived")
            total_iters = (
                derived_totals_by_dir.get(file_path.parent, {})
                .get(page_num, {})
                .get(derived_code, iter_num)
            )
            metadata["XMP-dc:Description"] = build_derived_description(
                derived_match.group("collection"),
                derived_match.group("year"),
                derived_match.group("book"),
                page_num,
                derived_code,
                iter_num,
                total_iters,
            )
            keywords = split_camel_keywords(derived_match.group("collection"))
            metadata["IPTC:Keywords"] = merge_keywords(metadata.get("IPTC:Keywords"), keywords)
        else:
            metadata["XMP-dc:Description"] = ""
        metadata["XMP-dc:Creator"] = CREATOR

        all_metadata.append(metadata)

        for key, value in metadata.items():
            if value not in ("", None, [], {}):
                if "keyword" in key.lower() or "subject" in key.lower() or "tag" in key.lower():
                    keyword_fields_found.add(key)

    if keyword_fields_found:
        print("\nKeyword-related fields found with values:")
        for field in sorted(keyword_fields_found):
            print(f"  - {field}")
    else:
        print("\nWARNING: No keyword-related fields found with values!")
        print("Check if keywords are actually set in your files.")

    return all_metadata


def write_tsv(all_metadata: list[dict], output_file: str) -> None:
    if not all_metadata:
        print("No metadata found!")
        return

    fields = ["FilePath"] + EXIFTOOL_FIELDS

    print(f"\nWriting {len(all_metadata)} records with {len(fields)} fields to {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for metadata in all_metadata:
            row = {}
            for field in fields:
                value = metadata.get(field, "")
                if isinstance(value, (list, dict)):
                    row[field] = json.dumps(value, ensure_ascii=False)
                else:
                    row[field] = value
            writer.writerow(row)

    print(f"Successfully wrote {output_file}")


def main() -> None:
    try:
        subprocess.run(["exiftool", "-ver"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: exiftool is not installed or not in PATH")
        print("Install from: https://exiftool.org/")
        sys.exit(1)

    if not PHOTO_ALBUMS_DIR.is_dir():
        print(f"Error: Directory does not exist: {PHOTO_ALBUMS_DIR}")
        sys.exit(1)

    print("=" * 70)
    print("Metadata Extraction Tool")
    print("=" * 70)
    print(f"Directory: {PHOTO_ALBUMS_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Extensions: {', '.join(sorted(FILE_EXTENSIONS))}")
    print()

    print("Scanning for files...")
    files = find_files(PHOTO_ALBUMS_DIR, FILE_EXTENSIONS)
    print(f"Found {len(files)} files")

    if not files:
        print("No files found!")
        sys.exit(0)

    print()

    scan_totals_by_dir = build_scan_totals(files)
    derived_totals_by_dir = build_derived_totals(files)
    all_metadata = collect_all_metadata(
        files,
        scan_totals_by_dir,
        derived_totals_by_dir,
        progress=True,
    )

    print()
    write_tsv(all_metadata, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
