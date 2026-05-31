from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

from ..naming import pages_dir_for_album_dir, parse_album_filename
from .xmp_sidecar import (
    DC_NS,
    EXIF_NS,
    IMAGO_NS,
    RDF_NS,
    XMP_NS,
    _get_or_create_rdf_desc,
    _read_or_create_xmp_tree,
    _set_seq_text,
    _set_simple_text,
    write_pipeline_step,
)

_ALBUM_YEAR_RE = re.compile(r"_(?P<year>\d{4}(?:-\d{4})?)_B", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff"}
_SEQUENCE_STEP_VERSION = "1"


@dataclass
class _PageDate:
    page: int
    sort_time: datetime
    original_date: str
    original_provenance: str
    provenance: str
    anchored: bool


def sequence_album_page_dates(archive_dir: str | Path) -> dict[str, object]:
    archive = Path(archive_dir)
    pages = pages_dir_for_album_dir(archive)
    start_year, end_year = _album_year_range(archive, pages)
    archive_by_page = _archive_images_by_page(archive)
    page_images = _page_view_images_by_page(pages)
    page_numbers = sorted(set(archive_by_page) | set(page_images))
    if not page_numbers:
        return {"sidecars_written": 0, "warnings": []}

    warnings: list[str] = []
    page_dates = _sequence_page_dates(page_numbers, page_images, start_year, end_year, warnings)
    written = 0
    for page_date in page_dates:
        if page_date.page in page_images:
            _write_sequence_sidecar(page_images[page_date.page].with_suffix(".xmp"), page_date)
            written += 1
        for archive_image in archive_by_page.get(page_date.page, []):
            _write_sequence_sidecar(archive_image.with_suffix(".xmp"), page_date)
            written += 1
    return {"sidecars_written": written, "warnings": warnings}


def _album_year_range(archive: Path, pages: Path) -> tuple[int, int]:
    for source in (archive.name, pages.name):
        match = _ALBUM_YEAR_RE.search(source)
        if match:
            return _parse_year_range(match.group("year"))
    for image_path in sorted([*archive.glob("*"), *pages.glob("*")]):
        _, year_text, _, _ = parse_album_filename(image_path.name)
        if year_text != "Unknown":
            return _parse_year_range(year_text)
    raise ValueError(f"Album must include a year or year range for date sequencing: {archive.name}")


def _parse_year_range(year_text: str) -> tuple[int, int]:
    if "-" not in year_text:
        year = int(year_text)
        return year, year
    start_text, end_text = year_text.split("-", 1)
    return int(start_text), int(end_text)


def _archive_images_by_page(archive: Path) -> dict[int, list[Path]]:
    by_page: dict[int, list[Path]] = {}
    for image_path in sorted(path for path in archive.iterdir() if path.suffix.casefold() in _IMAGE_SUFFIXES):
        _, _, _, page_text = parse_album_filename(image_path.name)
        if page_text.isdigit():
            by_page.setdefault(int(page_text), []).append(image_path)
    return by_page


def _page_view_images_by_page(pages: Path) -> dict[int, Path]:
    if not pages.is_dir():
        return {}
    by_page: dict[int, Path] = {}
    for image_path in sorted(path for path in pages.glob("*_V.*") if path.suffix.casefold() in _IMAGE_SUFFIXES):
        _, _, _, page_text = parse_album_filename(image_path.name)
        if page_text.isdigit():
            by_page[int(page_text)] = image_path
    return by_page


def _sequence_page_dates(
    page_numbers: list[int],
    page_images: dict[int, Path],
    start_year: int,
    end_year: int,
    warnings: list[str],
) -> list[_PageDate]:
    anchors = {
        page: _read_original_page_date(page_images[page].with_suffix(".xmp"))
        for page in page_numbers
        if page in page_images
    }
    anchors = {page: date_text for page, date_text in anchors.items() if date_text}
    ordered_anchors = _monotonic_anchors(anchors, warnings)
    boundaries = [(page_numbers[0] - 1, datetime(start_year, 1, 1, 12))]
    boundaries.extend(ordered_anchors)
    boundaries.append((page_numbers[-1] + 1, datetime(end_year, 12, 31, 12)))

    output: list[_PageDate] = []
    anchor_times = dict(ordered_anchors)
    for left, right in pairwise(boundaries):
        left_page, left_time = left
        right_page, right_time = right
        segment_pages = [page for page in page_numbers if left_page < page < right_page]
        if not segment_pages:
            continue
        step = (right_time - left_time) / (len(segment_pages) + 1)
        for index, page in enumerate(segment_pages, 1):
            output.append(
                _PageDate(
                    page=page,
                    sort_time=(left_time + step * index).replace(microsecond=0),
                    original_date="",
                    original_provenance="No original page date estimate was present.",
                    provenance="Viewer sort date slewed between album/page date anchors; noon/nudged time is synthetic.",
                    anchored=False,
                )
            )
    for page, original_date in anchors.items():
        output.append(
            _PageDate(
                page=page,
                sort_time=anchor_times[page],
                original_date=original_date,
                original_provenance="Original estimate read from page sidecar.",
                provenance="Viewer sort date pinned to date read from page sidecar; noon/nudged time is synthetic.",
                anchored=True,
            )
        )
    return sorted(output, key=lambda row: row.page)


def _monotonic_anchors(anchors: dict[int, str], warnings: list[str]) -> list[tuple[int, datetime]]:
    ordered: list[tuple[int, datetime]] = []
    previous: datetime | None = None
    for page, date_text in sorted(anchors.items()):
        sort_time = _date_text_to_sort_time(date_text)
        if previous is not None and sort_time <= previous:
            warnings.append(f"Page P{page:02d} has date {date_text} before an earlier page; nudged for viewer ordering.")
            sort_time = previous + timedelta(seconds=1)
        ordered.append((page, sort_time))
        previous = sort_time
    return ordered


def _read_original_page_date(sidecar: Path) -> str:
    if not sidecar.is_file():
        return ""
    try:
        root = ET.parse(sidecar).getroot()
    except ET.ParseError:
        return ""
    desc = root.find(f".//{{{RDF_NS}}}Description")
    if desc is None:
        return ""
    original = str(desc.findtext(f"{{{IMAGO_NS}}}OriginalEstimatedDate", default="") or "").strip()
    if original:
        return original
    dc_date = desc.find(f"{{{DC_NS}}}date")
    if dc_date is not None:
        item = dc_date.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")
        if item is not None and item.text:
            return item.text.strip()
    return str(desc.findtext(f"{{{XMP_NS}}}CreateDate", default="") or "").strip()


def _date_text_to_sort_time(date_text: str) -> datetime:
    text = str(date_text or "").strip()
    if len(text) == 4 and text.isdigit():
        return datetime(int(text), 7, 1, 12)
    if len(text) == 7 and text[4] == "-":
        return datetime(int(text[:4]), int(text[5:7]), 15, 12)
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.fromisoformat(f"{text}T12:00:00")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None, microsecond=0)


def _write_sequence_sidecar(sidecar: Path, page_date: _PageDate) -> None:
    tree = _read_or_create_xmp_tree(sidecar)
    desc = _get_or_create_rdf_desc(tree)
    sort_text = page_date.sort_time.replace(microsecond=0).isoformat()
    _set_seq_text(desc, f"{{{DC_NS}}}date", sort_text)
    _set_simple_text(desc, f"{{{EXIF_NS}}}DateTimeOriginal", sort_text)
    _set_simple_text(desc, f"{{{XMP_NS}}}CreateDate", sort_text)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}ViewerSortDate", sort_text)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}EstimatedDateProvenance", page_date.provenance)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OriginalEstimatedDate", page_date.original_date)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OriginalEstimatedDateProvenance", page_date.original_provenance)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}DateSequenceVersion", _SEQUENCE_STEP_VERSION)
    ET.indent(tree, space="  ")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tree.write(sidecar, encoding="utf-8", xml_declaration=True)
    write_pipeline_step(sidecar, "sequence-page-dates", extra={"version": _SEQUENCE_STEP_VERSION})
