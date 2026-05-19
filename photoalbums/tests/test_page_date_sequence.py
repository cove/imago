from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import cast

from photoalbums.lib import xmp_sidecar

IMAGO_NS = "https://imago.local/ns/1.0/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-image")


def _write_page_sidecar(path: Path, *, dc_date: str = "") -> None:
    xmp_sidecar.write_xmp_sidecar(
        path,
        person_names=[],
        subjects=[],
        description="",
        ocr_text="",
        dc_date=dc_date,
        date_time_original=dc_date,
        create_date=dc_date,
        replace_dc_date=bool(dc_date),
    )


def _read_text(sidecar: Path, namespace: str, name: str) -> str:
    root = ET.parse(sidecar).getroot()
    return str(root.findtext(f".//{{{namespace}}}{name}", default="") or "").strip()


def _read_sort_time(sidecar: Path) -> datetime:
    value = _read_text(sidecar, XMP_NS, "CreateDate")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class TestPageDateSequence(unittest.TestCase):
    def test_sequence_album_page_dates_pins_explicit_page_date_and_slews_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1970-1974_B01_Archive"
            pages = root / "Family_1970-1974_B01_Pages"
            archive.mkdir()
            pages.mkdir()

            for page in ("01", "02", "03"):
                _touch(archive / f"Family_1970-1974_B01_P{page}_S01.tif")
                _touch(pages / f"Family_1970-1974_B01_P{page}_V.jpg")

            _write_page_sidecar(pages / "Family_1970-1974_B01_P02_V.xmp", dc_date="1972-06-03")
            _write_page_sidecar(pages / "Family_1970-1974_B01_P03_V.xmp")

            from photoalbums.lib.page_date_sequence import sequence_album_page_dates

            result = sequence_album_page_dates(archive)

            page_01 = pages / "Family_1970-1974_B01_P01_V.xmp"
            page_02 = pages / "Family_1970-1974_B01_P02_V.xmp"
            page_03 = pages / "Family_1970-1974_B01_P03_V.xmp"
            archive_01 = archive / "Family_1970-1974_B01_P01_S01.xmp"

            self.assertTrue(page_01.is_file())
            self.assertTrue(archive_01.is_file())
            self.assertLess(_read_sort_time(page_01), _read_sort_time(page_02))
            self.assertLess(_read_sort_time(page_02), _read_sort_time(page_03))
            self.assertEqual(_read_sort_time(page_02).date().isoformat(), "1972-06-03")
            self.assertEqual(_read_sort_time(page_02).hour, 12)
            self.assertEqual(_read_text(page_02, IMAGO_NS, "OriginalEstimatedDate"), "1972-06-03")
            self.assertIn("page", _read_text(page_02, IMAGO_NS, "OriginalEstimatedDateProvenance").casefold())
            self.assertIn("page", _read_text(page_02, IMAGO_NS, "EstimatedDateProvenance").casefold())
            self.assertIn("slew", _read_text(page_03, IMAGO_NS, "EstimatedDateProvenance").casefold())
            self.assertGreaterEqual(cast(int, result["sidecars_written"]), 4)
            self.assertEqual(result["warnings"], [])

    def test_sequence_album_page_dates_errors_when_album_has_no_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_B01_Archive"
            archive.mkdir()
            _touch(archive / "Family_B01_P01_S01.tif")

            from photoalbums.lib.page_date_sequence import sequence_album_page_dates

            with self.assertRaisesRegex(ValueError, "year"):
                sequence_album_page_dates(archive)
