import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common


class TestCommon(unittest.TestCase):
    def test_derive_prefix(self):
        self.assertEqual(common.derive_prefix("Album_Archive"), "Album")
        self.assertEqual(common.derive_prefix(Path("Album_Archive")), "Album")
        self.assertEqual(common.derive_prefix("Album"), "Album")

    def test_get_next_filename_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Russia_1984_B02_Archive"
            watch_dir.mkdir()
            expected = "Russia_1984_B02_P01_S01.tif"
            self.assertEqual(common.get_next_filename(watch_dir), expected)

    def test_get_next_filename_progression(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Album_Archive"
            watch_dir.mkdir()

            (watch_dir / "Album_P01_S01.tif").touch()
            self.assertEqual(common.get_next_filename(watch_dir), "Album_P02_S01.tif")

            (watch_dir / "Album_P02_S01.tif").touch()
            self.assertEqual(common.get_next_filename(watch_dir), "Album_P02_S02.tif")

            (watch_dir / "Album_P02_S02.tif").touch()
            self.assertEqual(common.get_next_filename(watch_dir), "Album_P03_S01.tif")

    def test_list_page_scan_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp)
            for name in [
                "Album_P02_S02.tif",
                "Album_P02_S01.tif",
                "Album_P03_S01.tif",
            ]:
                (watch_dir / name).touch()

            groups = common.list_page_scan_groups(watch_dir, common.FILENAME_PATTERN)
            self.assertEqual(len(groups), 2)
            self.assertEqual(
                [Path(p).name for p in groups[0]],
                ["Album_P02_S01.tif", "Album_P02_S02.tif"],
            )
            self.assertEqual([Path(p).name for p in groups[1]], ["Album_P03_S01.tif"])

    def test_list_page_scans_for_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp)
            for name in [
                "Album_P02_S02.tif",
                "Album_P02_S01.tif",
                "Album_P03_S01.tif",
            ]:
                (watch_dir / name).touch()

            files = common.list_page_scans_for_page(watch_dir, 2)
            self.assertEqual(
                [Path(p).name for p in files],
                ["Album_P02_S01.tif", "Album_P02_S02.tif"],
            )

    def test_count_totals(self):
        import re

        file_re = re.compile(
            r"^(?P<collection>[A-Z]+)_(?P<year>\d{4})_B(?P<book>\d{2})_P(?P<page>\d{2})_S\d{2}\.tif$",
            re.IGNORECASE,
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "ALB_2001_B01_Archive"
            archive.mkdir()

            for name in [
                "ALB_2001_B01_P01_S01.tif",
                "ALB_2001_B01_P02_S01.tif",
                "ALB_2001_B01_P04_S01.tif",
            ]:
                (archive / name).touch()

            totals = common.count_totals(
                [archive],
                file_re,
                lambda name: common.parse_filename(name, file_re),
            )

            key = "ALB_2001_B01"
            self.assertEqual(totals[key]["total_pages"], 4)
            self.assertEqual(totals[key]["page_scans"][1], 1)
            self.assertEqual(totals[key]["page_scans"][2], 1)
            self.assertEqual(totals[key]["page_scans"][4], 1)

    def test_rename_with_retry(self):
        calls = []

        def fake_rename(_old, _new):
            if not calls:
                calls.append("fail")
                raise PermissionError("locked")
            calls.append("ok")

        with mock.patch("common.os.rename", side_effect=fake_rename), mock.patch(
            "common.time.sleep", return_value=None
        ):
            result = common.rename_with_retry("a", "b", attempts=2, delay=0)

        self.assertTrue(result)
        self.assertEqual(calls, ["fail", "ok"])

    def test_process_tiff_in_place_skips_when_not_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = Path(tmp) / "sample.tif"
            tiff_path.write_bytes(b"data")

            with mock.patch("common.tiff_needs_conversion", return_value=False):
                result = common.process_tiff_in_place(tiff_path)

            self.assertTrue(result)

    def test_process_tiff_in_place_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = Path(tmp) / "sample.tif"
            tiff_path.write_bytes(b"data")

            with mock.patch("common.tiff_needs_conversion", return_value=True), mock.patch(
                "common.validate_pixels", return_value=True
            ), mock.patch("common.subprocess.run") as run_mock:
                run_mock.return_value = mock.Mock()
                result = common.process_tiff_in_place(tiff_path, log_error=print)

            self.assertTrue(result)
            self.assertTrue(tiff_path.exists())


if __name__ == "__main__":
    unittest.main()
