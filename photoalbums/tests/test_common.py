import os
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

            with mock.patch(
                "common.tiff_needs_conversion", return_value=True
            ), mock.patch("common.validate_pixels", return_value=True), mock.patch(
                "common.subprocess.run"
            ) as run_mock:
                run_mock.return_value = mock.Mock()
                result = common.process_tiff_in_place(tiff_path, log_error=print)

            self.assertTrue(result)
            self.assertTrue(tiff_path.exists())

    def test_get_photo_albums_dir_env_override(self):
        with mock.patch.dict(
            "common.os.environ",
            {common.PHOTO_ALBUMS_DIR_ENV: "D:/Media/Photo Albums"},
            clear=False,
        ):
            result = common.get_photo_albums_dir()
        self.assertEqual(result, Path("D:/Media/Photo Albums"))

    def test_get_photo_albums_dir_darwin_prefers_existing_cloudstorage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = Path(tmp)
            expected = (
                fake_home
                / "Library"
                / "CloudStorage"
                / "OneDrive-Personal"
                / common.PHOTO_ALBUMS_SUBPATH
            )
            expected.mkdir(parents=True)

            with mock.patch.object(common.sys, "platform", "darwin"), mock.patch(
                "common.Path.home", return_value=fake_home
            ), mock.patch.dict("common.os.environ", {}, clear=False):
                result = common.get_photo_albums_dir()

        self.assertEqual(result, expected)

    def test_get_photo_albums_dir_windows_uses_onedrive_env_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            onedrive_root = Path(tmp) / "OneDriveCompany"
            expected = onedrive_root / common.PHOTO_ALBUMS_SUBPATH

            # _first_existing_path is mocked to None so the real OneDrive on
            # the host (if it exists) doesn't shadow the expected preferred path.
            with mock.patch.object(common.sys, "platform", "win32"), mock.patch(
                "common.Path.home", return_value=Path(tmp) / "home"
            ), mock.patch.dict(
                "common.os.environ", {"OneDrive": str(onedrive_root)}, clear=False
            ), mock.patch(
                "common._first_existing_path", return_value=None
            ):
                result = common.get_photo_albums_dir()

        self.assertEqual(result, expected)

    def test_get_imagemagick_dir_env_override(self):
        with mock.patch.dict(
            "common.os.environ",
            {common.IMAGEMAGICK_DIR_ENV: "D:/Tools/ImageMagick"},
            clear=False,
        ):
            result = common.get_imagemagick_dir()
        self.assertEqual(result, Path("D:/Tools/ImageMagick"))

    def test_configure_imagemagick_adds_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "common.get_imagemagick_dir",
            return_value=Path(tmp),
        ), mock.patch.dict("common.os.environ", {"PATH": "base"}, clear=False):
            common.configure_imagemagick()
            self.assertTrue(os.environ["PATH"].startswith(f"{tmp}{os.pathsep}"))

    def test_configure_imagemagick_skips_missing_path(self):
        with mock.patch(
            "common.get_imagemagick_dir",
            return_value=Path("Z:/path/that/does/not/exist"),
        ), mock.patch.dict("common.os.environ", {"PATH": "base"}, clear=False):
            common.configure_imagemagick()
            self.assertEqual(os.environ["PATH"], "base")

    def test_open_image_fullscreen_uses_open_on_macos(self):
        proc = object()
        with mock.patch.object(common.sys, "platform", "darwin"), mock.patch(
            "common.subprocess.Popen",
            return_value=proc,
        ) as popen_mock:
            result = common.open_image_fullscreen("/tmp/sample.jpg")

        self.assertIs(result, proc)
        popen_mock.assert_called_once_with(["open", "/tmp/sample.jpg"])


if __name__ == "__main__":
    unittest.main()
