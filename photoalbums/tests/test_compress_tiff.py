import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compress_tiff


class TestCompressTiff(unittest.TestCase):
    def test_convert_directory_processes_needed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Album_Archive"
            archive.mkdir()
            tiff_path = archive / "sample.tif"
            tiff_path.touch()

            with mock.patch(
                "compress_tiff.list_archive_dirs", return_value=[archive]
            ), mock.patch(
                "compress_tiff.tiff_needs_conversion", return_value=True
            ), mock.patch(
                "compress_tiff.process_tiff_in_place", return_value=True
            ) as process_mock:
                compress_tiff.convert_directory(base)

            process_mock.assert_called_once()
            args, kwargs = process_mock.call_args
            self.assertEqual(args[0], tiff_path)
            self.assertIn("log_error", kwargs)

    def test_convert_directory_skips_when_not_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Album_Archive"
            archive.mkdir()
            tiff_path = archive / "sample.tif"
            tiff_path.touch()

            with mock.patch(
                "compress_tiff.list_archive_dirs", return_value=[archive]
            ), mock.patch(
                "compress_tiff.tiff_needs_conversion", return_value=False
            ), mock.patch(
                "compress_tiff.process_tiff_in_place", return_value=True
            ) as process_mock:
                compress_tiff.convert_directory(base)

            process_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
