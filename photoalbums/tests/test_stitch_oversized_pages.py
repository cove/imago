import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stitch_oversized_pages as sop


class TestStitchOversizedPages(unittest.TestCase):
    def test_build_scans_text(self):
        self.assertEqual(sop.build_scans_text([1, 2, 10]), "S01 S02 S10")

    def test_build_source_filenames_text(self):
        self.assertEqual(
            sop.build_source_filenames_text(
                [
                    "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S01.tif",
                    "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S02.tif",
                ]
            ),
            "EU_1973_B02_P05_S01.tif; EU_1973_B02_P05_S02.tif",
        )

    def test_build_scan_header(self):
        header = sop.build_scan_header("EU", "1973", "02", 5, [1, 2])
        self.assertEqual(
            header,
            "EU (1973) - Book 02, Page 05, Scans S01 S02",
        )

    def test_extract_scan_numbers(self):
        files = [
            "EU_1973_B02_P05_S01.tif",
            "EU_1973_B02_P05_S02.tif",
            "no_scan_here.tif",
        ]
        self.assertEqual(sop.extract_scan_numbers(files), [1, 2])

    def test_build_derived_output_name_known(self):
        name = "EU_1973_B02_P05_D01_02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_B02_P05_D01_02.jpg",
        )

    def test_build_derived_output_name_unknown(self):
        name = "EU_1973_Custom_D01_02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_Custom_D01_02_D01_02.jpg",
        )

    def test_get_view_dirname(self):
        base = Path("C:/Photos/EU_1973_B02_Archive")
        view = sop.get_view_dirname(base)
        self.assertEqual(Path(view), Path("C:/Photos/EU_1973_B02_View"))

    def test_stitch_writes_metadata_without_burning_footer(self):
        files = [
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S01.tif",
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S02.tif",
        ]
        fake_result = mock.Mock()
        fake_result.size = 1

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "stitch_oversized_pages._require_stitcher"
        ), mock.patch(
            "stitch_oversized_pages._require_image_modules"
        ), mock.patch(
            "stitch_oversized_pages.output_is_valid", return_value=False
        ), mock.patch(
            "stitch_oversized_pages.AffineStitcher"
        ) as stitcher_mock, mock.patch(
            "stitch_oversized_pages.add_bottom_header"
        ) as footer_mock, mock.patch(
            "stitch_oversized_pages.write_jpeg"
        ) as write_mock:
            stitcher_mock.return_value.stitch.return_value = fake_result

            sop.stitch(files, tmp)

        footer_mock.assert_not_called()
        write_mock.assert_called_once_with(
            fake_result,
            str(Path(tmp) / "EU_1973_B02_P05_stitched.jpg"),
            "EU (1973) - Book 02, Page 05, Scans S01 S02",
            extra_tags={
                "XMP-dc:Source": "EU_1973_B02_P05_S01.tif; EU_1973_B02_P05_S02.tif"
            },
        )


if __name__ == "__main__":
    unittest.main()
