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

    def test_linear_pair_fallback_stitches_split_page(self):
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            self.skipTest(f"cv2/numpy unavailable: {exc}")

        background = np.array([214, 206, 190], dtype=np.uint8)
        page = np.empty((420, 720, 3), dtype=np.uint8)
        page[:] = background
        for y in range(0, page.shape[0], 12):
            page[y : y + 1] = background - np.array([10, 10, 10], dtype=np.uint8)

        cv2.rectangle(page, (20, 30), (180, 180), (40, 220, 40), -1)
        cv2.rectangle(page, (250, 70), (470, 220), (200, 90, 90), -1)
        cv2.rectangle(page, (520, 220), (690, 390), (40, 40, 220), -1)
        cv2.putText(
            page,
            "CENTER",
            (270, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (24, 24, 24),
            2,
            cv2.LINE_AA,
        )

        left_scan = page[:, :440].copy()
        right_scan = page[:, 280:].copy()
        shifted_right = np.empty_like(right_scan)
        shifted_right[:] = background
        shifted_right[18:] = right_scan[:-18]

        stitched = sop._stitch_linear_pair_images([left_scan, shifted_right])

        self.assertGreater(stitched.shape[1], left_scan.shape[1])
        self.assertGreater(int(stitched[:, :140, 1].max()), 200)
        self.assertGreater(int(stitched[:, -160:, 2].max()), 200)

    def test_read_stitch_image_falls_back_to_magick(self):
        try:
            import numpy as np
        except Exception as exc:
            self.skipTest(f"numpy unavailable: {exc}")

        fake = np.zeros((10, 20, 3), dtype=np.uint8)
        with mock.patch("stitch_oversized_pages.cv2.imread", return_value=None), mock.patch(
            "stitch_oversized_pages._read_with_pillow",
            side_effect=RuntimeError("pillow failed"),
        ), mock.patch(
            "stitch_oversized_pages._read_with_magick",
            return_value=fake,
        ) as magick_mock:
            result = sop._read_stitch_image("C:/Photos/bad-extension.jpg")

        self.assertIs(result, fake)
        magick_mock.assert_called_once_with("C:/Photos/bad-extension.jpg")


if __name__ == "__main__":
    unittest.main()
