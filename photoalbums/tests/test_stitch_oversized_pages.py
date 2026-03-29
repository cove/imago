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
    def test_build_derived_output_name_known(self):
        name = "EU_1973_B02_P05_D01-02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_B02_P05_D01-02_V.jpg",
        )

    def test_build_derived_output_name_unknown(self):
        name = "EU_1973_Custom_D01-02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_Custom_D01-02_D01-02_V.jpg",
        )

    def test_build_derived_output_name_legacy_media(self):
        name = "Family_1907-1946_B01_P28_D01_03.mp4"
        self.assertEqual(
            sop.build_derived_output_name(name, output_suffix=".mp4"),
            "Family_1907-1946_B01_P28_D01-03_V.mp4",
        )

    def test_get_view_dirname(self):
        base = Path("C:/Photos/EU_1973_B02_Archive")
        view = sop.get_view_dirname(base)
        self.assertEqual(Path(view), Path("C:/Photos/EU_1973_B02_View"))

    def test_stitch_writes_jpeg(self):
        files = [
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S01.tif",
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S02.tif",
        ]
        fake_result = mock.Mock()
        fake_result.size = 1

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("stitch_oversized_pages._require_stitcher"),
            mock.patch("stitch_oversized_pages._require_image_modules"),
            mock.patch("stitch_oversized_pages.AffineStitcher") as stitcher_mock,
            mock.patch("stitch_oversized_pages.write_jpeg") as write_mock,
        ):
            stitcher_mock.return_value.stitch.return_value = fake_result

            sop.stitch(files, tmp)

        write_mock.assert_called_once_with(
            fake_result,
            str(Path(tmp) / "EU_1973_B02_P05_V.jpg"),
        )

    def test_tif_to_jpg_skips_existing_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "EU_1973_B02_P05_V.jpg"
            out.write_bytes(b"x" * 1024)
            with (
                mock.patch("stitch_oversized_pages._require_image_modules"),
                mock.patch("stitch_oversized_pages._read_stitch_image") as read_mock,
                mock.patch("stitch_oversized_pages.write_jpeg") as write_mock,
            ):
                wrote = sop.tif_to_jpg("C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S01.tif", tmp)

        self.assertFalse(wrote)
        read_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_derived_to_jpg_skips_existing_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "EU_1973_B02_P05_D01-02_V.jpg"
            out.write_bytes(b"x" * 1024)
            with (
                mock.patch("stitch_oversized_pages._require_image_modules"),
                mock.patch("stitch_oversized_pages._read_stitch_image") as read_mock,
                mock.patch("stitch_oversized_pages.write_jpeg") as write_mock,
            ):
                wrote = sop.derived_to_jpg("C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_D01-02.tif", tmp)

        self.assertFalse(wrote)
        read_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_colorized_to_jpg_skips_existing_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Family_1907-1946_B01_P05_D01-02_C.jpg"
            out.write_bytes(b"x" * 1024)
            with (
                mock.patch("stitch_oversized_pages._require_image_modules"),
                mock.patch("stitch_oversized_pages._read_stitch_image") as read_mock,
                mock.patch("stitch_oversized_pages.write_jpeg") as write_mock,
            ):
                wrote = sop.colorized_to_jpg(
                    "C:/Photos/Family_1907-1946_B01_Archive/Family_1907-1946_B01_P05_D01-02_C.png",
                    tmp,
                )

        self.assertFalse(wrote)
        read_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_copy_derived_media_copies_and_renames(self):
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            src = Path(src_tmp) / "Family_1907-1946_B01_P28_D01_03.mp4"
            src.write_bytes(b"media")

            wrote = sop.copy_derived_media(str(src), dst_tmp)

            self.assertTrue(wrote)
            out = Path(dst_tmp) / "Family_1907-1946_B01_P28_D01-03_V.mp4"
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), b"media")

    def test_copy_derived_media_skips_existing_output(self):
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            src = Path(src_tmp) / "Family_1907-1946_B01_P28_D01_03.pdf"
            src.write_bytes(b"source")
            out = Path(dst_tmp) / "Family_1907-1946_B01_P28_D01-03_V.pdf"
            out.write_bytes(b"existing")

            wrote = sop.copy_derived_media(str(src), dst_tmp)

            self.assertFalse(wrote)
            self.assertEqual(out.read_bytes(), b"existing")

    def test_stitch_skips_existing_valid_output(self):
        files = [
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S01.tif",
            "C:/Photos/EU_1973_B02_Archive/EU_1973_B02_P05_S02.tif",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "EU_1973_B02_P05_V.jpg"
            out.write_bytes(b"x" * 1024)
            with (
                mock.patch("stitch_oversized_pages._require_stitcher"),
                mock.patch("stitch_oversized_pages._require_image_modules"),
                mock.patch("stitch_oversized_pages.build_stitched_image") as build_mock,
                mock.patch("stitch_oversized_pages.write_jpeg") as write_mock,
            ):
                wrote = sop.stitch(files, tmp)

        self.assertFalse(wrote)
        build_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_linear_pair_fallback_stitches_split_page(self):
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            self.skipTest(f"cv2/numpy unavailable: {exc}")

        background = np.array([214, 206, 190], dtype=np.uint8)
        page = np.empty((240, 360, 3), dtype=np.uint8)
        page[:] = background
        for y in range(0, page.shape[0], 8):
            page[y : y + 1] = background - np.array([10, 10, 10], dtype=np.uint8)

        cv2.rectangle(page, (14, 22), (90, 102), (40, 220, 40), -1)
        cv2.rectangle(page, (120, 36), (245, 128), (200, 90, 90), -1)
        cv2.rectangle(page, (248, 132), (342, 226), (40, 40, 220), -1)
        cv2.putText(
            page,
            "CENTER",
            (124, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (24, 24, 24),
            2,
            cv2.LINE_AA,
        )

        left_scan = page[:, :220].copy()
        right_scan = page[:, 140:].copy()
        shifted_right = np.empty_like(right_scan)
        shifted_right[:] = background
        shifted_right[18:] = right_scan[:-18]

        with (
            mock.patch.object(sop, "LINEAR_FALLBACK_TARGET_WIDTH", 240),
            mock.patch.object(sop, "LINEAR_FALLBACK_OVERLAP_STEP", 24),
            mock.patch.object(sop, "LINEAR_FALLBACK_VERTICAL_STEP", 8),
            mock.patch.object(sop, "LINEAR_FALLBACK_REFINE_OVERLAP_RADIUS", 6),
            mock.patch.object(sop, "LINEAR_FALLBACK_REFINE_VERTICAL_RADIUS", 2),
        ):
            stitched = sop._stitch_linear_pair_images([left_scan, shifted_right])

        self.assertGreater(stitched.shape[1], left_scan.shape[1])
        self.assertGreater(int(stitched[:, :140, 1].max()), 200)
        self.assertGreater(int(stitched[:, -160:, 2].max()), 200)

    def test_write_jpeg_raises_on_imwrite_failure(self):
        try:
            import numpy as np
        except Exception as exc:
            self.skipTest(f"numpy unavailable: {exc}")

        fake_image = mock.Mock()
        with (
            mock.patch("stitch_oversized_pages._require_image_modules"),
            mock.patch("stitch_oversized_pages.cv2") as cv2_mock,
            tempfile.TemporaryDirectory() as tmp,
        ):
            cv2_mock.imwrite.return_value = False
            cv2_mock.IMWRITE_JPEG_QUALITY = 1
            out = Path(tmp) / "out.jpg"
            with self.assertRaises(RuntimeError, msg="write_jpeg must raise when cv2.imwrite returns False"):
                sop.write_jpeg(fake_image, out)

    def test_read_stitch_image_falls_back_to_magick(self):
        try:
            import numpy as np
        except Exception as exc:
            self.skipTest(f"numpy unavailable: {exc}")

        fake = np.zeros((10, 20, 3), dtype=np.uint8)
        with (
            mock.patch("stitch_oversized_pages.cv2.imread", return_value=None),
            mock.patch(
                "stitch_oversized_pages._read_with_pillow",
                side_effect=RuntimeError("pillow failed"),
            ),
            mock.patch(
                "stitch_oversized_pages._read_with_magick",
                return_value=fake,
            ) as magick_mock,
        ):
            result = sop._read_stitch_image("C:/Photos/bad-extension.jpg")

        self.assertIs(result, fake)
        magick_mock.assert_called_once_with("C:/Photos/bad-extension.jpg")


if __name__ == "__main__":
    unittest.main()
