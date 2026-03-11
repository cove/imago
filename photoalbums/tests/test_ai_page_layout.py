import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_page_layout


class TestAIPageLayout(unittest.TestCase):
    def test_classify_image_kind_recognizes_page_variants(self):
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01.jpg"),
            "page_view",
        )
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_Archive") / "Family_2020_B01_P01_S01.tif"),
            "page_scan",
        )
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01_D01_01.jpg"),
            "detail",
        )

    def test_prepare_image_layout_detects_two_photos_and_trims_footer(self):
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp) / "Family_View"
            view.mkdir()
            path = view / "Family_2020_B01_P01_stitched.jpg"

            image = np.full((900, 800, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (60, 60), (360, 360), (0, 0, 0), thickness=8)
            cv2.rectangle(image, (75, 75), (345, 345), (180, 180, 180), thickness=-1)
            cv2.line(image, (90, 120), (330, 300), (40, 40, 40), thickness=6)
            cv2.circle(image, (210, 200), 55, (90, 90, 90), thickness=5)

            cv2.rectangle(image, (430, 90), (740, 390), (0, 0, 0), thickness=8)
            cv2.rectangle(image, (445, 105), (725, 375), (200, 200, 200), thickness=-1)
            cv2.line(image, (470, 140), (700, 350), (50, 50, 50), thickness=6)
            cv2.rectangle(image, (520, 170), (640, 300), (80, 80, 80), thickness=5)

            image[720:, :] = 0
            cv2.putText(
                image,
                "Creator: Audrey D. Cordell",
                (90, 790),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Stitched: 2026-03-08 14:00:00",
                (90, 845),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            self.assertTrue(cv2.imwrite(str(path), image))

            with ai_page_layout.prepare_image_layout(path, split_mode="auto") as layout:
                self.assertEqual(layout.kind, "page_view")
                self.assertTrue(layout.page_like)
                self.assertTrue(layout.footer_trimmed)
                self.assertTrue(layout.split_applied)
                self.assertFalse(layout.fallback_used)
                self.assertEqual(len(layout.subphotos), 2)
                self.assertLess(layout.content_bounds.height, image.shape[0])
                self.assertLess(layout.subphotos[0].bounds.x, layout.subphotos[1].bounds.x)
                self.assertTrue(layout.content_path.exists())
                self.assertTrue(layout.subphotos[0].path.exists())
                self.assertTrue(layout.subphotos[1].path.exists())


    def test_prepare_image_layout_handles_grayscale_tif(self):
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            archive.mkdir()
            path = archive / "Family_1907-1946_B01_P04_S02.tif"

            gray2d = np.full((200, 200), 180, dtype=np.uint8)
            cv2.imwrite(str(path), gray2d)

            with ai_page_layout.prepare_image_layout(path, split_mode="none") as layout:
                self.assertEqual(layout.kind, "page_scan")

    def test_load_image_bgr_handles_single_channel_3d(self):
        """Regression: cv2.imread with IMREAD_UNCHANGED can return (H,W,1) for some TIFs."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        from unittest.mock import patch

        img_1ch = np.full((200, 200, 1), 180, dtype=np.uint8)
        with patch("cv2.imread", return_value=img_1ch):
            result = ai_page_layout._load_image_bgr(Path("fake.tif"))
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
