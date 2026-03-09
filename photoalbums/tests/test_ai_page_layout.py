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


if __name__ == "__main__":
    unittest.main()
