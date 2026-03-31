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
    def test_normalize_page_split_mode_defaults_to_off(self):
        self.assertEqual(ai_page_layout.normalize_page_split_mode(None), "off")
        self.assertEqual(ai_page_layout.normalize_page_split_mode("invalid"), "off")

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
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01_D01-01_V.jpg"),
            "derived",
        )
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01_V.jpg"),
            "page_view",
        )
        # Legacy names still recognised during transition
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01_VC.jpg"),
            "page_view",
        )
        self.assertEqual(
            ai_page_layout.classify_image_kind(Path("Family_View") / "Family_2020_B01_P01_stitched.jpg"),
            "page_view",
        )

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
