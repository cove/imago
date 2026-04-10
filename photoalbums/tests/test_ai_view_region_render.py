"""Tests for ai_view_region_render debug rendering."""
from __future__ import annotations

import io
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


def _make_jpeg(width: int, height: int, path: Path) -> None:
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 120, 140))
    img.save(str(path), format="JPEG")


def _is_valid_jpeg(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


class TestRenderRegionsDebug(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL not available")

    def test_returns_valid_jpeg_bytes(self):
        from photoalbums.lib.ai_view_region_render import render_regions_debug

        regions = [
            {"index": 0, "x": 0, "y": 0, "width": 400, "height": 600, "caption": "Left"},
            {"index": 1, "x": 400, "y": 0, "width": 400, "height": 600, "caption": "Right"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            out_path = Path(tmp) / "_debug" / "test_V_regions_debug.jpg"
            _make_jpeg(800, 600, img_path)

            result = render_regions_debug(img_path, regions, out_path)

            self.assertIsInstance(result, bytes)
            self.assertTrue(len(result) > 0)
            self.assertTrue(_is_valid_jpeg(result))
            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.read_bytes(), result)

    def test_empty_regions_still_returns_valid_jpeg(self):
        from photoalbums.lib.ai_view_region_render import render_regions_debug

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            out_path = Path(tmp) / "out.jpg"
            _make_jpeg(400, 300, img_path)

            result = render_regions_debug(img_path, [], out_path)

            self.assertTrue(_is_valid_jpeg(result))

    def test_downscales_large_image(self):
        from PIL import Image
        from photoalbums.lib.ai_view_region_render import render_regions_debug, _MAX_EDGE

        with tempfile.TemporaryDirectory() as tmp:
            # Image larger than _MAX_EDGE on longest edge
            big_w, big_h = _MAX_EDGE * 2, _MAX_EDGE + 100
            img_path = Path(tmp) / "big_V.jpg"
            out_path = Path(tmp) / "big_debug.jpg"
            _make_jpeg(big_w, big_h, img_path)

            result = render_regions_debug(img_path, [], out_path)

            out_img = Image.open(io.BytesIO(result))
            out_w, out_h = out_img.size
            self.assertLessEqual(max(out_w, out_h), _MAX_EDGE)

    def test_small_image_not_upscaled(self):
        from PIL import Image
        from photoalbums.lib.ai_view_region_render import render_regions_debug, _MAX_EDGE

        with tempfile.TemporaryDirectory() as tmp:
            small_w, small_h = 400, 300
            img_path = Path(tmp) / "small_V.jpg"
            out_path = Path(tmp) / "small_debug.jpg"
            _make_jpeg(small_w, small_h, img_path)

            result = render_regions_debug(img_path, [], out_path)

            out_img = Image.open(io.BytesIO(result))
            out_w, out_h = out_img.size
            self.assertEqual(out_w, small_w)
            self.assertEqual(out_h, small_h)


if __name__ == "__main__":
    unittest.main()
