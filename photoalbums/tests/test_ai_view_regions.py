"""Tests for ai_view_regions detection, coordinate conversion, and caption association."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.ai_view_regions import (
    RegionResult,
    RegionWithCaption,
    associate_captions,
    pixel_to_mwgrs,
    _parse_region_response,
    _has_xmp_regions,
    _read_regions_from_xmp,
)
from photoalbums.lib.xmp_sidecar import write_region_list, read_region_list


# ---------------------------------------------------------------------------
# pixel_to_mwgrs
# ---------------------------------------------------------------------------

class TestPixelToMwgrs(unittest.TestCase):
    def test_centre_point_computed_correctly(self):
        cx, cy, nw, nh = pixel_to_mwgrs(100, 200, 400, 300, 1000, 1000)
        self.assertAlmostEqual(cx, 0.3)   # (100 + 200) / 1000
        self.assertAlmostEqual(cy, 0.35)  # (200 + 150) / 1000
        self.assertAlmostEqual(nw, 0.4)
        self.assertAlmostEqual(nh, 0.3)

    def test_full_image_region(self):
        cx, cy, nw, nh = pixel_to_mwgrs(0, 0, 800, 600, 800, 600)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(nw, 1.0)
        self.assertAlmostEqual(nh, 1.0)

    def test_corner_region(self):
        cx, cy, nw, nh = pixel_to_mwgrs(0, 0, 100, 100, 1000, 1000)
        self.assertAlmostEqual(cx, 0.05)
        self.assertAlmostEqual(cy, 0.05)
        self.assertAlmostEqual(nw, 0.1)
        self.assertAlmostEqual(nh, 0.1)


# ---------------------------------------------------------------------------
# _parse_region_response
# ---------------------------------------------------------------------------

class TestParseRegionResponse(unittest.TestCase):
    def _make_response(self, regions):
        return json.dumps({"regions": regions})

    def test_valid_response_parsed(self):
        # Model returns normalised 0-1 coords; we convert to pixel using img dims
        payload = self._make_response([
            {"index": 0, "x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0, "confidence": 0.9, "caption_hint": "Beach"},
            {"index": 1, "x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0, "confidence": 0.85, "caption_hint": ""},
        ])
        results = _parse_region_response(payload, img_w=1000, img_h=800)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].x, 0)
        self.assertEqual(results[0].width, 500)
        self.assertAlmostEqual(results[0].confidence, 0.9)
        self.assertEqual(results[0].caption_hint, "Beach")
        self.assertEqual(results[1].x, 500)

    def test_empty_response_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_region_response("")

    def test_malformed_json_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_region_response("not json at all")

    def test_json_embedded_in_text(self):
        payload = 'Sure! Here is the answer:\n' + json.dumps({"regions": [
            {"index": 0, "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.8, "confidence": 1.0, "caption_hint": ""}
        ]})
        results = _parse_region_response(payload, img_w=1000, img_h=1000)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].x, 100)

    def test_malformed_entry_skipped(self):
        payload = json.dumps({"regions": [
            {"index": 0, "x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5, "confidence": 0.9, "caption_hint": ""},
            {"index": 1, "x": "bad", "y": 0.0, "width": 0.5, "height": 0.5, "confidence": 0.5, "caption_hint": ""},
        ]})
        results = _parse_region_response(payload, img_w=800, img_h=600)
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# detect_regions — model call mocking
# ---------------------------------------------------------------------------

class TestDetectRegions(unittest.TestCase):
    def _mock_response(self, regions):
        return {
            "choices": [{"message": {"content": json.dumps({"regions": regions})}}]
        }

    def test_happy_path_returns_regions(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        # Model returns normalised 0-1 coords; image is 800×600 → right half starts at x=400
        mock_regions = [
            {"index": 0, "x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0, "confidence": 0.9, "caption_hint": "Left photo"},
            {"index": 1, "x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0, "confidence": 0.88, "caption_hint": "Right photo"},
        ]
        mock_resp = self._mock_response(mock_regions)

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            try:
                from PIL import Image
                img = Image.new("RGB", (800, 600), color=(128, 128, 128))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", return_value=mock_resp):
                results = detect_regions(img_path, force=True)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].caption_hint, "Left photo")
        self.assertEqual(results[1].x, 400)

    def test_retry_on_malformed_json(self):
        from photoalbums.lib.ai_view_regions import detect_regions, _MAX_RETRIES

        bad_resp = {"choices": [{"message": {"content": "not json"}}]}
        good_regions = [
            {"index": 0, "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "confidence": 1.0, "caption_hint": ""}
        ]
        good_resp = self._mock_response(good_regions)

        responses = [bad_resp, good_resp]
        call_count = 0

        def mock_post(url, payload, timeout):
            nonlocal call_count
            call_count += 1
            return responses[min(call_count - 1, len(responses) - 1)]

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            try:
                from PIL import Image
                img = Image.new("RGB", (800, 600))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                results = detect_regions(img_path, force=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(call_count, 2)

    def test_all_retries_fail_returns_empty(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            try:
                from PIL import Image
                img = Image.new("RGB", (800, 600))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with mock.patch(
                "photoalbums.lib.ai_view_regions._lmstudio_post",
                side_effect=RuntimeError("LM Studio is unreachable"),
            ):
                results = detect_regions(img_path, force=True)

        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# associate_captions
# ---------------------------------------------------------------------------

class TestAssociateCaptions(unittest.TestCase):
    def _make_regions(self, coords):
        return [
            RegionResult(index=i, x=x, y=y, width=w, height=h)
            for i, (x, y, w, h) in enumerate(coords)
        ]

    def test_unambiguous_assignment(self):
        # Two regions side-by-side: left caption -> region 0, right -> region 1
        regions = self._make_regions([(0, 0, 400, 600), (400, 0, 400, 600)])
        captions = [
            {"text": "Left caption", "x": 50, "y": 620, "w": 300, "h": 40},
            {"text": "Right caption", "x": 450, "y": 620, "w": 300, "h": 40},
        ]
        result = associate_captions(regions, captions, img_width=800)
        texts = {rwc.region.index: rwc.caption for rwc in result}
        self.assertEqual(texts[0], "Left caption")
        self.assertEqual(texts[1], "Right caption")
        self.assertFalse(any(rwc.caption_ambiguous for rwc in result))

    def test_ambiguous_broadcasts_to_all(self):
        # Caption exactly between two regions
        regions = self._make_regions([(0, 0, 400, 600), (400, 0, 400, 600)])
        captions = [
            {"text": "Shared caption", "x": 360, "y": 620, "w": 80, "h": 40},  # centre at 400, equidistant
        ]
        result = associate_captions(regions, captions, img_width=800)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(rwc.caption_ambiguous for rwc in result))
        self.assertTrue(all(rwc.caption == "Shared caption" for rwc in result))

    def test_no_captions_returns_empty_captions(self):
        regions = self._make_regions([(0, 0, 400, 600), (400, 0, 400, 600)])
        result = associate_captions(regions, [], img_width=800)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(rwc.caption == "" for rwc in result))

    def test_caption_without_position_broadcasts(self):
        regions = self._make_regions([(0, 0, 400, 600), (400, 0, 400, 600)])
        captions = [{"text": "No position caption"}]  # no x/y/w/h
        result = associate_captions(regions, captions, img_width=800)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(rwc.caption_ambiguous for rwc in result))


# ---------------------------------------------------------------------------
# write_region_list / read_region_list round-trip
# ---------------------------------------------------------------------------

class TestRegionListXmpRoundTrip(unittest.TestCase):
    def test_write_and_read_back(self):
        regions = [
            RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=600), "Left"),
            RegionWithCaption(RegionResult(index=1, x=400, y=0, width=400, height=600), "Right"),
        ]
        img_w, img_h = 800, 600
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            write_region_list(xmp_path, regions, img_w, img_h)
            self.assertTrue(xmp_path.is_file())
            content = xmp_path.read_text(encoding="utf-8")
            self.assertIn("mwg-rs:RegionList", content)
            self.assertIn("mwg-rs:Type", content)

            read_back = read_region_list(xmp_path, img_w, img_h)
            self.assertEqual(len(read_back), 2)
            # Pixel coords should round-trip within 2px
            self.assertAlmostEqual(read_back[0]["x"], 0, delta=2)
            self.assertAlmostEqual(read_back[0]["width"], 400, delta=2)
            self.assertAlmostEqual(read_back[1]["x"], 400, delta=2)
            self.assertEqual(read_back[0]["caption"], "Left")
            self.assertEqual(read_back[1]["caption"], "Right")

    def test_existing_region_list_replaced(self):
        img_w, img_h = 800, 600
        initial = [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=800, height=600), "")]
        updated = [
            RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=600), "A"),
            RegionWithCaption(RegionResult(index=1, x=400, y=0, width=400, height=600), "B"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            write_region_list(xmp_path, initial, img_w, img_h)
            write_region_list(xmp_path, updated, img_w, img_h)
            read_back = read_region_list(xmp_path, img_w, img_h)
            self.assertEqual(len(read_back), 2)
            self.assertEqual(read_back[0]["caption"], "A")

    def test_has_xmp_regions_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            self.assertFalse(_has_xmp_regions(xmp_path))
            regions = [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=100, height=100), "")]
            write_region_list(xmp_path, regions, 200, 200)
            self.assertTrue(_has_xmp_regions(xmp_path))


if __name__ == "__main__":
    unittest.main()
