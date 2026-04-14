"""Tests for ai_view_regions detection, coordinate conversion, and caption association."""

from __future__ import annotations

import base64
import io
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
    RegionFailure,
    RegionResult,
    RegionWithCaption,
    ValidationResult,
    _accepted_regions_debug_path,
    _build_data_url_with_size,
    associate_captions,
    pixel_to_mwgrs,
    validate_region_set,
    validate_regions_for_write,
    _build_repair_prompt,
    _failed_regions_debug_path,
    _parse_region_response,
    _has_xmp_regions,
    _read_regions_from_xmp,
    _DOCLING_SYSTEM_PROMPT,
    _DOCLING_USER_PROMPT,
    _SYSTEM_PROMPT,
    _build_user_prompt,
    _build_user_prompt_strict,
)
from photoalbums.lib.album_sets import resolve_archive_set
from photoalbums.lib.prompt_debug import PromptDebugSession
from photoalbums.lib.xmp_sidecar import read_pipeline_step, write_pipeline_step, write_region_list, read_region_list


# ---------------------------------------------------------------------------
# pixel_to_mwgrs
# ---------------------------------------------------------------------------


class TestPixelToMwgrs(unittest.TestCase):
    def test_centre_point_computed_correctly(self):
        cx, cy, nw, nh = pixel_to_mwgrs(100, 200, 400, 300, 1000, 1000)
        self.assertAlmostEqual(cx, 0.3)  # (100 + 200) / 1000
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
    def _make_response(self, boxes):
        """Return a JSON array of box_2d items. boxes is a list of [ymin, xmin, ymax, xmax]."""
        return json.dumps([{"box_2d": b, "label": "photograph"} for b in boxes])

    def test_valid_response_parsed(self):
        # box_2d [ymin, xmin, ymax, xmax] in 0–1000
        # Left half: xmin=0, xmax=500 → x=0, w=500 on img_w=1000
        # Right half: xmin=500, xmax=1000 → x=500, w=500
        payload = self._make_response([[0, 0, 1000, 500], [0, 500, 1000, 1000]])
        results = _parse_region_response(payload, img_w=1000, img_h=800)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].x, 0)
        self.assertEqual(results[0].width, 500)
        self.assertEqual(results[0].height, 800)
        self.assertEqual(results[1].x, 500)

    def test_empty_response_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_region_response("")

    def test_malformed_json_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_region_response("not json at all")

    def test_json_array_embedded_in_text(self):
        payload = "Sure! Here is the answer:\n" + json.dumps(
            [{"box_2d": [100, 100, 900, 600], "label": "photograph"}]
        )
        results = _parse_region_response(payload, img_w=1000, img_h=1000)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].x, 100)

    def test_malformed_entry_skipped(self):
        payload = json.dumps(
            [
                {"box_2d": [0, 0, 500, 500], "label": "photograph"},
                {"box_2d": "bad", "label": "photograph"},
            ]
        )
        results = _parse_region_response(payload, img_w=800, img_h=600)
        self.assertEqual(len(results), 1)

    def test_degenerate_box_skipped(self):
        # ymin >= ymax → degenerate
        payload = self._make_response([[500, 0, 500, 500], [0, 0, 1000, 1000]])
        results = _parse_region_response(payload, img_w=100, img_h=100)
        self.assertEqual(len(results), 1)

    def test_coordinate_conversion(self):
        # box_2d [250, 100, 750, 600] on 1000×500 image
        # x = 100/1000*1000 = 100, y = 250/1000*500 = 125
        # w = (600-100)/1000*1000 = 500, h = (750-250)/1000*500 = 250
        payload = self._make_response([[250, 100, 750, 600]])
        results = _parse_region_response(payload, img_w=1000, img_h=500)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].x, 100)
        self.assertEqual(results[0].y, 125)
        self.assertEqual(results[0].width, 500)
        self.assertEqual(results[0].height, 250)


class TestViewRegionPrompts(unittest.TestCase):
    def test_prompts_describe_physical_print_boundaries_and_separators(self):
        user_prompt = _build_user_prompt(800, 600)
        strict_prompt = _build_user_prompt_strict(800, 600)
        for prompt in (_SYSTEM_PROMPT, user_prompt, strict_prompt):
            self.assertIn("photograph", prompt)
            self.assertIn("album", prompt)
            self.assertIn("box_2d", prompt)
        # System prompt establishes the engine persona
        self.assertIn("Vision-Coordinate Engine", _SYSTEM_PROMPT)
        # User prompts embed image dimensions
        self.assertIn("800", user_prompt)
        self.assertIn("600", user_prompt)
        self.assertIn("800", strict_prompt)
        self.assertIn("600", strict_prompt)


class TestBuildDataUrlWithSize(unittest.TestCase):
    def test_full_size_jpeg_is_sent_without_resize_or_reencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "test_V.jpg"
            try:
                from PIL import Image

                img = Image.new("RGB", (3000, 1800), color=(128, 128, 128))
                img.save(str(image_path), format="JPEG", quality=93)
            except ImportError:
                self.skipTest("PIL not available")

            original_bytes = image_path.read_bytes()
            data_url, width, height = _build_data_url_with_size(image_path, 0)

        self.assertEqual((width, height), (3000, 1800))
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        encoded = data_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), original_bytes)


# ---------------------------------------------------------------------------
# detect_regions — model call mocking
# ---------------------------------------------------------------------------


class TestDetectRegions(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch("photoalbums.lib.ai_view_regions.default_view_region_models", return_value=["gemma-3"]),
            mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="gemma-3"),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()

    def _mock_response(self, boxes):
        """boxes is a list of [ymin, xmin, ymax, xmax] in 0–1000."""
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [{"box_2d": b, "label": "photograph"} for b in boxes]
                        )
                    }
                }
            ]
        }

    def test_happy_path_returns_regions(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        # Image is 800×600; left half xmax=500 → x=0,w=400; right half xmin=500 → x=400,w=400
        mock_resp = self._mock_response([[0, 0, 1000, 500], [0, 500, 1000, 1000]])

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
        self.assertEqual(results[1].x, 400)

    def test_retry_on_malformed_json(self):
        from photoalbums.lib.ai_view_regions import detect_regions, _MAX_RETRIES

        bad_resp = {"choices": [{"message": {"content": "not json"}}]}
        good_resp = self._mock_response([[0, 0, 1000, 500], [0, 500, 1000, 1000]])

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

        self.assertEqual(len(results), 2)
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

    def test_prompt_includes_album_context_page_caption_and_people_roster(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        captured_payload: dict[str, object] = {}

        def mock_post(url, payload, timeout):
            captured_payload["payload"] = payload
            return self._mock_response([])

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            try:
                from PIL import Image

                img = Image.new("RGB", (800, 600), color=(128, 128, 128))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                detect_regions(
                    img_path,
                    force=True,
                    album_context="Egypt 1975, book 00, page 26",
                    page_caption="audrey-leslie at the Sphinx",
                    people_roster={"audrey": "Audrey Cordell", "leslie": "Leslie Cordell"},
                )

        payload = captured_payload["payload"]
        assert isinstance(payload, dict)
        messages = payload["messages"]
        self.assertIsInstance(messages, list)
        user_message = messages[1]
        self.assertEqual(user_message["role"], "user")
        text_part = user_message["content"][0]["text"]
        self.assertIn("Album context: Egypt 1975, book 00, page 26.", text_part)
        self.assertIn("Page caption context: audrey-leslie at the Sphinx.", text_part)
        self.assertIn("audrey -> Audrey Cordell", text_part)
        self.assertIn("leslie -> Leslie Cordell", text_part)
        self.assertIn("box_2d", text_part)
        self.assertNotIn("The full image is", text_part)

    def test_docling_call_uses_rest_chat_and_uploads_image(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        fake_lms = mock.Mock()
        fake_model = mock.Mock()
        fake_chat = mock.Mock()
        fake_prediction = mock.Mock()
        fake_prediction.text = """<loc_0><loc_0><loc_500><loc_500></picture>
<picture><loc_55><loc_5><loc_193><loc_155></picture>
<picture><loc_56><loc_187><loc_173><loc_400></picture>
<picture><loc_202><loc_5><loc_342><loc_155></picture>
<picture><loc_181><loc_190><loc_368><loc_367></picture>
<picture><loc_374><loc_189><loc_491><loc_400></picture>
<picture><loc_427><loc_415><loc_440><loc_435></picture>"""
        fake_lms.prepare_image.return_value = "image-handle"
        fake_lms.llm.return_value = fake_model
        fake_lms.Chat.return_value = fake_chat
        fake_model.respond.return_value = fake_prediction

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            try:
                from PIL import Image

                img = Image.new("RGB", (3000, 1800), color=(128, 128, 128))
                img.save(str(img_path), format="JPEG")
                original_bytes = img_path.read_bytes()
            except ImportError:
                self.skipTest("PIL not available")

            with (
                mock.patch.dict(sys.modules, {"lmstudio": fake_lms}),
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_models", return_value=["docling"]),
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="docling"),
            ):
                results = detect_regions(img_path, force=True, skip_validation=True)

        fake_lms.prepare_image.assert_called_once()
        sent_bytes = fake_lms.prepare_image.call_args.args[0]
        self.assertIsInstance(sent_bytes, (bytes, bytearray))
        self.assertEqual(sent_bytes, original_bytes)
        from PIL import Image

        with Image.open(io.BytesIO(sent_bytes)) as decoded_img:
            self.assertEqual(decoded_img.size, (3000, 1800))
        fake_lms.llm.assert_called_once_with("docling")
        fake_chat.add_system_prompt.assert_called_once_with(_DOCLING_SYSTEM_PROMPT)
        fake_chat.add_user_message.assert_called_once_with(_DOCLING_USER_PROMPT, images=["image-handle"])
        fake_model.respond.assert_called_once_with(fake_chat)
        self.assertEqual(len(results), 6)

    def test_prompt_debug_records_request_response_metadata(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        prompt_debug = PromptDebugSession("Egypt_1975_B00_P26_V.jpg")

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            try:
                from PIL import Image

                img = Image.new("RGB", (800, 600), color=(128, 128, 128))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", return_value=self._mock_response([])):
                detect_regions(img_path, force=True, prompt_debug=prompt_debug)

        artifact = prompt_debug.to_artifact()
        self.assertEqual(artifact["kind"], "photoalbums_prompts")
        self.assertEqual(artifact["step_count"], 1)
        step = artifact["steps"][0]
        self.assertEqual(step["step"], "view_regions")
        self.assertEqual(step["engine"], "lmstudio")
        self.assertTrue(step["response"])
        self.assertEqual(step["metadata"]["returned_region_count"], 0)
        self.assertEqual(step["metadata"]["attempt_number"], 1)
        self.assertFalse(step["metadata"]["strict_prompt"])


class TestValidateRegionsForWrite(unittest.TestCase):
    def test_zero_area_regions_are_rejected_but_small_regions_are_preserved(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=0, height=100, confidence=0.9),
            RegionResult(index=1, x=10, y=10, width=5, height=5, confidence=0.9),
            RegionResult(index=2, x=20, y=20, width=80, height=60, confidence=0.9),
        ]
        validated = validate_regions_for_write(regions, img_w=200, img_h=100)
        self.assertEqual([region.index for region in validated], [1, 2])

    def test_heavy_overlap_keeps_highest_ranked_region(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=100, height=100, confidence=0.6),
            RegionResult(index=1, x=2, y=2, width=98, height=98, confidence=0.95),
            RegionResult(index=2, x=120, y=0, width=80, height=100, confidence=0.8),
        ]
        validated = validate_regions_for_write(regions, img_w=200, img_h=100)
        self.assertEqual([region.index for region in validated], [1, 2])


# ---------------------------------------------------------------------------
# validate_region_set — structured failure payloads
# ---------------------------------------------------------------------------


class TestValidateRegionSet(unittest.TestCase):
    def test_valid_set_returns_empty_failures(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=90, height=90, confidence=0.9),
            RegionResult(index=1, x=100, y=0, width=90, height=90, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=100)
        self.assertTrue(result.valid)
        self.assertEqual(result.failures, [])
        self.assertEqual([r.index for r in result.kept], [0, 1])

    def test_zero_area_region_produces_hard_failure(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=0, height=50, confidence=0.9),
            RegionResult(index=1, x=0, y=0, width=120, height=200, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=200)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.failures), 1)
        failure = result.failures[0]
        self.assertEqual(failure.region_index, 0)
        self.assertEqual(failure.reason, "zero_area")
        self.assertEqual(failure.severity, "hard")
        self.assertEqual([r.index for r in result.kept], [1])

    def test_full_page_region_produces_hard_failure(self):
        # Region covers 92% of the image
        regions = [
            RegionResult(index=0, x=0, y=0, width=200, height=184, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=200)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].reason, "full_page")
        self.assertEqual(result.failures[0].severity, "hard")
        self.assertEqual(result.kept, [])

    def test_insufficient_page_coverage_produces_hard_failure(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=98, height=100, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=100)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].reason, "insufficient_page_coverage")
        self.assertEqual(result.failures[0].severity, "hard")
        self.assertEqual(result.failures[0].page_fraction, 0.49)
        self.assertEqual([r.index for r in result.kept], [0])

    def test_exactly_50_percent_page_coverage_is_kept(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=100, height=100, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=100)
        self.assertTrue(result.valid)
        self.assertEqual(result.failures, [])
        self.assertEqual([r.index for r in result.kept], [0])

    def test_overlap_failure_records_pairwise_info(self):
        # index=0 has lower confidence; index=1 nearly contains it
        regions = [
            RegionResult(index=0, x=0, y=0, width=100, height=100, confidence=0.6),
            RegionResult(index=1, x=2, y=2, width=98, height=98, confidence=0.95),
            RegionResult(index=2, x=120, y=0, width=80, height=100, confidence=0.8),
        ]
        result = validate_region_set(regions, img_w=220, img_h=100)
        self.assertFalse(result.valid)
        # Only the lower-confidence duplicate is rejected
        overlap_failures = [f for f in result.failures if f.reason == "overlap"]
        self.assertEqual(len(overlap_failures), 1)
        f = overlap_failures[0]
        self.assertEqual(f.region_index, 0)
        self.assertEqual(f.overlap_with, 1)
        self.assertIsNotNone(f.overlap_fraction)
        self.assertGreaterEqual(f.overlap_fraction, 0.05)
        # Non-overlapping region is kept
        self.assertIn(2, [r.index for r in result.kept])

    def test_moderate_overlap_is_rejected(self):
        regions = [
            RegionResult(index=1, x=308, y=1195, width=2024, height=650, confidence=1.0),
            RegionResult(index=2, x=303, y=1678, width=2024, height=652, confidence=1.0),
            RegionResult(index=3, x=2565, y=1678, width=1484, height=652, confidence=1.0),
            RegionResult(index=4, x=303, y=2161, width=2024, height=652, confidence=1.0),
        ]
        result = validate_region_set(regions, img_w=4049, img_h=3500)
        overlap_failures = [f for f in result.failures if f.reason == "overlap"]
        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(overlap_failures), 2)
        overlap_pairs = {(f.region_index, f.overlap_with) for f in overlap_failures}
        self.assertIn((1, 2), overlap_pairs)
        self.assertIn((4, 2), overlap_pairs)

    def test_validation_result_valid_flag_false_when_any_failure(self):
        # One valid, one zero-area
        regions = [
            RegionResult(index=0, x=0, y=0, width=0, height=50, confidence=0.9),
            RegionResult(index=1, x=10, y=10, width=80, height=80, confidence=0.9),
        ]
        result = validate_region_set(regions, img_w=200, img_h=200)
        self.assertFalse(result.valid)

    def test_validate_regions_for_write_is_wrapper(self):
        """validate_regions_for_write should return the same kept list as validate_region_set."""
        regions = [
            RegionResult(index=0, x=0, y=0, width=100, height=100, confidence=0.6),
            RegionResult(index=1, x=2, y=2, width=98, height=98, confidence=0.95),
        ]
        kept_via_set = validate_region_set(regions, img_w=200, img_h=200).kept
        kept_via_write = validate_regions_for_write(regions, img_w=200, img_h=200)
        self.assertEqual([r.index for r in kept_via_set], [r.index for r in kept_via_write])


# ---------------------------------------------------------------------------
# Stale XMP region reprocessing
# ---------------------------------------------------------------------------


class TestStaleXmpRegionReprocessing(unittest.TestCase):
    """Stored XMP regions that fail validation must be reprocessed."""

    def _make_jpeg(self, path: Path, width: int = 200, height: int = 100) -> None:
        try:
            from PIL import Image

            img = Image.new("RGB", (width, height), color=(128, 128, 128))
            img.save(str(path), format="JPEG")
        except ImportError:
            self.skipTest("PIL not available")

    def test_run_detect_reprocesses_page_with_invalid_stored_regions(self):
        """When stored XMP regions fail validation, run_detect_view_regions reprocesses the page."""
        from photoalbums.commands import run_detect_view_regions

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            view_dir = tmp_path / "TestAlbum_View"
            view_dir.mkdir()
            view_path = view_dir / "TestAlbum_B00_P26_V.jpg"
            self._make_jpeg(view_path)
            xmp_path = view_path.with_suffix(".xmp")

            # Write two heavily overlapping regions (invalid) into the XMP
            from photoalbums.lib.xmp_sidecar import write_pipeline_step, write_region_list
            from photoalbums.lib.ai_view_regions import RegionWithCaption

            img_w, img_h = 200, 100
            # Two nearly-identical boxes — one will be the overlap failure
            r0 = RegionWithCaption(
                RegionResult(index=0, x=0, y=0, width=100, height=100, confidence=0.6), caption=""
            )
            r1 = RegionWithCaption(
                RegionResult(index=1, x=2, y=2, width=98, height=98, confidence=0.95), caption=""
            )
            write_region_list(xmp_path, [r0, r1], img_w, img_h)
            write_pipeline_step(xmp_path, "view_regions", model="test", extra={"result": "regions_found"})

            detect_calls: list = []

            def mock_detect(path, **kwargs):
                detect_calls.append(kwargs)
                return []  # return empty so we don't need to write anything

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", side_effect=mock_detect):
                run_detect_view_regions(
                    album_id="TestAlbum",
                    photos_root=str(tmp_path),
                    page=None,
                    force=False,
                    redo_no_regions=False,
                )

            # The page must have been reprocessed (not skipped) due to the invalid stored regions
            self.assertGreater(len(detect_calls), 0, "Expected page to be reprocessed, not skipped")


# ---------------------------------------------------------------------------
# Repair prompt content
# ---------------------------------------------------------------------------


class TestRepairPrompt(unittest.TestCase):
    def test_repair_prompt_includes_prior_regions(self):
        prior = [
            RegionResult(index=0, x=10, y=20, width=100, height=80),
            RegionResult(index=1, x=12, y=22, width=98, height=78),
        ]
        failures = [
            RegionFailure(
                region_index=0, reason="overlap", severity="hard", overlap_with=1, overlap_fraction=0.92
            )
        ]
        prompt = _build_repair_prompt(prior, failures, img_w=200, img_h=200)
        # Prior regions are listed in box_2d format
        self.assertIn("index=0", prompt)
        self.assertIn("index=1", prompt)
        self.assertIn("box_2d=", prompt)
        # Failure is described
        self.assertIn("overlaps region 1", prompt)
        self.assertIn("92%", prompt)
        self.assertIn("limit: 5%", prompt)
        # Asks for complete replacement
        self.assertIn("COMPLETE revised JSON array", prompt)

    def test_repair_prompt_zero_area_failure_description(self):
        prior = [RegionResult(index=0, x=0, y=0, width=0, height=50)]
        failures = [RegionFailure(region_index=0, reason="zero_area", severity="hard")]
        prompt = _build_repair_prompt(prior, failures, img_w=200, img_h=200)
        self.assertIn("zero or negative area", prompt)

    def test_repair_prompt_full_page_failure_description(self):
        prior = [RegionResult(index=0, x=0, y=0, width=200, height=200)]
        failures = [RegionFailure(region_index=0, reason="full_page", severity="hard")]
        prompt = _build_repair_prompt(prior, failures, img_w=200, img_h=200)
        self.assertIn("album background", prompt)

    def test_repair_prompt_insufficient_page_coverage_description(self):
        prior = [RegionResult(index=0, x=0, y=0, width=98, height=100)]
        failures = [
            RegionFailure(
                region_index=-1,
                reason="insufficient_page_coverage",
                severity="hard",
                page_fraction=0.49,
            )
        ]
        prompt = _build_repair_prompt(prior, failures, img_w=200, img_h=200)
        self.assertIn("covers only 49% of the page", prompt)
        self.assertIn("minimum: 50%", prompt)


# ---------------------------------------------------------------------------
# Repair retry loop in detect_regions
# ---------------------------------------------------------------------------


class TestDetectRegionsRepairLoop(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch("photoalbums.lib.ai_view_regions.default_view_region_models", return_value=["gemma-3"]),
            mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="gemma-3"),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()

    def _make_jpeg(self, path: Path, width: int = 800, height: int = 600) -> None:
        try:
            from PIL import Image

            img = Image.new("RGB", (width, height), color=(128, 128, 128))
            img.save(str(path), format="JPEG")
        except ImportError:
            self.skipTest("PIL not available")

    def _mock_response(self, boxes):
        """boxes is a list of [ymin, xmin, ymax, xmax] in 0–1000."""
        return {"choices": [{"message": {"content": json.dumps(
            [{"box_2d": b, "label": "photograph"} for b in boxes]
        )}}]}

    def test_repair_prompt_used_after_validation_failure(self):
        """When validation fails, the next attempt should receive prior_regions in the payload."""
        from photoalbums.lib.ai_view_regions import detect_regions

        # Two heavily overlapping full-page regions (will fail validation)
        overlapping = [[0, 0, 1000, 1000], [2, 2, 998, 998]]
        # Second attempt returns valid non-overlapping halves
        valid_regions = [[0, 0, 1000, 500], [0, 500, 1000, 1000]]
        payloads_received: list[dict] = []

        def mock_post(url, payload, timeout):
            payloads_received.append(payload)
            if len(payloads_received) == 1:
                return self._mock_response(overlapping)
            return self._mock_response(valid_regions)

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            self._make_jpeg(img_path)
            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                results = detect_regions(img_path, force=True)
            self.assertTrue(_failed_regions_debug_path(img_path, attempt_number=1).is_file())
            self.assertTrue(_accepted_regions_debug_path(img_path, attempt_number=2).is_file())
            self.assertTrue(_accepted_regions_debug_path(img_path).is_file())
            self.assertFalse(_failed_regions_debug_path(img_path).is_file())

        self.assertEqual(len(results), 2)
        self.assertEqual(len(payloads_received), 2, "Expected exactly two model calls")
        # Second call's user text should include repair context (prior region coordinates)
        second_user_text = payloads_received[1]["messages"][1]["content"][0]["text"]
        self.assertIn("Prior region set", second_user_text)
        self.assertIn("Validation failures", second_user_text)

    def test_no_regions_still_writes_accepted_debug_image(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        def mock_post(url, payload, timeout):
            return self._mock_response([])

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            self._make_jpeg(img_path)
            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                results = detect_regions(img_path, force=True)
            self.assertEqual(results, [])
            self.assertTrue(_accepted_regions_debug_path(img_path, attempt_number=1).is_file())
            self.assertTrue(_accepted_regions_debug_path(img_path).is_file())

    def test_returns_empty_when_all_retries_fail_validation(self):
        """When all retries produce overlapping regions, detect_regions returns []."""
        from photoalbums.lib.ai_view_regions import detect_regions

        # Every response has a full-page region (always fails validation)
        def mock_post(url, payload, timeout):
            return self._mock_response([[0, 0, 1000, 1000]])

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            self._make_jpeg(img_path)
            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                results = detect_regions(img_path, force=True)
            self.assertTrue(_failed_regions_debug_path(img_path, attempt_number=1).is_file())
            self.assertTrue(_failed_regions_debug_path(img_path, attempt_number=2).is_file())
            self.assertTrue(_failed_regions_debug_path(img_path, attempt_number=3).is_file())
            self.assertTrue(_failed_regions_debug_path(img_path).is_file())

        self.assertEqual(results, [])

    def test_detect_regions_falls_back_to_next_model(self):
        from photoalbums.lib.ai_view_regions import detect_regions

        attempted_models = []

        def mock_post(url, payload, timeout):
            attempted_models.append(payload["model"])
            if payload["model"] == "bad-model":
                raise RuntimeError("bad-model failed")
            # Single region covering ~70% of the page (passes both full_page and coverage checks)
            return self._mock_response([[0, 0, 1000, 700]])

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            try:
                from PIL import Image

                img = Image.new("RGB", (800, 600))
                img.save(str(img_path), format="JPEG")
            except ImportError:
                self.skipTest("PIL not available")

            with (
                mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post),
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_models", return_value=["bad-model", "good-model"]),
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="bad-model"),
            ):
                results = detect_regions(img_path, force=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(attempted_models[:4], ["bad-model", "bad-model", "bad-model", "good-model"])

    def test_repair_metadata_captured_in_debug_metadata(self):
        """repair_prompt flag is captured in debug metadata on the second attempt."""
        from photoalbums.lib.ai_view_regions import detect_regions
        from photoalbums.lib.prompt_debug import PromptDebugSession

        records: list[dict] = []

        def capture_record(**kwargs):
            records.append(kwargs)

        debug_session = PromptDebugSession("test_V.jpg")
        debug_session.record = capture_record

        call_count = 0

        def mock_post(url, payload, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Full-page box → fails validation → triggers repair on attempt 2
                return self._mock_response([[0, 0, 1000, 1000]])
            return self._mock_response([[0, 0, 1000, 500], [0, 500, 1000, 1000]])

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test_V.jpg"
            self._make_jpeg(img_path)
            with mock.patch("photoalbums.lib.ai_view_regions._lmstudio_post", side_effect=mock_post):
                detect_regions(img_path, force=True, prompt_debug=debug_session)

        repair_records = [r for r in records if (r.get("metadata") or {}).get("repair_prompt")]
        self.assertTrue(len(repair_records) > 0, "Expected at least one repair_prompt=True record")


# ---------------------------------------------------------------------------
# associate_captions
# ---------------------------------------------------------------------------


class TestAssociateCaptions(unittest.TestCase):
    def _make_regions(self, coords):
        return [RegionResult(index=i, x=x, y=y, width=w, height=h) for i, (x, y, w, h) in enumerate(coords)]

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

    def test_caption_hint_and_person_names_round_trip(self):
        """Region written with caption_hint and person_names reads back with the same values."""
        img_w, img_h = 800, 600
        regions = [
            RegionWithCaption(
                RegionResult(
                    index=0,
                    x=0,
                    y=0,
                    width=400,
                    height=600,
                    caption_hint="People at the beach",
                    person_names=("Alice", "Bob"),
                ),
                caption="",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            write_region_list(xmp_path, regions, img_w, img_h)
            read_back = read_region_list(xmp_path, img_w, img_h)
            self.assertEqual(len(read_back), 1)
            self.assertEqual(read_back[0]["caption_hint"], "People at the beach")
            self.assertEqual(read_back[0]["person_names"], ["Alice", "Bob"])

    def test_empty_person_names_reads_back_as_empty_list(self):
        """Region with empty person_names reads back as []."""
        img_w, img_h = 800, 600
        regions = [
            RegionWithCaption(
                RegionResult(index=0, x=0, y=0, width=400, height=600, caption_hint="", person_names=()),
                caption="",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            write_region_list(xmp_path, regions, img_w, img_h)
            read_back = read_region_list(xmp_path, img_w, img_h)
            self.assertEqual(read_back[0]["person_names"], [])


if __name__ == "__main__":
    unittest.main()


class TestRunDetectViewRegions(unittest.TestCase):
    def _write_jpeg(self, path: Path) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")
        image = Image.new("RGB", (200, 100), color=(120, 120, 120))
        image.save(path, format="JPEG")

    def test_skips_when_pipeline_state_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)
            write_region_list(
                view_path.with_suffix(".xmp"),
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=120, height=100), "")],
                200,
                100,
            )
            write_pipeline_step(
                view_path.with_suffix(".xmp"),
                "view_regions",
                model="gemma",
                extra={"completed": "2026-04-11T08:00:00Z", "result": "regions_found"},
            )

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions") as detect_mock:
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_not_called()

    def test_writes_no_regions_pipeline_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)
            write_region_list(
                view_path.with_suffix(".xmp"),
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=200, height=100), "")],
                200,
                100,
            )

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]):
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            state = read_pipeline_step(view_path.with_suffix(".xmp"), "view_regions")
            assert state is not None
            self.assertEqual(state["result"], "no_regions")
            self.assertEqual(read_region_list(view_path.with_suffix(".xmp"), 200, 100), [])

    def test_debug_writes_request_response_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)

            def detect_with_debug(*args, **kwargs):
                prompt_debug = kwargs["prompt_debug"]
                prompt_debug.record(
                    step="view_regions",
                    engine="lmstudio",
                    model="gemma",
                    prompt="Describe the regions",
                    system_prompt="Return JSON",
                    response='{"regions":[]}',
                    metadata={"attempt_number": 1},
                )
                return [RegionResult(index=0, x=10, y=10, width=80, height=60, confidence=0.9)]

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", side_effect=detect_with_debug):
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    debug=True,
                )

            self.assertEqual(exit_code, 0)
            debug_path = root / "_debug" / f"{view_path.stem}.view-regions.debug.json"
            self.assertTrue(debug_path.is_file())
            artifact = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["kind"], "photoalbums_prompts")
            self.assertEqual(artifact["image_path"], str(view_path))
            self.assertEqual(artifact["step_count"], 1)

    def test_reads_page_caption_and_roster_into_detection_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Cordell_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Cordell_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)

            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_path.with_suffix(".xmp"),
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="audrey-leslie on the lawn",
                ocr_text="",
            )

            with mock.patch(
                "photoalbums.lib.album_sets.find_archive_set_by_photos_root",
                return_value=resolve_archive_set("cordell"),
            ):
                with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]) as detect_mock:
                    from photoalbums.commands import run_detect_view_regions

                    exit_code = run_detect_view_regions(
                        album_id="Cordell_1975",
                        photos_root=str(root),
                        page=None,
                        force=False,
                    )

            self.assertEqual(exit_code, 0)
            kwargs = detect_mock.call_args.kwargs
            self.assertEqual(kwargs["page_caption"], "audrey-leslie on the lawn")
            self.assertEqual(kwargs["album_context"], "Cordell 1975, book 00, page 26")
            self.assertEqual(
                kwargs["people_roster"],
                {
                    "audrey": "Audrey Cordell",
                    "leslie": "Leslie Cordell",
                },
            )

    def test_force_clears_existing_pipeline_state_and_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)
            write_pipeline_step(
                view_path.with_suffix(".xmp"),
                "view_regions",
                model="old-model",
                extra={"completed": "2026-04-11T08:00:00Z", "result": "no_regions"},
            )
            regions = [RegionResult(index=0, x=0, y=0, width=200, height=100, caption_hint="Page")]

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=regions) as detect_mock:
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=True,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_called_once()
            state = read_pipeline_step(view_path.with_suffix(".xmp"), "view_regions")
            assert state is not None
            self.assertEqual(state["result"], "regions_found")
            from photoalbums.lib.ai_model_settings import default_view_region_model

            self.assertEqual(state["model"], default_view_region_model())

    def test_reruns_when_regions_found_state_exists_but_region_list_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)
            write_pipeline_step(
                view_path.with_suffix(".xmp"),
                "view_regions",
                model="old-model",
                extra={"completed": "2026-04-11T08:00:00Z", "result": "regions_found"},
            )
            regions = [RegionResult(index=0, x=0, y=0, width=200, height=100, caption_hint="Page")]

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=regions) as detect_mock:
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_called_once()
            state = read_pipeline_step(view_path.with_suffix(".xmp"), "view_regions")
            assert state is not None
            self.assertEqual(state["result"], "regions_found")

    def test_title_page_p01_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P01_V.jpg"
            self._write_jpeg(view_path)

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions") as detect_mock:
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_not_called()
            self.assertIsNone(read_pipeline_step(view_path.with_suffix(".xmp"), "view_regions"))

    def test_album_wide_run_ignores_derived_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            page_view = view_dir / "Egypt_1975_B00_P26_V.jpg"
            derived_view = view_dir / "Egypt_1975_B00_P26_D01-02_V.jpg"
            self._write_jpeg(page_view)
            self._write_jpeg(derived_view)

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]) as detect_mock:
                from photoalbums.commands import run_detect_view_regions

                exit_code = run_detect_view_regions(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            processed_paths = {call.args[0] for call in detect_mock.call_args_list}
            self.assertIn(page_view, processed_paths)
            self.assertNotIn(derived_view, processed_paths)
