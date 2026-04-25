"""Tests for Docling-backed view region detection."""

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
    RegionFailure,
    RegionResult,
    RegionWithCaption,
    _accepted_regions_debug_path,
    _docling_raw_debug_path,
    _failed_regions_debug_path,
    _has_xmp_regions,
    _read_regions_from_xmp,
    _region_association_overlay_path,
    associate_captions,
    detect_regions,
    pixel_to_mwgrs,
    validate_region_set,
    validate_regions_for_write,
)
from photoalbums.lib.album_sets import resolve_archive_set
from photoalbums.lib.prompt_debug import PromptDebugSession
from photoalbums.lib.xmp_sidecar import read_pipeline_step, read_region_list, write_pipeline_step, write_region_list


class TestPixelToMwgrs(unittest.TestCase):
    def test_centre_point_computed_correctly(self):
        cx, cy, nw, nh = pixel_to_mwgrs(100, 200, 400, 300, 1000, 1000)
        self.assertAlmostEqual(cx, 0.3)
        self.assertAlmostEqual(cy, 0.35)
        self.assertAlmostEqual(nw, 0.4)
        self.assertAlmostEqual(nh, 0.3)


class TestValidateRegions(unittest.TestCase):
    def test_validate_regions_for_write_keeps_valid_non_overlapping_regions(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=80, height=80, confidence=0.7),
            RegionResult(index=1, x=90, y=0, width=80, height=80, confidence=0.9),
            RegionResult(index=2, x=0, y=0, width=0, height=10, confidence=1.0),
        ]
        kept = validate_regions_for_write(regions, img_w=200, img_h=100)
        self.assertEqual([region.index for region in kept], [0, 1])

    def test_validate_region_set_reports_overlap(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=95, height=95, confidence=0.9),
            RegionResult(index=1, x=2, y=2, width=95, height=95, confidence=0.8),
        ]
        result = validate_region_set(regions, img_w=200, img_h=200)
        self.assertTrue(result.valid)
        self.assertEqual(result.failures, [])
        self.assertEqual([region.index for region in result.kept], [0, 1])


class TestDetectRegionsDocling(unittest.TestCase):
    def _write_jpeg(self, path: Path, size=(800, 600)) -> None:
        from PIL import Image

        Image.new("RGB", size, color=(128, 128, 128)).save(path, format="JPEG")

    def test_detect_regions_uses_docling_pipeline_and_returns_regions(self):
        fake_regions = [
            RegionResult(index=0, x=10, y=20, width=100, height=80, caption_hint="left"),
            RegionResult(index=1, x=200, y=50, width=150, height=120, caption_hint="right"),
        ]
        fake_result = mock.Mock(regions=fake_regions, debug_payload={"docling": True})

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(img_path)
            prompt_debug = PromptDebugSession(img_path, label=img_path.name)

            with (
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="granite-docling-258m"),
                mock.patch("photoalbums.lib.ai_view_regions.default_docling_preset", return_value="granite_docling"),
                mock.patch("photoalbums.lib.ai_view_regions.default_docling_backend", return_value="auto_inline"),
                mock.patch("photoalbums.lib.ai_view_regions.default_docling_device", return_value="auto"),
                mock.patch("photoalbums.lib.ai_view_regions.default_docling_retries", return_value=3),
                mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline", return_value=fake_result) as mock_pipeline,
                mock.patch("photoalbums.lib.ai_view_regions._apply_lmstudio_captions", side_effect=lambda regions, *_args: regions),
            ):
                regions = detect_regions(img_path, force=True, prompt_debug=prompt_debug, skip_validation=True)

        self.assertEqual(len(regions), 2)
        self.assertEqual(mock_pipeline.call_args.kwargs["preset"], "granite_docling")
        self.assertEqual(mock_pipeline.call_args.kwargs["backend"], "auto_inline")
        self.assertEqual(mock_pipeline.call_args.kwargs["device"], "auto")
        self.assertEqual(mock_pipeline.call_args.kwargs["retries"], 3)
        debug_path = _docling_raw_debug_path(img_path)
        self.assertTrue(debug_path.is_file())
        self.assertEqual(json.loads(debug_path.read_text(encoding="utf-8")), {"docling": True})

    def test_detect_regions_records_failed_pipeline_state_for_runtime_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(img_path)
            xmp_path = img_path.with_suffix(".xmp")
            prompt_debug = PromptDebugSession(img_path, label=img_path.name)

            runtime_error = Exception("boom")
            runtime_error.debug_payload = {"error": "boom"}  # type: ignore[attr-defined]
            with (
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="granite-docling-258m"),
                mock.patch(
                    "photoalbums.lib._docling_pipeline.run_docling_pipeline",
                    side_effect=__import__("photoalbums.lib._docling_pipeline", fromlist=["DoclingPipelineRuntimeError"]).DoclingPipelineRuntimeError(
                        "Docling pipeline failed due to: boom",
                        debug_payload={"error": "boom"},
                    ),
                ),
            ):
                regions = detect_regions(img_path, force=True, prompt_debug=prompt_debug)
                self.assertEqual(regions, [])
                state = read_pipeline_step(xmp_path, "detect-regions/docling")
                assert state is not None
                self.assertEqual(state["result"], "failed")
                self.assertEqual(json.loads(_docling_raw_debug_path(img_path).read_text(encoding="utf-8")), {"error": "boom"})

    def test_detect_regions_records_validation_failed_without_retrying_cached_failure(self):
        fake_regions = [RegionResult(index=0, x=0, y=0, width=800, height=600)]
        fake_result = mock.Mock(regions=fake_regions, debug_payload={"docling": True})

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(img_path)
            xmp_path = img_path.with_suffix(".xmp")

            with (
                mock.patch("photoalbums.lib.ai_view_regions.default_view_region_model", return_value="granite-docling-258m"),
                mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline", return_value=fake_result) as mock_pipeline,
            ):
                regions = detect_regions(img_path, force=True)
                self.assertEqual(regions, [])
                state = read_pipeline_step(xmp_path, "detect-regions/docling")
                assert state is not None
                self.assertEqual(state["result"], "validation_failed")

                regions = detect_regions(img_path, force=False)
                self.assertEqual(regions, [])
                self.assertEqual(mock_pipeline.call_count, 1)

    def test_detect_regions_uses_cached_xmp_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(img_path, size=(200, 100))
            xmp_path = img_path.with_suffix(".xmp")
            write_region_list(
                xmp_path,
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=100, height=100, caption_hint="Beach"), "")],
                200,
                100,
            )
            with mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline") as mock_pipeline:
                regions = detect_regions(img_path, force=False)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].caption_hint, "Beach")
        self.assertTrue(_accepted_regions_debug_path(img_path).is_file())
        self.assertTrue(_region_association_overlay_path(img_path).is_file())
        mock_pipeline.assert_not_called()


class TestAssociateCaptions(unittest.TestCase):
    def test_empty_caption_list_preserves_region_order(self):
        regions = [
            RegionResult(index=0, x=0, y=0, width=100, height=100),
            RegionResult(index=1, x=100, y=0, width=100, height=100),
        ]
        result = associate_captions(regions, [], img_width=200)
        self.assertEqual([row.region.index for row in result], [0, 1])


class TestRegionListXmpRoundTrip(unittest.TestCase):
    def test_write_and_read_back_uses_mwg_rs_name_for_caption(self):
        regions = [
            RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=600, caption_hint="Left hint"), ""),
            RegionWithCaption(RegionResult(index=1, x=400, y=0, width=400, height=600), "Right caption"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            xmp_path = Path(tmp) / "test_V.xmp"
            write_region_list(xmp_path, regions, 800, 600)
            self.assertTrue(_has_xmp_regions(xmp_path))

            xml = xmp_path.read_text(encoding="utf-8")
            self.assertIn('mwg-rs:Name="Left hint"', xml)
            self.assertIn('mwg-rs:Name="Right caption"', xml)
            self.assertNotIn("dc:description", xml)

            read_back = read_region_list(xmp_path, 800, 600)
            reread_regions = _read_regions_from_xmp(xmp_path, 800, 600)

        self.assertEqual(read_back[0]["caption"], "Left hint")
        self.assertEqual(read_back[1]["caption"], "Right caption")
        self.assertEqual(reread_regions[0].caption_hint, "Left hint")
        self.assertEqual(reread_regions[1].caption_hint, "Right caption")


class TestRunDetectViewRegions(unittest.TestCase):
    def _write_jpeg(self, path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (200, 100), color=(120, 120, 120)).save(path, format="JPEG")

    def test_skips_when_terminal_pipeline_state_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_Pages"
            view_dir.mkdir()
            view_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)
            write_pipeline_step(
                view_path.with_suffix(".xmp"),
                "view_regions",
                model="granite-docling-258m",
                extra={"completed": "2026-04-11T08:00:00Z", "result": "failed"},
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

    def test_reads_page_caption_and_roster_into_detection_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Cordell_1975_Pages"
            view_dir.mkdir()
            view_path = view_dir / "Cordell_1975_B00_P26_V.jpg"
            self._write_jpeg(view_path)

            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_path.with_suffix(".xmp"),
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
        self.assertEqual(kwargs["people_roster"], {"audrey": "Audrey Cordell", "leslie": "Leslie Cordell"})


class TestSortRegionsReadingOrder(unittest.TestCase):
    def test_single_region_returned_unchanged(self):
        from photoalbums.lib._caption_matching import sort_regions_reading_order

        region = RegionResult(index=0, x=50, y=80, width=100, height=60)
        result = sort_regions_reading_order([region], img_height=400)
        self.assertEqual(result, [region])

    def test_two_row_layout_sorted_top_to_bottom(self):
        from photoalbums.lib._caption_matching import sort_regions_reading_order

        top_left = RegionResult(index=0, x=10, y=10, width=100, height=80)
        top_right = RegionResult(index=1, x=200, y=12, width=100, height=80)
        bottom = RegionResult(index=2, x=100, y=300, width=100, height=80)
        result = sort_regions_reading_order([bottom, top_right, top_left], img_height=600)
        self.assertEqual(result[0], top_left)
        self.assertEqual(result[1], top_right)
        self.assertEqual(result[2], bottom)

    def test_same_row_tie_breaking_by_x(self):
        from photoalbums.lib._caption_matching import sort_regions_reading_order

        left = RegionResult(index=0, x=10, y=50, width=80, height=60)
        right = RegionResult(index=1, x=300, y=55, width=80, height=60)
        result = sort_regions_reading_order([right, left], img_height=400)
        self.assertEqual(result[0], left)
        self.assertEqual(result[1], right)

    def test_empty_list_returns_empty(self):
        from photoalbums.lib._caption_matching import sort_regions_reading_order

        self.assertEqual(sort_regions_reading_order([], img_height=400), [])


class TestCallGemma4CaptionMatchingParsing(unittest.TestCase):
    def _call_with_response(self, content: str, *, locations_shown: list[dict] | None = None) -> dict:
        from photoalbums.lib._caption_matching import call_lmstudio_caption_matching

        fake_response = {"choices": [{"message": {"content": content}}]}
        with mock.patch("photoalbums.lib._caption_matching._post_json", return_value=fake_response):
            with mock.patch("photoalbums.lib._caption_matching._encode_image", return_value="data:image/jpeg;base64,abc"):
                return call_lmstudio_caption_matching(
                    "fake.jpg",
                    base_url="http://localhost:1234/v1",
                    model="gemma",
                    locations_shown=locations_shown,
                )

    def test_clean_json_parsed_correctly(self):
        result = self._call_with_response('{"region-1": "At the beach", "region-2": "Summer 1985"}')
        self.assertEqual(
            result,
            {1: {"caption": "At the beach", "location": ""}, 2: {"caption": "Summer 1985", "location": ""}},
        )

    def test_json_in_markdown_code_fence_extracted(self):
        content = '```json\n{"region-1": "Holiday", "region-2": "Party"}\n```'
        result = self._call_with_response(content)
        self.assertEqual(
            result,
            {1: {"caption": "Holiday", "location": ""}, 2: {"caption": "Party", "location": ""}},
        )

    def test_multi_location_json_parsed_correctly(self):
        result = self._call_with_response(
            '{"region-1": {"caption": "Cairo street scene", "location": "Cairo, Egypt"}, "region-2": {"caption": "Karnak Temple", "location": "Luxor, Egypt"}}',
            locations_shown=[{"name": "Cairo, Egypt"}, {"name": "Luxor, Egypt"}],
        )
        self.assertEqual(
            result,
            {
                1: {"caption": "Cairo street scene", "location": "Cairo, Egypt"},
                2: {"caption": "Karnak Temple", "location": "Luxor, Egypt"},
            },
        )

    def test_multi_location_response_normalizes_to_known_location_match(self):
        result = self._call_with_response(
            '{"region-1": {"caption": "Liberation Square - Cairo Egypt Dec. 1975", "location": "Liberation Square - Cairo Egypt Dec. 1975"}}',
            locations_shown=[{"name": "Hilton Hotel"}, {"name": "Liberation Square"}],
        )
        self.assertEqual(
            result,
            {
                1: {
                    "caption": "Liberation Square - Cairo Egypt Dec. 1975",
                    "location": "Liberation Square",
                }
            },
        )

    def test_single_location_prompt_uses_legacy_string_values(self):
        from photoalbums.lib._caption_matching import call_lmstudio_caption_matching

        fake_response = {"choices": [{"message": {"content": '{"region-1": "Holiday"}'}}]}
        with mock.patch("photoalbums.lib._caption_matching._post_json", return_value=fake_response) as post_json:
            with mock.patch("photoalbums.lib._caption_matching._encode_image", return_value="data:image/jpeg;base64,abc"):
                call_lmstudio_caption_matching(
                    "fake.jpg",
                    base_url="http://localhost:1234/v1",
                    model="gemma",
                    locations_shown=[{"name": "Cairo, Egypt"}],
                )
        prompt = post_json.call_args.args[1]["messages"][0]["content"][1]["text"]
        self.assertIn('{"region-1": "", "region-2": "", "region-3": ""}', prompt)
        self.assertNotIn('"location": ""', prompt)

    def test_multi_location_prompt_uses_object_values_and_known_locations(self):
        from photoalbums.lib._caption_matching import call_lmstudio_caption_matching

        fake_response = {"choices": [{"message": {"content": '{"region-1": {"caption": "", "location": ""}}'}}]}
        with mock.patch("photoalbums.lib._caption_matching._post_json", return_value=fake_response) as post_json:
            with mock.patch("photoalbums.lib._caption_matching._encode_image", return_value="data:image/jpeg;base64,abc"):
                call_lmstudio_caption_matching(
                    "fake.jpg",
                    base_url="http://localhost:1234/v1",
                    model="gemma",
                    locations_shown=[{"name": "Cairo, Egypt"}, {"name": "Luxor, Egypt"}],
                )
        prompt = post_json.call_args.args[1]["messages"][0]["content"][1]["text"]
        self.assertIn('"location": ""', prompt)
        self.assertIn("Known locations: Cairo, Egypt, Luxor, Egypt", prompt)

    def test_malformed_json_returns_empty_dict(self):
        result = self._call_with_response("Here is my answer: {region-1: broken")
        self.assertEqual(result, {})

    def test_lmstudio_offline_returns_empty_dict(self):
        from photoalbums.lib._caption_matching import call_lmstudio_caption_matching

        with mock.patch("photoalbums.lib._caption_matching._post_json", side_effect=RuntimeError("LM Studio is unreachable")):
            with mock.patch("photoalbums.lib._caption_matching._encode_image", return_value="data:image/jpeg;base64,abc"):
                result = call_lmstudio_caption_matching("fake.jpg", base_url="http://localhost:1234/v1", model="gemma")
        self.assertEqual(result, {})


class TestAssignCaptionsFromGemma4(unittest.TestCase):
    def test_three_regions_get_correct_captions(self):
        from photoalbums.lib._caption_matching import assign_captions_from_lmstudio

        r0 = RegionResult(index=0, x=10, y=10, width=80, height=60)
        r1 = RegionResult(index=1, x=200, y=10, width=80, height=60)
        r2 = RegionResult(index=2, x=100, y=200, width=80, height=60)
        captions = {1: "First", 2: "Second", 3: "Third"}
        result = assign_captions_from_lmstudio([r0, r1, r2], captions)
        self.assertEqual(result[0].caption_hint, "First")
        self.assertEqual(result[1].caption_hint, "Second")
        self.assertEqual(result[2].caption_hint, "Third")
        self.assertEqual(result[0].location_hint, "")

    def test_fewer_captions_than_regions_leaves_remainder_empty(self):
        from photoalbums.lib._caption_matching import assign_captions_from_lmstudio

        r0 = RegionResult(index=0, x=0, y=0, width=100, height=80)
        r1 = RegionResult(index=1, x=200, y=0, width=100, height=80)
        result = assign_captions_from_lmstudio([r0, r1], {1: "Only one"})
        self.assertEqual(result[0].caption_hint, "Only one")
        self.assertEqual(result[1].caption_hint, "")

    def test_extra_captions_beyond_region_count_ignored(self):
        from photoalbums.lib._caption_matching import assign_captions_from_lmstudio

        r0 = RegionResult(index=0, x=0, y=0, width=100, height=80)
        result = assign_captions_from_lmstudio([r0], {1: "A", 2: "B", 3: "C"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].caption_hint, "A")

    def test_multi_location_assignment_carries_location_hint(self):
        from photoalbums.lib._caption_matching import assign_captions_from_lmstudio

        region = RegionResult(index=0, x=0, y=0, width=100, height=80)
        result = assign_captions_from_lmstudio(
            [region],
            {1: {"caption": "Temple visit", "location": "Luxor, Egypt"}},
        )
        self.assertEqual(result[0].caption_hint, "Temple visit")
        self.assertEqual(result[0].location_hint, "Luxor, Egypt")

    def test_unknown_overlay_key_is_ignored(self):
        from photoalbums.lib._caption_matching import assign_captions_from_lmstudio

        region = RegionResult(index=0, x=0, y=0, width=100, height=80)
        result = assign_captions_from_lmstudio([region], {2: "Wrong region"})
        self.assertEqual(result[0].caption_hint, "")


class TestGemma4CaptionInDetectRegions(unittest.TestCase):
    """Integration-style: detect_regions with Gemma4 step wired in."""

    def _write_jpeg(self, path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (200, 100), color=(120, 120, 120)).save(path, format="JPEG")

    def test_lmstudio_captions_assigned_when_model_configured(self):
        from photoalbums.lib._docling_pipeline import DoclingPipelineResult

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Test_1980_B00_P01_V.jpg"
            self._write_jpeg(img_path)
            raw_regions = [
                RegionResult(index=0, x=0, y=0, width=100, height=80),
                RegionResult(index=1, x=100, y=0, width=100, height=80),
                RegionResult(index=2, x=0, y=80, width=200, height=80),
            ]
            mock_pipeline_result = DoclingPipelineResult(regions=raw_regions, debug_payload={})
            with mock.patch("photoalbums.lib.ai_view_regions.default_caption_matching_model", return_value="gemma"):
                with mock.patch("photoalbums.lib.ai_view_regions.default_lmstudio_base_url", return_value="http://localhost:1234/v1"):
                    with mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline", return_value=mock_pipeline_result):
                        with mock.patch(
                            "photoalbums.lib._caption_matching.call_lmstudio_caption_matching",
                            return_value={
                                1: {"caption": "Caption A", "location": ""},
                                2: {"caption": "Caption B", "location": ""},
                                3: {"caption": "Caption C", "location": ""},
                            },
                        ) as caption_call:
                            regions = detect_regions(img_path, force=True, skip_validation=True)

        self.assertEqual(len(regions), 3)
        captions = [r.caption_hint for r in regions]
        self.assertIn("Caption A", captions)
        self.assertIn("Caption B", captions)
        self.assertIn("Caption C", captions)
        self.assertTrue(str(caption_call.call_args.args[0]).endswith(".view-regions.association-overlay.jpg"))

    def test_lmstudio_offline_regions_saved_with_empty_captions(self):
        from photoalbums.lib._docling_pipeline import DoclingPipelineResult

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "Test_1980_B00_P02_V.jpg"
            self._write_jpeg(img_path)
            raw_regions = [RegionResult(index=0, x=0, y=0, width=200, height=100)]
            mock_pipeline_result = DoclingPipelineResult(regions=raw_regions, debug_payload={})
            with mock.patch("photoalbums.lib.ai_view_regions.default_caption_matching_model", return_value="gemma"):
                with mock.patch("photoalbums.lib.ai_view_regions.default_lmstudio_base_url", return_value="http://localhost:1234/v1"):
                    with mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline", return_value=mock_pipeline_result):
                        with mock.patch(
                            "photoalbums.lib._caption_matching.call_lmstudio_caption_matching",
                            return_value={},
                        ):
                            regions = detect_regions(img_path, force=True, skip_validation=True)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].caption_hint, "")

    def test_multi_location_detect_regions_resolves_location_payload(self):
        from photoalbums.lib._docling_pipeline import DoclingPipelineResult
        from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Egypt_1975_B00_Archive"
            archive_dir.mkdir()
            view_dir = root / "Egypt_1975_B00_Pages"
            view_dir.mkdir()
            img_path = view_dir / "Egypt_1975_B00_P26_V.jpg"
            self._write_jpeg(img_path)
            (archive_dir / "Egypt_1975_B00_P26_S01.tif").write_bytes(b"scan")
            write_xmp_sidecar(
                (archive_dir / "Egypt_1975_B00_P26_S01.xmp"),
                person_names=[],
                subjects=[],
                description="",
                ocr_text="",
                locations_shown=[
                    {"name": "Cairo, Egypt"},
                    {"name": "Luxor, Egypt"},
                ],
            )
            raw_regions = [RegionResult(index=0, x=0, y=0, width=200, height=100)]
            mock_pipeline_result = DoclingPipelineResult(regions=raw_regions, debug_payload={})
            with mock.patch("photoalbums.lib.ai_view_regions.default_caption_matching_model", return_value="gemma"):
                with mock.patch("photoalbums.lib.ai_view_regions.default_lmstudio_base_url", return_value="http://localhost:1234/v1"):
                    with mock.patch("photoalbums.lib._docling_pipeline.run_docling_pipeline", return_value=mock_pipeline_result):
                        with mock.patch(
                            "photoalbums.lib._caption_matching.call_lmstudio_caption_matching",
                            return_value={1: {"caption": "Temple visit", "location": "Luxor, Egypt"}},
                        ):
                            with mock.patch("photoalbums.lib.ai_geocode.NominatimGeocoder") as geocoder_cls:
                                geocoder = geocoder_cls.return_value
                                geocoder.geocode.return_value = mock.Mock(
                                    query="Luxor, Egypt",
                                    display_name="Luxor, Egypt",
                                    latitude="25.6872",
                                    longitude="32.6396",
                                    source="nominatim",
                                    city="Luxor",
                                    state="Luxor",
                                    country="Egypt",
                                    sublocation="",
                                )
                                regions = detect_regions(img_path, force=True, skip_validation=True)

        self.assertEqual(regions[0].caption_hint, "Temple visit")
        self.assertEqual(regions[0].location_payload["city"], "Luxor")


if __name__ == "__main__":
    unittest.main()

