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

if __name__ == "__main__":
    unittest.main()

