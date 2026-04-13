"""Tests for ai_photo_crops: region crop module."""

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

from photoalbums.lib.ai_photo_crops import (
    crop_output_path,
    crop_page_regions,
    mwgrs_normalised_to_pixel_rect,
    resolve_region_caption,
)
from photoalbums.lib.xmp_sidecar import read_pipeline_step, read_region_list, write_pipeline_step


# ---------------------------------------------------------------------------
# resolve_region_caption
# ---------------------------------------------------------------------------


class TestResolveRegionCaption(unittest.TestCase):
    def test_region_description_wins(self):
        result = resolve_region_caption("Region desc", "Hint", "Page desc")
        self.assertEqual(result, "Region desc")

    def test_hint_used_when_description_empty(self):
        result = resolve_region_caption("", "Hint", "Page desc")
        self.assertEqual(result, "Hint")

    def test_page_caption_used_when_both_empty(self):
        result = resolve_region_caption("", "", "Page desc")
        self.assertEqual(result, "Page desc")

    def test_empty_string_when_all_empty(self):
        result = resolve_region_caption("", "", "")
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# mwgrs_normalised_to_pixel_rect
# ---------------------------------------------------------------------------


class TestMwgrsNormalisedToPixelRect(unittest.TestCase):
    def test_centre_region(self):
        # cx=0.5, cy=0.5, w=0.5, h=0.5 -> left=250, top=250, right=750, bottom=750
        left, top, right, bottom = mwgrs_normalised_to_pixel_rect(0.5, 0.5, 0.5, 0.5, 1000, 1000)
        self.assertEqual(left, 250)
        self.assertEqual(top, 250)
        self.assertEqual(right, 750)
        self.assertEqual(bottom, 750)

    def test_edge_touching_region(self):
        # left edge: cx=0.25, cy=0.5, w=0.5, h=1.0 -> left=0, top=0, right=500, bottom=1000
        left, top, right, bottom = mwgrs_normalised_to_pixel_rect(0.25, 0.5, 0.5, 1.0, 1000, 1000)
        self.assertEqual(left, 0)
        self.assertEqual(top, 0)
        self.assertEqual(right, 500)
        self.assertEqual(bottom, 1000)

    def test_out_of_bounds_region_clamped(self):
        # Extends beyond image bounds: cx=0.9, cy=0.9, w=0.5, h=0.5
        # raw: left=0.65*1000=650, top=650, right=1.15*1000=1150, bottom=1150
        left, top, right, bottom = mwgrs_normalised_to_pixel_rect(0.9, 0.9, 0.5, 0.5, 1000, 1000)
        self.assertEqual(left, 650)
        self.assertEqual(top, 650)
        self.assertEqual(right, 1000)  # clamped to img_w
        self.assertEqual(bottom, 1000)  # clamped to img_h


# ---------------------------------------------------------------------------
# crop_output_path
# ---------------------------------------------------------------------------


class TestCropOutputPath(unittest.TestCase):
    def test_builds_expected_path(self):
        view_path = Path("/Photos/Egypt_1975_View/Egypt_1975_B00_P26_V.jpg")
        photos_dir = Path("/Photos/Egypt_1975_Photos")
        result = crop_output_path(view_path, 2, photos_dir)
        self.assertEqual(result.name, "Egypt_1975_B00_P26_D02-00_V.jpg")
        self.assertEqual(result.parent, photos_dir)

    def test_region_index_zero_padded(self):
        view_path = Path("/Photos/Egypt_1975_View/Egypt_1975_B00_P01_V.jpg")
        photos_dir = Path("/Photos/Egypt_1975_Photos")
        result = crop_output_path(view_path, 1, photos_dir)
        self.assertEqual(result.name, "Egypt_1975_B00_P01_D01-00_V.jpg")


# ---------------------------------------------------------------------------
# crop_page_regions
# ---------------------------------------------------------------------------


def _make_minimal_jpeg(path: Path, width: int = 100, height: int = 100) -> None:
    """Write a tiny solid JPEG using Pillow."""
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(200, 200, 200))
        img.save(str(path), format="JPEG", quality=85)
    except ImportError:
        # Fallback: write a minimal valid JPEG header (3x3 pixels)
        # This is a pre-encoded 1x1 white JPEG
        _TINY_JPEG = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\x18\x18\x1b-7,-\x1f/2\x1e\x1f\xff\xc0\x00\x0b\x08\x00\x01\x00"
            b"\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
            b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05"
            b"\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02"
            b"\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05"
            b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1'
            b"\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJKLMNOPQR"
            b"STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\x03\xff\xd9"
        )
        path.write_bytes(_TINY_JPEG)


def _write_region_xmp(xmp_path: Path, regions: list[dict], img_w: int, img_h: int) -> None:
    """Write an XMP sidecar with mwg-rs regions using write_region_list."""
    from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
    from photoalbums.lib.xmp_sidecar import write_region_list

    rwcs = []
    for r in regions:
        result = RegionResult(
            index=r.get("index", 0),
            x=r.get("x", 0),
            y=r.get("y", 0),
            width=r.get("width", img_w),
            height=r.get("height", img_h),
            caption_hint=r.get("caption_hint", ""),
            person_names=tuple(r.get("person_names", [])),
        )
        rwcs.append(RegionWithCaption(result, caption=r.get("caption", "")))
    write_region_list(xmp_path, rwcs, img_w, img_h)


class TestCropPageRegions(unittest.TestCase):
    def test_derived_view_input_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_D01-01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100, "caption": "Derived"},
                ],
                img_w,
                img_h,
            )

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 0)
            self.assertFalse(photos_dir.exists())

    def test_no_regions_returns_zero_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)
            # No XMP sidecar -> no regions
            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 0)
            self.assertFalse(photos_dir.exists())

    def test_no_regions_pipeline_state_clears_stale_region_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, 200, 100)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 200, "y": 100, "width": 200, "height": 100, "caption": "Overflow"},
                ],
                200,
                100,
            )
            write_pipeline_step(view_xmp, "view_regions", model="test-model", extra={"result": "no_regions"})

            count = crop_page_regions(view_jpg, photos_dir)

            self.assertEqual(count, 0)
            self.assertEqual(read_region_list(view_xmp, 200, 100), [])
            self.assertFalse(photos_dir.exists())

    def test_two_regions_writes_two_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 100, "height": 100, "caption": "Left photo"},
                    {"index": 1, "x": 100, "y": 0, "width": 100, "height": 100, "caption": "Right photo"},
                ],
                img_w,
                img_h,
            )

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 2)
            self.assertTrue((photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg").exists())
            self.assertTrue((photos_dir / "Egypt_1975_B00_P01_D02-00_V.jpg").exists())

    def test_region_caption_resolved_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)

            # Region with empty caption + non-empty page caption -> page caption
            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="Beach day",
                ocr_text="",
            )
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100, "caption": ""},
                ],
                img_w,
                img_h,
            )

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 1)

            crop_xmp = photos_dir / "Egypt_1975_B00_P01_D01-00_V.xmp"
            self.assertTrue(crop_xmp.exists())
            xml = crop_xmp.read_text(encoding="utf-8")
            self.assertIn("Beach day", xml)

    def test_existing_file_skipped_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100, "caption": "Test"},
                ],
                img_w,
                img_h,
            )

            existing_crop = photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg"
            existing_crop.write_bytes(b"existing")

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 0)
            # File still has original content
            self.assertEqual(existing_crop.read_bytes(), b"existing")

    def test_existing_file_overwritten_with_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100, "caption": "Test"},
                ],
                img_w,
                img_h,
            )

            existing_crop = photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg"
            existing_crop.write_bytes(b"old content")

            count = crop_page_regions(view_jpg, photos_dir, force=True)
            self.assertEqual(count, 1)
            # File should be a proper JPEG now
            self.assertNotEqual(existing_crop.read_bytes(), b"old content")

    def test_empty_clamped_region_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": img_w, "y": img_h, "width": img_w, "height": img_h, "caption": "Overflow"},
                ],
                img_w,
                img_h,
            )

            with self.assertLogs("photoalbums.lib.ai_photo_crops", level="WARNING") as logs:
                count = crop_page_regions(view_jpg, photos_dir)

            self.assertEqual(count, 0)
            self.assertTrue(any("Ignoring empty crop region" in line for line in logs.output))
            self.assertFalse((photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg").exists())


# ---------------------------------------------------------------------------
# _write_crop_sidecar
# ---------------------------------------------------------------------------


class TestWriteCropSidecar(unittest.TestCase):
    def test_sidecar_contains_document_id_and_derived_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)

            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="A page",
                ocr_text="",
            )
            from photoalbums.lib.xmpmm_provenance import assign_document_id

            view_doc_id = assign_document_id(view_xmp)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "Test caption", {}, [], [])

            crop_xmp = crop_jpg.with_suffix(".xmp")
            self.assertTrue(crop_xmp.exists())
            xml = crop_xmp.read_text(encoding="utf-8")
            self.assertIn("DocumentID", xml)
            self.assertIn("DerivedFrom", xml)
            self.assertIn(view_doc_id, xml)

    def test_caption_written_when_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, img_w, img_h)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "Beautiful sunset", {}, [], [])

            xml = crop_jpg.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertIn("Beautiful sunset", xml)

    def test_no_dc_description_when_caption_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "", {}, [], [])

            xml = crop_jpg.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertNotIn("dc:description", xml)

    def test_location_propagated_to_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            view_state = {
                "gps_latitude": "30.0444",
                "gps_longitude": "31.2357",
                "location_city": "Cairo",
                "location_country": "Egypt",
                "create_date": "1975",
                "dc_date_values": ["1975"],
            }
            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "", view_state, [], [])

            xml = crop_jpg.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertIn("Cairo", xml)
            self.assertIn("Egypt", xml)
            self.assertIn("GPSLatitude", xml)

    def test_empty_location_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "", {}, [], [])

            xml = crop_jpg.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertNotIn("GPSLatitude", xml)
            self.assertNotIn("photoshop:City", xml)

    def test_person_names_written_to_person_in_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")
            _write_crop_sidecar(crop_jpg, view_jpg, "", {}, [], ["Audrey Cordell"])

            xml = crop_jpg.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertIn("Audrey Cordell", xml)
            self.assertIn("PersonInImage", xml)

    def test_rerun_preserves_unrelated_sidecar_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            _make_minimal_jpeg(view_jpg, 200, 100)

            from photoalbums.lib.ai_photo_crops import _write_crop_sidecar
            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            crop_jpg = photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg"
            crop_jpg.write_bytes(b"placeholder")

            # Write sidecar with a manual subject
            crop_xmp = crop_jpg.with_suffix(".xmp")
            write_xmp_sidecar(
                crop_xmp,
                creator_tool="manual",
                person_names=[],
                subjects=["manual-tag"],
                description="",
                ocr_text="",
            )

            # Rerun crop sidecar write should preserve manual-tag
            _write_crop_sidecar(crop_jpg, view_jpg, "New caption", {}, [], [])

            xml = crop_xmp.read_text(encoding="utf-8")
            self.assertIn("New caption", xml)
            # subjects are preserved via write_xmp_sidecar merge
            self.assertIn("manual-tag", xml)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class TestCropPageRegionsPipelineState(unittest.TestCase):
    def test_second_call_without_force_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100},
                ],
                img_w,
                img_h,
            )

            # First call
            count1 = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count1, 1)
            self.assertIsNotNone(read_pipeline_step(view_xmp, "crop_regions"))

            # Second call without force
            count2 = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count2, 0)

    def test_force_clears_state_and_recrups(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100},
                ],
                img_w,
                img_h,
            )

            # First call
            crop_page_regions(view_jpg, photos_dir)

            # Force re-run
            count = crop_page_regions(view_jpg, photos_dir, force=True)
            self.assertEqual(count, 1)
            # State still written after force run
            self.assertIsNotNone(read_pipeline_step(view_xmp, "crop_regions"))

    def test_missing_outputs_rerun_even_when_pipeline_state_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            view_jpg = view_dir / "Egypt_1975_B00_P26_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 100, "height": 100},
                    {"index": 1, "x": 100, "y": 0, "width": 100, "height": 100},
                ],
                img_w,
                img_h,
            )

            count1 = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count1, 2)
            self.assertIsNotNone(read_pipeline_step(view_xmp, "crop_regions"))

            missing_crop = photos_dir / "Egypt_1975_B00_P26_D02-00_V.jpg"
            missing_sidecar = missing_crop.with_suffix(".xmp")
            missing_crop.unlink()
            missing_sidecar.unlink()

            count2 = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count2, 1)
            self.assertTrue(missing_crop.exists())
            self.assertTrue(missing_sidecar.exists())

    def test_orphan_cleanup_on_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            view_dir = Path(tmp) / "Egypt_1975_View"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Egypt_1975_Photos"
            photos_dir.mkdir()
            view_jpg = view_dir / "Egypt_1975_B00_P01_V.jpg"
            view_xmp = view_jpg.with_suffix(".xmp")
            _make_minimal_jpeg(view_jpg, img_w, img_h)

            # Write 3 regions first
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 66, "height": 100},
                    {"index": 1, "x": 66, "y": 0, "width": 66, "height": 100},
                    {"index": 2, "x": 132, "y": 0, "width": 68, "height": 100},
                ],
                img_w,
                img_h,
            )
            count1 = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count1, 3)
            orphan = photos_dir / "Egypt_1975_B00_P01_D03-00_V.jpg"
            self.assertTrue(orphan.exists())

            # Now update XMP to only 2 regions and force re-run
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 100, "height": 100},
                    {"index": 1, "x": 100, "y": 0, "width": 100, "height": 100},
                ],
                img_w,
                img_h,
            )
            count2 = crop_page_regions(view_jpg, photos_dir, force=True)
            self.assertEqual(count2, 2)
            self.assertFalse(orphan.exists())
            self.assertTrue((photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg").exists())
            self.assertTrue((photos_dir / "Egypt_1975_B00_P01_D02-00_V.jpg").exists())


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# Integration tests (tasks 7.x)
# ---------------------------------------------------------------------------


class TestIntegrationCropPipeline(unittest.TestCase):
    """Integration tests for the full crop-regions pipeline step."""

    def _setup_album(self, tmp: str, img_w: int = 200, img_h: int = 100, page: int = 1) -> tuple:
        """Create a minimal album structure. Returns (view_dir, photos_dir, view_jpg, view_xmp)."""
        root = Path(tmp)
        archive_dir = root / "Egypt_1975_Archive"
        archive_dir.mkdir(parents=True)
        page_token = f"P{page:02d}"
        scan = archive_dir / f"Egypt_1975_B00_{page_token}_S01.tif"
        scan.write_bytes(b"tif")
        view_dir = root / "Egypt_1975_View"
        view_dir.mkdir(parents=True)
        photos_dir = root / "Egypt_1975_Photos"
        view_jpg = view_dir / f"Egypt_1975_B00_{page_token}_V.jpg"
        view_xmp = view_jpg.with_suffix(".xmp")
        _make_minimal_jpeg(view_jpg, img_w, img_h)
        return view_dir, photos_dir, view_jpg, view_xmp

    def test_full_pipeline_creates_photos_dir_with_correct_crops(self):
        """7.1: _Photos/ created, correct crop count, each sidecar has DocumentID and DerivedFrom."""
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            _, photos_dir, view_jpg, view_xmp = self._setup_album(tmp, img_w, img_h)

            # Assign DocumentID to page view (simulates render step)
            from photoalbums.lib.xmpmm_provenance import assign_document_id
            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="Egypt trip 1975",
                ocr_text="",
            )
            view_doc_id = assign_document_id(view_xmp)

            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 100, "height": 100, "caption": "Pyramid"},
                    {"index": 1, "x": 100, "y": 0, "width": 100, "height": 100, "caption": "Sphinx"},
                ],
                img_w,
                img_h,
            )

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 2)
            self.assertTrue(photos_dir.exists())

            for idx, expected_caption in [(1, "Pyramid"), (2, "Sphinx")]:
                crop_jpg = photos_dir / f"Egypt_1975_B00_P01_D{idx:02d}-00_V.jpg"
                crop_xmp = crop_jpg.with_suffix(".xmp")
                self.assertTrue(crop_jpg.exists(), f"Missing crop: {crop_jpg.name}")
                self.assertTrue(crop_xmp.exists(), f"Missing sidecar: {crop_xmp.name}")
                xml = crop_xmp.read_text(encoding="utf-8")
                self.assertIn("DocumentID", xml)
                self.assertIn("DerivedFrom", xml)
                self.assertIn(view_doc_id, xml)
                self.assertIn(expected_caption, xml)

    def test_skip_crops_produces_no_photos_dir(self):
        """7.2: --skip-crops produces no _Photos/ files and no pipeline.crop_regions state."""
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            _, photos_dir, view_jpg, view_xmp = self._setup_album(tmp, img_w, img_h)
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 200, "height": 100},
                ],
                img_w,
                img_h,
            )

            from photoalbums.lib.xmp_sidecar import read_pipeline_step

            # run_render_pipeline with skip_crops=True - mock detect_regions to avoid model call
            with (
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="test"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(img_w, img_h)),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
            ):
                from photoalbums.commands import run_render_pipeline

                run_render_pipeline(
                    album_id="Egypt_1975",
                    photos_root=str(Path(tmp)),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertFalse(photos_dir.exists())
            self.assertIsNone(read_pipeline_step(view_xmp, "crop_regions"))

    def test_render_pipeline_releases_page_lock_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, view_jpg, _ = self._setup_album(tmp, 200, 100)

            with (
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="test"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(200, 100)),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
            ):
                from photoalbums.commands import run_render_pipeline

                exit_code = run_render_pipeline(
                    album_id="Egypt_1975",
                    photos_root=str(Path(tmp)),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(view_jpg.with_name(f"{view_jpg.name}.photoalbums-ai.lock").exists())

    def test_render_pipeline_releases_page_lock_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, view_jpg, _ = self._setup_album(tmp, 200, 100, page=26)

            with (
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", side_effect=RuntimeError("boom")),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="test"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(200, 100)),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
            ):
                from photoalbums.commands import run_render_pipeline

                exit_code = run_render_pipeline(
                    album_id="Egypt_1975",
                    photos_root=str(Path(tmp)),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(view_jpg.with_name(f"{view_jpg.name}.photoalbums-ai.lock").exists())

    def test_page_caption_inherited_by_all_crops_when_no_per_region_captions(self):
        """7.4: Page with only page-level dc:description -> all crop sidecars inherit page caption."""
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            _, photos_dir, view_jpg, view_xmp = self._setup_album(tmp, img_w, img_h)

            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="Family reunion 1975",
                ocr_text="",
            )
            _write_region_xmp(
                view_xmp,
                [
                    {"index": 0, "x": 0, "y": 0, "width": 100, "height": 100, "caption": ""},
                    {"index": 1, "x": 100, "y": 0, "width": 100, "height": 100, "caption": ""},
                ],
                img_w,
                img_h,
            )

            count = crop_page_regions(view_jpg, photos_dir)
            self.assertEqual(count, 2)

            for idx in [1, 2]:
                crop_xmp = photos_dir / f"Egypt_1975_B00_P01_D{idx:02d}-00_V.xmp"
                xml = crop_xmp.read_text(encoding="utf-8")
                self.assertIn("Family reunion 1975", xml)

    def test_run_crop_regions_generates_missing_view_regions_before_cropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_w, img_h = 200, 100
            _, photos_dir, _, view_xmp = self._setup_album(tmp, img_w, img_h, page=26)

            from photoalbums.commands import run_crop_regions
            from photoalbums.lib.ai_view_regions import RegionResult
            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="Egypt trip 1975",
                ocr_text="",
            )

            with (
                mock.patch(
                    "photoalbums.lib.ai_view_regions.detect_regions",
                    return_value=[RegionResult(index=0, x=0, y=0, width=200, height=100, caption_hint="Pyramid")],
                ) as detect_mock,
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="test-model"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
            ):
                exit_code = run_crop_regions(
                    album_id="Egypt_1975",
                    photos_root=str(Path(tmp)),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_called_once()
            self.assertTrue((photos_dir / "Egypt_1975_B00_P26_D01-00_V.jpg").exists())

    def test_run_crop_regions_skips_title_page_p01(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, photos_dir, _, view_xmp = self._setup_album(tmp, 200, 100, page=1)

            from photoalbums.commands import run_crop_regions
            from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

            write_xmp_sidecar(
                view_xmp,
                creator_tool="test",
                person_names=[],
                subjects=[],
                description="Egypt trip 1975",
                ocr_text="",
            )

            with mock.patch("photoalbums.lib.ai_view_regions.detect_regions") as detect_mock:
                exit_code = run_crop_regions(
                    album_id="Egypt_1975",
                    photos_root=str(Path(tmp)),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            detect_mock.assert_not_called()
            self.assertFalse(any(photos_dir.glob("*.jpg")))
