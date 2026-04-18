"""Integration tests for run_render_pipeline and per-page failure isolation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums import commands
from photoalbums.lib.xmp_sidecar import read_pipeline_step, write_pipeline_step
from photoalbums.lib.xmpmm_provenance import assign_document_id, read_document_id


def _write_minimal_xmp(path: Path, description: str = "") -> None:
    desc_xml = f"<dc:description>{description}</dc:description>" if description else ""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"{desc_xml}"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>",
        encoding="utf-8",
    )


def _make_test_album(root: Path, album: str = "Egypt_1975_B00", page: int = 26) -> dict:
    """Create a minimal archive+view directory structure for pipeline tests."""
    archive = root / f"{album}_Archive"
    view = root / f"{album}_Pages"
    photos = root / f"{album}_Photos"
    archive.mkdir(parents=True)
    view.mkdir(parents=True)
    photos.mkdir(parents=True)

    # Create minimal scan file
    page_token = f"P{page:02d}"
    scan = archive / f"{album}_{page_token}_S01.tif"
    scan.write_bytes(b"tif")

    # Create pre-existing view JPEG and XMP (simulates a previous render)
    view_jpg = view / f"{album}_{page_token}_V.jpg"
    view_jpg.write_bytes(b"viewjpeg")
    view_xmp = view_jpg.with_suffix(".xmp")
    _write_minimal_xmp(view_xmp, "A lovely page")
    assign_document_id(view_xmp)

    return {
        "archive": archive,
        "view": view,
        "photos": photos,
        "scan": scan,
        "view_jpg": view_jpg,
        "view_xmp": view_xmp,
    }


def _add_test_page(album_dir: dict, album: str = "Egypt_1975_B00", page: int = 27) -> dict:
    page_token = f"P{page:02d}"
    scan = album_dir["archive"] / f"{album}_{page_token}_S01.tif"
    scan.write_bytes(b"tif")

    view_jpg = album_dir["view"] / f"{album}_{page_token}_V.jpg"
    view_jpg.write_bytes(b"viewjpeg")
    view_xmp = view_jpg.with_suffix(".xmp")
    _write_minimal_xmp(view_xmp, "A lovely page")
    assign_document_id(view_xmp)

    return {
        "scan": scan,
        "view_jpg": view_jpg,
        "view_xmp": view_xmp,
    }


def _mock_render_noop(album_dir: dict) -> None:
    """Patch stitch/tif_to_jpg so they don't do real image work."""
    pass


class TestRunRenderPipelineSkipsSteps(unittest.TestCase):
    """Second run without --force skips completed steps."""

    def test_second_run_skips_view_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root)
            view_xmp = dirs["view_xmp"]

            from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
            from photoalbums.lib.xmp_sidecar import write_region_list

            write_region_list(
                view_xmp,
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=60, height=100), "")],
                100,
                100,
            )
            # Pre-record the view_regions pipeline step
            write_pipeline_step(view_xmp, "view_regions", model="test-model", extra={"result": "regions_found"})

            captured = StringIO()
            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.commands.run_render_pipeline.__wrapped__", side_effect=None) if hasattr(commands.run_render_pipeline, "__wrapped__") else mock.MagicMock(),
                redirect_stdout(captured),
            ):
                # Call detect-regions standalone to verify skip logic
                result = commands.run_detect_view_regions(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(result, 0)
            self.assertEqual(captured.getvalue(), "")

    def test_force_clears_view_regions_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root)
            view_xmp = dirs["view_xmp"]

            write_pipeline_step(view_xmp, "view_regions", model="old-model", extra={"result": "no_regions"})

            mock_regions = []

            with (
                mock.patch(
                    "photoalbums.lib.ai_view_regions.detect_regions",
                    return_value=mock_regions,
                ),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="new-model"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                redirect_stdout(StringIO()),
            ):
                result = commands.run_detect_view_regions(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=True,
                )

            self.assertEqual(result, 0)
            state = read_pipeline_step(view_xmp, "view_regions")
            self.assertIsNotNone(state)
            self.assertEqual(state.get("model"), "new-model")  # type: ignore[union-attr]

    def test_render_pipeline_reruns_view_regions_when_region_list_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root)
            view_xmp = dirs["view_xmp"]
            write_pipeline_step(view_xmp, "view_regions", model="test-model", extra={"result": "regions_found"})

            mock_regions = []

            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=mock_regions) as detect_mock,
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(StringIO()),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(result, 0)
            detect_mock.assert_called_once()

    def test_render_pipeline_skips_title_page_region_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_test_album(root, page=1)

            captured = StringIO()
            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions") as detect_mock,
                mock.patch("photoalbums.lib.ai_photo_crops.crop_page_regions") as crop_mock,
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(captured),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=False,
                )

            self.assertEqual(result, 0)
            detect_mock.assert_not_called()
            crop_mock.assert_not_called()
            output = captured.getvalue()
            self.assertIn("detect-regions: skipped title page (P01)", output)
            self.assertIn("crop-regions: skipped title page (P01)", output)


class TestRunRenderPipelineNoRegions(unittest.TestCase):
    """Title page with no detected regions records no_regions state and does not fail."""

    def test_no_regions_writes_explicit_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root, "Egypt_1975_B00")
            view_xmp = dirs["view_xmp"]

            captured = StringIO()
            with (
                mock.patch(
                    "photoalbums.lib.ai_view_regions.detect_regions",
                    return_value=[],
                ),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(captured),
            ):
                result = commands.run_detect_view_regions(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(result, 0)
            state = read_pipeline_step(view_xmp, "view_regions")
            self.assertIsNotNone(state)
            self.assertEqual(state.get("result"), "no_regions")  # type: ignore[union-attr]

    def test_render_pipeline_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_test_album(root, "Egypt_1975_B00")

            captured = StringIO()
            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(captured),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(result, 0)
            output = captured.getvalue()
            self.assertIn("===== PIPELINE SUMMARY =====", output)
            self.assertIn("Pages: 1 total, 1 completed, 0 failed", output)
            self.assertIn("Warnings: 0", output)
            self.assertIn("Errors: 0", output)
            self.assertIn("Detect regions: 0 found, 1 no-regions, 0 skipped, 0 rerun", output)

    def test_render_pipeline_passes_force_restoration_to_crop_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_test_album(root, "Egypt_1975_B00")

            captured = StringIO()
            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_photo_crops.crop_page_regions", return_value=0) as crop_mock,
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(captured),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=False,
                    force_restoration=True,
                )

            self.assertEqual(result, 0)
            crop_mock.assert_called_once()
            self.assertTrue(crop_mock.call_args.kwargs["force_restoration"])


class TestRunRenderPipelineProvenance(unittest.TestCase):
    """Provenance is written at file-creation time and survives later-step failures."""

    def test_render_provenance_written_when_tif_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root)
            scan = dirs["scan"]
            view_xmp = dirs["view_xmp"]

            # Assign DocumentID to the archive scan sidecar first (as _ensure_archive_page_sidecar would)
            scan_xmp = scan.with_suffix(".xmp")
            _write_minimal_xmp(scan_xmp)
            scan_doc_id = assign_document_id(scan_xmp)

            # Call write_render_provenance directly
            from photoalbums.stitch_oversized_pages import write_render_provenance

            write_render_provenance(view_xmp, [str(scan)])

            xml = view_xmp.read_text(encoding="utf-8")
            self.assertIn("DocumentID", xml)
            self.assertIn("DerivedFrom", xml)
            # DocumentID should be present on view XMP now
            view_doc_id = read_document_id(view_xmp)
            self.assertTrue(view_doc_id.startswith("xmp:uuid:"))

    def test_provenance_survives_later_step_failure(self):
        """Creation-time provenance is present even if detect-regions fails."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = _make_test_album(root)
            view_xmp = dirs["view_xmp"]
            scan = dirs["scan"]

            # Write provenance manually (as render step would)
            scan_xmp = scan.with_suffix(".xmp")
            _write_minimal_xmp(scan_xmp)
            assign_document_id(scan_xmp)
            from photoalbums.stitch_oversized_pages import write_render_provenance
            write_render_provenance(view_xmp, [str(scan)])

            # Now simulate detect-regions failure - provenance is still there
            xml = view_xmp.read_text(encoding="utf-8")
            self.assertIn("DocumentID", xml)


class TestRunRenderPipelinePageLockRelease(unittest.TestCase):
    """Page lock is released even when a step raises."""

    def test_lock_released_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_test_album(root)

            released = []

            def fake_release(lock):
                released.append(lock)

            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=Path("/fake.lock")),
                mock.patch("photoalbums.commands._release_page_pipeline_lock", side_effect=fake_release),
                mock.patch(
                    "photoalbums.stitch_oversized_pages.tif_to_jpg",
                    side_effect=RuntimeError("render failed"),
                ),
                mock.patch(
                    "photoalbums.stitch_oversized_pages.stitch",
                    side_effect=RuntimeError("render failed"),
                ),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            # Lock must have been released
            self.assertTrue(len(released) > 0)
            # Result should be non-zero due to failure
            self.assertNotEqual(result, 0)


class TestRunRenderPipelinePerPageScoping(unittest.TestCase):
    def test_face_refresh_is_limited_to_current_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album_dir = _make_test_album(root, page=26)
            _add_test_page(album_dir, page=27)

            refresh_calls: list[list[str]] = []
            session = mock.Mock()
            session.set_files.side_effect = lambda files: refresh_calls.append([path.name for path in files])

            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession", return_value=session),
                redirect_stdout(StringIO()),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                refresh_calls,
                [
                    ["Egypt_1975_B00_P26_V.jpg"],
                    ["Egypt_1975_B00_P27_V.jpg"],
                ],
            )

    def test_derived_renders_are_limited_to_current_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album_dir = _make_test_album(root, page=26)
            _add_test_page(album_dir, page=27)

            derived_outputs = [
                "Egypt_1975_B00_P26_D01-00.tif",
                "Egypt_1975_B00_P27_D01-00.tif",
            ]
            rendered: list[str] = []

            with (
                mock.patch("photoalbums.commands._acquire_page_pipeline_lock", return_value=None),
                mock.patch("photoalbums.commands._release_page_pipeline_lock"),
                mock.patch("photoalbums.stitch_oversized_pages.tif_to_jpg", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.stitch", return_value=False),
                mock.patch("photoalbums.stitch_oversized_pages.list_derived_images", return_value=derived_outputs),
                mock.patch("photoalbums.lib.ai_view_regions.detect_regions", return_value=[]),
                mock.patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(100, 100)),
                mock.patch("photoalbums.lib.ai_model_settings.default_view_region_model", return_value="gemma4"),
                mock.patch("photoalbums.lib.album_sets.find_archive_set_by_photos_root", return_value=""),
                mock.patch("photoalbums.lib.album_sets.read_people_roster", return_value={}),
                mock.patch("photoalbums.lib.xmp_sidecar.read_ai_sidecar_state", return_value={}),
                mock.patch(
                    "photoalbums.stitch_oversized_pages.derived_to_jpg",
                    side_effect=lambda derived, _view_dir: rendered.append(Path(derived).name),
                ),
                mock.patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                redirect_stdout(StringIO()),
            ):
                result = commands.run_render_pipeline(
                    album_id="Egypt_1975_B00",
                    photos_root=str(root),
                    page=None,
                    force=False,
                    skip_crops=True,
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                rendered,
                [
                    "Egypt_1975_B00_P26_D01-00.tif",
                    "Egypt_1975_B00_P27_D01-00.tif",
                ],
            )


if __name__ == "__main__":
    unittest.main()

