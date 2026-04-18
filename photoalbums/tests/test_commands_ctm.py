from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums import commands
from photoalbums.lib.ai_ctm_restoration import CTMResult
from photoalbums.lib.xmp_sidecar import read_pipeline_step, write_ctm_to_archive_xmp, write_pipeline_step


class TestRunCTM(unittest.TestCase):
    def test_generate_uses_stitched_view_image_and_archive_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")

            result = CTMResult(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                model_name="gemma",
                source_path=str(view),
            )

            with (
                mock.patch(
                    "photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]
                ) as archive_dirs_mock,
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch(
                    "photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm",
                    return_value=(scan.with_suffix(".xmp"), result),
                ) as generate_mock,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["generate", "--photos-root", str(root)])

            self.assertEqual(exit_code, 0)
            archive_dirs_mock.assert_called_once_with(root)
            generate_mock.assert_called_once_with(
                view,
                archive_sidecar_path=scan.with_suffix(".xmp"),
                force=False,
            )
            payload = json.loads(stdout.getvalue().strip())
            self.assertEqual(payload["image"], scan.name)
            self.assertEqual(payload["source_image"], str(view))
            self.assertEqual(payload["archive_xmp"], str(scan.with_suffix(".xmp")))

    def test_review_reports_missing_ctm_as_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            scan.write_bytes(b"scan")

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["review", "--photos-root", str(root)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue().strip())
            self.assertEqual(payload["image"], scan.name)
            self.assertIsNone(payload["ctm"])

    def test_generate_skips_when_pipeline_state_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")
            archive_xmp = scan.with_suffix(".xmp")
            write_ctm_to_archive_xmp(
                archive_xmp,
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                creator_tool="imago-test",
                source_image_path=scan.name,
                model_name="gemma",
            )
            write_pipeline_step(
                archive_xmp,
                "ctm",
                model="gemma",
                extra={"completed": "2026-04-11T07:00:00Z"},
            )

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch("photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm") as generate_mock,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["generate", "--photos-root", str(root)])

            self.assertEqual(exit_code, 0)
            generate_mock.assert_not_called()
            self.assertIn(
                "Skipping Family_2020_B01_P01_S01.tif CTM generation (already completed at 2026-04-11T07:00:00Z; use --force to rerun)",
                stdout.getvalue(),
            )

    def test_generate_writes_pipeline_state_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")

            result = CTMResult(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                model_name="gemma",
                source_path=str(view),
            )

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch(
                    "photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm",
                    return_value=(scan.with_suffix(".xmp"), result),
                ),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["generate", "--photos-root", str(root)])

            self.assertEqual(exit_code, 0)
            state = read_pipeline_step(scan.with_suffix(".xmp"), "ctm")
            assert state is not None
            self.assertEqual(state["model"], "gemma")
            payload = json.loads(stdout.getvalue().strip())
            self.assertEqual(payload["archive_xmp"], str(scan.with_suffix(".xmp")))

    def test_generate_force_clears_old_pipeline_state_before_regenerating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")
            archive_xmp = scan.with_suffix(".xmp")
            write_pipeline_step(
                archive_xmp,
                "ctm",
                model="old-model",
                extra={"completed": "2026-04-10T07:00:00Z"},
            )

            result = CTMResult(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                model_name="gemma",
                source_path=str(view),
            )

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch(
                    "photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm",
                    return_value=(archive_xmp, result),
                ) as generate_mock,
            ):
                exit_code = commands.run_ctm(["generate", "--photos-root", str(root), "--force"])

            self.assertEqual(exit_code, 0)
            generate_mock.assert_called_once_with(
                view,
                archive_sidecar_path=archive_xmp,
                force=True,
            )
            state = read_pipeline_step(archive_xmp, "ctm")
            assert state is not None
            self.assertEqual(state["model"], "gemma")
            self.assertNotEqual(state["completed"], "2026-04-10T07:00:00Z")

    def test_generate_per_photo_uses_crop_sidecars_and_writes_pipeline_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            photos_dir = root / "Family_2020_B01_Photos"
            archive_dir.mkdir()
            photos_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            crop = photos_dir / "Family_2020_B01_P01_D01-00_V.jpg"
            scan.write_bytes(b"scan")
            crop.write_bytes(b"crop")

            result = CTMResult(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                model_name="gemma",
                source_path=str(crop),
            )

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch(
                    "photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm",
                    return_value=(crop.with_suffix(".xmp"), result),
                ) as generate_mock,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["generate", "--photos-root", str(root), "--per-photo"])

            self.assertEqual(exit_code, 0)
            generate_mock.assert_called_once_with(
                crop,
                archive_sidecar_path=crop.with_suffix(".xmp"),
                force=False,
            )
            state = read_pipeline_step(crop.with_suffix(".xmp"), "ctm")
            assert state is not None
            self.assertEqual(state["model"], "gemma")
            payload = json.loads(stdout.getvalue().strip())
            self.assertEqual(payload["image"], crop.name)

    def test_run_ctm_apply_applies_page_and_crop_ctms_and_skips_on_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_Pages"
            photos_dir = root / "Family_2020_B01_Photos"
            archive_dir.mkdir()
            view_dir.mkdir()
            photos_dir.mkdir()

            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            crop = photos_dir / "Family_2020_B01_P01_D01-00_V.jpg"
            crop_no_ctm = photos_dir / "Family_2020_B01_P01_D02-00_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")
            crop.write_bytes(b"crop")
            crop_no_ctm.write_bytes(b"crop-no-ctm")

            write_ctm_to_archive_xmp(
                scan.with_suffix(".xmp"),
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="page",
                creator_tool="imago-test",
                source_image_path=scan.name,
                model_name="page-model",
            )
            write_ctm_to_archive_xmp(
                crop.with_suffix(".xmp"),
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="crop",
                creator_tool="imago-test",
                source_image_path=crop.name,
                model_name="crop-model",
            )

            with (
                mock.patch(
                    "photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]
                ) as archive_dirs_mock,
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch("photoalbums.lib.ai_ctm_restoration.apply_ctm_to_jpeg") as apply_mock,
            ):
                exit_code = commands.run_ctm_apply(
                    album_id="Family_2020_B01",
                    photos_root=str(root),
                    page="1",
                    force=False,
                )

                self.assertEqual(exit_code, 0)
                archive_dirs_mock.assert_called_once_with(root)
                self.assertEqual(
                    [call.args[0] for call in apply_mock.call_args_list],
                    [view, crop],
                )
                page_state = read_pipeline_step(view.with_suffix(".xmp"), "ctm_applied")
                crop_state = read_pipeline_step(crop.with_suffix(".xmp"), "ctm_applied")
                self.assertIsNotNone(page_state)
                self.assertIsNotNone(crop_state)
                self.assertIsNone(read_pipeline_step(crop_no_ctm.with_suffix(".xmp"), "ctm_applied"))

                apply_mock.reset_mock()
                exit_code = commands.run_ctm_apply(
                    album_id="Family_2020_B01",
                    photos_root=str(root),
                    page="1",
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            apply_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

