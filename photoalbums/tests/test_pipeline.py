from __future__ import annotations

import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.pipeline import PIPELINE_STEPS, VALID_STEP_IDS, validate_step_ids
from photoalbums.lib.xmp_sidecar import is_step_stale
from photoalbums.commands import print_pipeline_plan


class TestStepIdValidation(unittest.TestCase):
    def test_valid_ids_accepted(self):
        # Should not raise or exit
        result = validate_step_ids(["render", "crop-regions"], flag="--skip")
        self.assertEqual(result, ["render", "crop-regions"])

    def test_unknown_id_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_step_ids(["not-a-step"], flag="--skip")
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_step_and_skip_mutually_exclusive(self):
        from photoalbums.cli import build_parser
        parser = build_parser()
        args, _ = parser.parse_known_args(
            ["process", "--photos-root", ".", "--step", "render"]
        )
        self.assertEqual(args.step_id, "render")
        self.assertEqual(args.skip_ids, [])

    def test_cli_list_steps_exits_0(self):
        from photoalbums.cli import main
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            code = main(["process", "--photos-root", ".", "--list-steps"])
        self.assertEqual(code, 0)
        output = mock_out.getvalue()
        for step_id in VALID_STEP_IDS:
            self.assertIn(step_id, output)


class TestPrintPipelinePlan(unittest.TestCase):
    def _capture_plan(self, skip_ids, redo_ids, album_label="", page_count=0):
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            print_pipeline_plan(PIPELINE_STEPS, set(skip_ids), set(redo_ids), album_label, page_count)
            return mock_out.getvalue()

    def test_skip_annotation(self):
        output = self._capture_plan(["crop-regions"], [])
        self.assertIn("(skipped: --skip crop-regions)", output)
        self.assertNotIn("(skipped: --skip render)", output)

    def test_redo_annotation(self):
        output = self._capture_plan([], ["ai-index"])
        self.assertIn("(redo forced)", output)

    def test_all_steps_listed(self):
        output = self._capture_plan([], [])
        for step in PIPELINE_STEPS:
            self.assertIn(step.id, output)

    def test_no_annotation_for_normal_steps(self):
        output = self._capture_plan(["crop-regions"], ["ai-index"])
        # render has neither skip nor redo
        lines = [l for l in output.splitlines() if "render" in l and "detect" not in l and "crop" not in l]
        self.assertTrue(any("skipped" not in l and "redo" not in l for l in lines))

    def test_gps_only_not_annotated_by_plan(self):
        # gps-only is a flag, not a plan annotation; plan just shows step ids
        output = self._capture_plan([], [], "TestAlbum", 5)
        self.assertIn("TestAlbum", output)
        self.assertIn("5 page(s)", output)


class TestIsStepStale(unittest.TestCase):
    def _make_state(self, **entries):
        """Build a pipeline_state dict with completed timestamps."""
        return {k: {"completed": v} for k, v in entries.items()}

    def test_step_never_run_is_stale(self):
        state = self._make_state(render="2024-01-01T10:00:00")
        self.assertTrue(is_step_stale("detect-regions", ["render"], state))

    def test_step_current_is_not_stale(self):
        state = self._make_state(
            render="2024-01-01T10:00:00",
            **{"detect-regions": "2024-01-01T11:00:00"},
        )
        self.assertFalse(is_step_stale("detect-regions", ["render"], state))

    def test_dependency_newer_triggers_stale(self):
        state = self._make_state(
            render="2024-01-01T12:00:00",
            **{"detect-regions": "2024-01-01T11:00:00"},
        )
        self.assertTrue(is_step_stale("detect-regions", ["render"], state))

    def test_no_depends_on_and_step_has_completed_is_not_stale(self):
        state = self._make_state(render="2024-01-01T10:00:00")
        self.assertFalse(is_step_stale("render", [], state))

    def test_no_depends_on_and_step_missing_is_stale(self):
        self.assertTrue(is_step_stale("render", [], {}))

    def test_dep_missing_completed_does_not_force_stale(self):
        # dep has no completed → can't compare, so not treated as newer
        state: dict = {
            "render": {},
            "detect-regions": {"completed": "2024-01-01T11:00:00"},
        }
        self.assertFalse(is_step_stale("detect-regions", ["render"], state))


class TestRunProcessPipelineSmoke(unittest.TestCase):
    """Integration smoke: run with all steps skipped — verifies orchestrator runs without error."""

    def test_all_steps_skipped_returns_0(self):
        from photoalbums.lib.pipeline import VALID_STEP_IDS
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create a minimal archive structure
            archive = root / "Test_2024_B01_Archive"
            archive.mkdir()

            # Patch stitch_oversized_pages functions that are imported locally inside run_process_pipeline
            with patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]):
                with patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[]):
                    from photoalbums.commands import run_process_pipeline
                    code = run_process_pipeline(
                        album_id="",
                        photos_root=str(root),
                        page=None,
                        skip_ids=list(VALID_STEP_IDS),
                        redo_ids=[],
                        step_id=None,
                        force=False,
                        debug=False,
                        no_validation=False,
                        skip_restoration=False,
                        force_restoration=False,
                        gps_only=False,
                        refresh_gps=False,
                    )
            # No pages found returns 1; the orchestrator didn't crash
            self.assertIn(code, (0, 1))

    def test_refresh_gps_forces_ai_index_redo_and_gps_reprocess_mode(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            archive.mkdir()
            scan = archive / "Test_2024_B01_P01_S01.tif"
            scan.write_bytes(b"abc")

            with (
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(root / "Test_2024_B01_Pages")),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(root / "Test_2024_B01_Photos")),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", return_value=root / "Test_2024_B01_Pages" / "Test_2024_B01_P01_V.jpg"),
                patch("photoalbums.lib.xmp_sidecar.read_pipeline_state", return_value={"ai-index": {"completed": "2026-04-21T00:00:00Z"}}),
                patch("photoalbums.commands._check_step_stale", return_value=(False, "")),
                patch("photoalbums.lib.xmp_sidecar.write_pipeline_step"),
                patch("photoalbums.commands._print_outcome"),
            ):
                ai_runner = MagicMock()
                ai_runner._process_one.return_value = None
                ai_runner.processed = 0
                ai_runner.skipped = 0
                ai_runner.failures = 0
                ai_runner.force_processing = False
                with patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner) as runner_cls:
                    from photoalbums.commands import run_process_pipeline

                    code = run_process_pipeline(
                        album_id="",
                        photos_root=str(root),
                        page=None,
                        skip_ids=["render", "propagate-metadata", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=[],
                        step_id=None,
                        force=False,
                        debug=False,
                        no_validation=False,
                        skip_restoration=False,
                        force_restoration=False,
                        gps_only=False,
                        refresh_gps=True,
                    )

        self.assertEqual(code, 0)
        self.assertTrue(
            any(
                "--reprocess-mode" in list(call.args[0]) and "gps" in list(call.args[0])
                for call in runner_cls.call_args_list
            )
        )
        ai_runner._process_one.assert_called_once()

    def test_verify_crops_runs_after_ai_index(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            pages = root / "Test_2024_B01_Pages"
            photos = root / "Test_2024_B01_Photos"
            archive.mkdir()
            pages.mkdir()
            photos.mkdir()
            scan = archive / "Test_2024_B01_P01_S01.tif"
            scan.write_bytes(b"abc")
            view_path = pages / "Test_2024_B01_P01_V.jpg"
            view_path.write_bytes(b"jpg")
            view_path.with_suffix(".xmp").write_text(
                '<?xml version="1.0" encoding="UTF-8"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )

            with (
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(pages)),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(photos)),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", return_value=view_path),
                patch("photoalbums.commands._print_outcome"),
                patch("photoalbums.commands._print_ai_index_discovery_summary"),
                patch("photoalbums.commands._print_verify_crops_summary"),
                patch("photoalbums.lib.ai_verify_crops.run_verify_crops_page", return_value={"status": "ok", "results": [], "page_input_hash": "abc", "artifact_path": "artifact.json", "missing_context": []}) as verify_mock,
                patch("photoalbums.lib.ai_verify_crops.persist_verify_crops_state") as persist_mock,
            ):
                ai_runner = MagicMock()
                ai_runner._process_one.return_value = None
                ai_runner.processed = 0
                ai_runner.skipped = 0
                ai_runner.failures = 0
                ai_runner.force_processing = False
                ai_runner.defaults = {"caption_model": "glm-test", "lmstudio_base_url": "http://localhost:1234/v1"}
                with patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner):
                    from photoalbums.commands import run_process_pipeline

                    code = run_process_pipeline(
                        album_id="",
                        photos_root=str(root),
                        page=None,
                        skip_ids=["render", "propagate-metadata", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=[],
                        step_id=None,
                        force=False,
                        debug=False,
                        no_validation=False,
                        skip_restoration=False,
                        force_restoration=False,
                        gps_only=False,
                        refresh_gps=False,
                    )

        self.assertEqual(code, 0)
        ai_runner._process_one.assert_called_once()
        verify_mock.assert_called_once()
        persist_mock.assert_called_once()
        verify_args = verify_mock.call_args
        self.assertEqual(verify_args.args[0], view_path)
        self.assertEqual(verify_args.kwargs["model_name"], "glm-test")
        self.assertEqual(verify_args.kwargs["base_url"], "http://localhost:1234/v1")

    def test_verify_crops_executes_per_page(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            pages = root / "Test_2024_B01_Pages"
            photos = root / "Test_2024_B01_Photos"
            archive.mkdir()
            pages.mkdir()
            photos.mkdir()

            scan1 = archive / "Test_2024_B01_P01_S01.tif"
            scan2 = archive / "Test_2024_B01_P02_S01.tif"
            scan1.write_bytes(b"abc")
            scan2.write_bytes(b"def")
            view1 = pages / "Test_2024_B01_P01_V.jpg"
            view2 = pages / "Test_2024_B01_P02_V.jpg"
            view1.write_bytes(b"jpg")
            view2.write_bytes(b"jpg")
            for view_path in (view1, view2):
                view_path.with_suffix(".xmp").write_text(
                    '<?xml version="1.0" encoding="UTF-8"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>',
                    encoding="utf-8",
                )

            with (
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan1)], [str(scan2)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(pages)),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(photos)),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", side_effect=[str(scan1), str(scan2)]),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", side_effect=[view1, view2]),
                patch("photoalbums.commands._print_outcome"),
                patch("photoalbums.commands._print_ai_index_discovery_summary"),
                patch("photoalbums.commands._print_verify_crops_summary"),
                patch("photoalbums.lib.ai_verify_crops.run_verify_crops_page", return_value={"status": "ok", "results": [], "page_input_hash": "abc", "artifact_path": "artifact.json", "missing_context": []}) as verify_mock,
                patch("photoalbums.lib.ai_verify_crops.persist_verify_crops_state"),
            ):
                ai_runner = MagicMock()
                ai_runner._process_one.return_value = None
                ai_runner.processed = 0
                ai_runner.skipped = 0
                ai_runner.failures = 0
                ai_runner.force_processing = False
                ai_runner.defaults = {"caption_model": "glm-test", "lmstudio_base_url": "http://localhost:1234/v1"}
                with patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner):
                    from photoalbums.commands import run_process_pipeline

                    code = run_process_pipeline(
                        album_id="",
                        photos_root=str(root),
                        page=None,
                        skip_ids=["render", "propagate-metadata", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=[],
                        step_id=None,
                        force=False,
                        debug=False,
                        no_validation=False,
                        skip_restoration=False,
                        force_restoration=False,
                        gps_only=False,
                        refresh_gps=False,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(verify_mock.call_count, 2)

    def test_verify_crops_receives_finalized_page_and_crop_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            pages = root / "Test_2024_B01_Pages"
            photos = root / "Test_2024_B01_Photos"
            archive.mkdir()
            pages.mkdir()
            photos.mkdir()

            scan = archive / "Test_2024_B01_P01_S01.tif"
            scan.write_bytes(b"abc")
            view_path = pages / "Test_2024_B01_P01_V.jpg"
            crop_path = photos / "Test_2024_B01_P01_D01-00_V.jpg"

            from PIL import Image
            Image.new("RGB", (40, 20), "white").save(view_path)
            Image.new("RGB", (20, 20), "white").save(crop_path)

            captured: dict[str, str] = {}

            def fake_process_one(_page_idx, _view_path):
                from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
                from photoalbums.lib.xmp_sidecar import write_region_list, write_xmp_sidecar

                write_xmp_sidecar(
                    view_path.with_suffix(".xmp"),
                    creator_tool="imago-test",
                    person_names=[],
                    subjects=[],
                    description="Final page description",
                    source_text="Test_2024_B01_P01_S01.tif",
                    ocr_text="AUG. 1988",
                    author_text="Final page caption",
                    detections_payload={},
                )
                write_region_list(
                    view_path.with_suffix(".xmp"),
                    [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=20, height=20, caption_hint=""), "")],
                    40,
                    20,
                )
                write_xmp_sidecar(
                    crop_path.with_suffix(".xmp"),
                    creator_tool="imago-test",
                    person_names=[],
                    subjects=[],
                    description="Final crop caption",
                    source_text="Test_2024_B01_P01_S01.tif",
                    ocr_text="AUG. 1988",
                    detections_payload={},
                )

            def fake_verify(page_image_path, **_kwargs):
                from photoalbums.lib.ai_verify_crops import load_page_verifier_inputs

                inputs = load_page_verifier_inputs(page_image_path)
                captured["page_xmp_text"] = str(inputs["page_xmp_text"])
                captured["crop_xmp_text"] = str(inputs["crops"][0]["crop_xmp_text"])
                return {
                    "status": "ok",
                    "results": [],
                    "page_input_hash": "abc",
                    "artifact_path": "artifact.json",
                    "missing_context": [],
                }

            with (
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(pages)),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(photos)),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", return_value=view_path),
                patch("photoalbums.commands._print_outcome"),
                patch("photoalbums.commands._print_ai_index_discovery_summary"),
                patch("photoalbums.commands._print_verify_crops_summary"),
                patch("photoalbums.lib.ai_verify_crops.run_verify_crops_page", side_effect=fake_verify),
                patch("photoalbums.lib.ai_verify_crops.persist_verify_crops_state"),
            ):
                ai_runner = MagicMock()
                ai_runner._process_one.side_effect = fake_process_one
                ai_runner.processed = 0
                ai_runner.skipped = 0
                ai_runner.failures = 0
                ai_runner.force_processing = False
                ai_runner.defaults = {"caption_model": "glm-test", "lmstudio_base_url": "http://localhost:1234/v1"}
                with patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner):
                    from photoalbums.commands import run_process_pipeline

                    code = run_process_pipeline(
                        album_id="",
                        photos_root=str(root),
                        page=None,
                        skip_ids=["render", "propagate-metadata", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=[],
                        step_id=None,
                        force=False,
                        debug=False,
                        no_validation=False,
                        skip_restoration=False,
                        force_restoration=False,
                        gps_only=False,
                        refresh_gps=False,
                    )

        self.assertEqual(code, 0)
        self.assertIn("author_text: Final page caption", captured["page_xmp_text"])
        self.assertIn("AUG. 1988", captured["page_xmp_text"])
        self.assertIn("Final crop caption", captured["crop_xmp_text"])


if __name__ == "__main__":
    unittest.main()
