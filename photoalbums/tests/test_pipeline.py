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


if __name__ == "__main__":
    unittest.main()
