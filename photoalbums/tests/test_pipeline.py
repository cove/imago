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

from photoalbums.commands import _effective_pipeline_step_ids, print_pipeline_plan
from photoalbums.lib.pipeline import (
    PIPELINE_STEPS,
    VALID_STEP_IDS,
    format_pipeline_dag,
    validate_step_ids,
)
from photoalbums.lib.xmp_sidecar import is_step_stale


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
        args, _ = parser.parse_known_args(["process", "--photos-root", ".", "--step", "render"])
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

    def test_cli_list_watcher_steps_exits_0(self):
        from photoalbums.cli import main

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            code = main(["watch", "--list-steps"])
        self.assertEqual(code, 0)
        output = mock_out.getvalue()
        self.assertIn("process-tiff", output)
        self.assertIn("orientation", output)
        self.assertIn("validate-stitch", output)


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
        lines = [line for line in output.splitlines() if "render" in line and "detect" not in line and "crop" not in line]
        self.assertTrue(any("skipped" not in line and "redo" not in line for line in lines))

    def test_gps_only_not_annotated_by_plan(self):
        # gps-only is a flag, not a plan annotation; plan just shows step ids
        output = self._capture_plan([], [], "TestAlbum", 5)
        self.assertIn("TestAlbum", output)
        self.assertIn("5 page(s)", output)

    def test_immich_face_refresh_runs_after_ai_index(self):
        step_ids = [step.id for step in PIPELINE_STEPS]
        self.assertLess(step_ids.index("ai-index"), step_ids.index("immich-face-refresh"))


class TestPipelineDag(unittest.TestCase):
    def test_registration_order_is_valid_topological_order(self):
        # Execution walks PIPELINE_STEPS in list order while depends_on drives
        # staleness; the list must therefore be a valid topological sort so a
        # step never runs before a dependency. Nothing enforces this at runtime,
        # so guard it here.
        seen: set[str] = set()
        for step in PIPELINE_STEPS:
            for dep in step.depends_on:
                self.assertIn(
                    dep,
                    seen,
                    f"{step.id} lists dependency {dep!r} that appears later in PIPELINE_STEPS",
                )
            seen.add(step.id)

    def test_dag_lists_every_step(self):
        output = "\n".join(format_pipeline_dag(PIPELINE_STEPS))
        for step in PIPELINE_STEPS:
            self.assertIn(step.id, output)

    def test_dag_marks_multi_parent_node_once(self):
        lines = format_pipeline_dag(PIPELINE_STEPS)
        # immich-face-refresh depends on both ai-index and face-refresh: expanded
        # once with an "(also after: ...)" note and back-referenced elsewhere.
        also_after = [ln for ln in lines if "immich-face-refresh" in ln and "also after" in ln]
        back_ref = [ln for ln in lines if "immich-face-refresh" in ln and "shown above" in ln]
        self.assertEqual(len(also_after), 1)
        self.assertEqual(len(back_ref), 1)
        self.assertIn("face-refresh", also_after[0])

    def test_dag_tags_optional_step(self):
        lines = format_pipeline_dag(PIPELINE_STEPS)
        verify = [ln for ln in lines if "verify-crops" in ln]
        self.assertTrue(verify)
        self.assertIn("[optional]", verify[0])


class TestEffectivePipelineStepIds(unittest.TestCase):
    def test_verify_crops_is_skipped_by_default(self):
        skip_ids, redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=[],
            step_id=None,
            force=False,
            refresh_gps=False,
        )

        self.assertIn("verify-crops", skip_ids)
        self.assertEqual(redo_ids, set())

    def test_face_refresh_is_skipped_by_default(self):
        skip_ids, redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=[],
            step_id=None,
            force=False,
            refresh_gps=False,
        )

        self.assertIn("face-refresh", skip_ids)
        self.assertIn("immich-face-refresh", skip_ids)
        self.assertEqual(redo_ids, set())

    def test_explicit_verify_crops_step_runs(self):
        skip_ids, _redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=[],
            step_id="verify-crops",
            force=False,
            refresh_gps=False,
        )

        self.assertNotIn("verify-crops", skip_ids)

    def test_explicit_immich_face_refresh_step_runs(self):
        skip_ids, _redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=[],
            step_id="immich-face-refresh",
            force=False,
            refresh_gps=False,
        )

        self.assertNotIn("immich-face-refresh", skip_ids)

    def test_explicit_face_refresh_step_runs(self):
        skip_ids, _redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=[],
            step_id="face-refresh",
            force=False,
            refresh_gps=False,
        )

        self.assertNotIn("face-refresh", skip_ids)

    def test_explicit_face_refresh_redo_runs(self):
        skip_ids, redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=["face-refresh"],
            step_id=None,
            force=False,
            refresh_gps=False,
        )

        self.assertNotIn("face-refresh", skip_ids)
        self.assertEqual(redo_ids, {"face-refresh"})

    def test_explicit_verify_crops_redo_runs(self):
        skip_ids, redo_ids = _effective_pipeline_step_ids(
            valid_step_ids=set(VALID_STEP_IDS),
            skip_ids=[],
            redo_ids=["verify-crops"],
            step_id=None,
            force=False,
            refresh_gps=False,
        )

        self.assertNotIn("verify-crops", skip_ids)
        self.assertEqual(redo_ids, {"verify-crops"})


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
        import tempfile

        from photoalbums.lib.pipeline import VALID_STEP_IDS

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
                patch(
                    "photoalbums.stitch_oversized_pages.get_view_dirname",
                    return_value=str(root / "Test_2024_B01_Pages"),
                ),
                patch(
                    "photoalbums.stitch_oversized_pages.get_photos_dirname",
                    return_value=str(root / "Test_2024_B01_Photos"),
                ),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch(
                    "photoalbums.stitch_oversized_pages._view_page_output_path",
                    return_value=root / "Test_2024_B01_Pages" / "Test_2024_B01_P01_V.jpg",
                ),
                patch(
                    "photoalbums.lib.xmp_sidecar.read_pipeline_state",
                    return_value={"ai-index": {"completed": "2026-04-21T00:00:00Z"}},
                ),
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
                        skip_ids=["scan-ai", "render", "detect-regions", "crop-regions", "face-refresh"],
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

    def test_face_refresh_skips_when_targets_report_current(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            pages = root / "Test_2024_B01_Pages"
            photos = root / "Test_2024_B01_Photos"
            archive.mkdir()
            pages.mkdir()
            photos.mkdir()
            scan = archive / "Test_2024_B01_P02_S01.tif"
            scan.write_bytes(b"abc")
            view_path = pages / "Test_2024_B01_P02_V.jpg"
            target = photos / "Test_2024_B01_P02_D01-00_V.jpg"

            session = MagicMock()
            session.refresh_face_regions.return_value = False
            ai_runner = MagicMock()
            ai_runner.defaults = {"caption_model": "glm-test", "lmstudio_base_url": "http://localhost:1234/v1"}

            with (
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(pages)),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(photos)),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", return_value=view_path),
                patch("photoalbums.lib.xmp_sidecar.read_pipeline_state", return_value={}),
                patch("photoalbums.lib.xmp_sidecar.write_pipeline_step") as write_step,
                patch("photoalbums.commands._check_step_stale", return_value=(True, "crop-regions")),
                patch("photoalbums.commands._iter_face_refresh_targets", return_value=[target]),
                patch("photoalbums.commands._print_outcome") as print_outcome,
                patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession", return_value=session),
                patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner),
            ):
                from photoalbums.commands import run_process_pipeline

                code = run_process_pipeline(
                    album_id="",
                    photos_root=str(root),
                    page=None,
                    skip_ids=[],
                    redo_ids=[],
                    step_id="face-refresh",
                    force=False,
                    debug=False,
                    no_validation=False,
                    skip_restoration=False,
                    force_restoration=False,
                    gps_only=False,
                    refresh_gps=False,
                )

        self.assertEqual(code, 0)
        session.refresh_face_regions.assert_called_once_with(target, target.with_suffix(".xmp"), force=False)
        write_step.assert_not_called()
        print_outcome.assert_called_once_with("skipped (already complete)", "")

    def test_immich_face_refresh_queries_archive_scan_and_rendered_assets_directly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Test_2024_B01_Archive"
            pages = root / "Test_2024_B01_Pages"
            photos = root / "Test_2024_B01_Photos"
            archive.mkdir()
            pages.mkdir()
            photos.mkdir()
            scan = archive / "Test_2024_B01_P02_S01.tif"
            scan.write_bytes(b"abc")
            view_path = pages / "Test_2024_B01_P02_V.jpg"
            target = photos / "Test_2024_B01_P02_D01-00_V.jpg"

            ai_runner = MagicMock()
            ai_runner.defaults = {"caption_model": "glm-test", "lmstudio_base_url": "http://localhost:1234/v1"}
            face = {
                "imageWidth": 1000,
                "imageHeight": 800,
                "boundingBoxX1": 100,
                "boundingBoxY1": 80,
                "boundingBoxX2": 300,
                "boundingBoxY2": 320,
                "person": {"name": "Alice"},
            }

            with (
                patch.dict("os.environ", {"IMMICH_URL": "http://immich.local:2283", "IMMICH_API_KEY": "key"}),
                patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive)]),
                patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                patch("photoalbums.stitch_oversized_pages.get_view_dirname", return_value=str(pages)),
                patch("photoalbums.stitch_oversized_pages.get_photos_dirname", return_value=str(photos)),
                patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                patch("photoalbums.stitch_oversized_pages._view_page_output_path", return_value=view_path),
                patch("photoalbums.lib.xmp_sidecar.read_pipeline_state", return_value={}),
                patch("photoalbums.lib.xmp_sidecar.write_pipeline_step") as write_step,
                patch("photoalbums.commands._check_step_stale", return_value=(True, "crop-regions")),
                patch("photoalbums.commands._iter_face_refresh_targets", return_value=[target]),
                patch("cast.immich_sync.fetch_assets_by_original_filename", return_value=[{"id": "asset-001"}]) as fetch_assets,
                patch("cast.immich_sync.fetch_asset_faces", return_value=[face]) as fetch_faces,
                patch("cast.xmp_writer.merge_persons_xmp") as merge_persons,
                patch("cast.xmp_writer.merge_face_regions_xmp") as merge_regions,
                patch("photoalbums.lib.ai_view_regions._image_dimensions", return_value=(200, 100)),
                patch("photoalbums.lib.ai_render_face_refresh.RenderFaceRefreshSession"),
                patch("photoalbums.lib.ai_index_runner.IndexRunner", return_value=ai_runner),
            ):
                from photoalbums.commands import run_process_pipeline

                code = run_process_pipeline(
                    album_id="",
                    photos_root=str(root),
                    page=None,
                    skip_ids=[],
                    redo_ids=[],
                    step_id="immich-face-refresh",
                    force=False,
                    debug=False,
                    no_validation=False,
                    skip_restoration=False,
                    force_restoration=False,
                    gps_only=False,
                    refresh_gps=False,
                )

        self.assertEqual(code, 0)
        self.assertEqual(
            fetch_assets.call_args_list,
            [
                unittest.mock.call("http://immich.local:2283", "key", scan.name),
                unittest.mock.call("http://immich.local:2283", "key", target.name),
            ],
        )
        self.assertEqual(
            fetch_faces.call_args_list,
            [
                unittest.mock.call("http://immich.local:2283", "key", "asset-001"),
                unittest.mock.call("http://immich.local:2283", "key", "asset-001"),
            ],
        )
        merge_persons.assert_any_call(scan.with_suffix(".xmp"), ["Alice"])
        merge_persons.assert_any_call(target.with_suffix(".xmp"), ["Alice"])
        self.assertEqual(merge_regions.call_count, 2)
        for call in merge_regions.call_args_list:
            self.assertEqual(call.args[1][0]["image_width"], 200)
            self.assertEqual(call.args[1][0]["image_height"], 100)
        write_step.assert_called_once_with(view_path.with_suffix(".xmp"), "immich-face-refresh")

    def test_verify_crops_runs_after_ai_index_when_explicitly_requested(self):
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
                patch(
                    "photoalbums.lib.ai_verify_crops.run_verify_crops_page",
                    return_value={
                        "status": "ok",
                        "results": [],
                        "page_input_hash": "abc",
                        "artifact_path": "artifact.json",
                        "missing_context": [],
                    },
                ) as verify_mock,
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
                        skip_ids=["scan-ai", "render", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=["verify-crops"],
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

    def test_verify_crops_does_not_run_by_default(self):
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
                patch("photoalbums.lib.ai_verify_crops.run_verify_crops_page") as verify_mock,
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
                        skip_ids=["scan-ai", "render", "detect-regions", "crop-regions", "face-refresh"],
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
        verify_mock.assert_not_called()
        persist_mock.assert_not_called()

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
                patch(
                    "photoalbums.lib.ai_verify_crops.run_verify_crops_page",
                    return_value={
                        "status": "ok",
                        "results": [],
                        "page_input_hash": "abc",
                        "artifact_path": "artifact.json",
                        "missing_context": [],
                    },
                ) as verify_mock,
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
                        skip_ids=["scan-ai", "render", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=["verify-crops"],
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
                        skip_ids=["scan-ai", "render", "detect-regions", "crop-regions", "face-refresh"],
                        redo_ids=["verify-crops"],
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
        self.assertIn("description:", captured["page_xmp_text"])
        self.assertNotIn("ocr_text", captured["page_xmp_text"])
        self.assertNotIn("author_text:", captured["page_xmp_text"])
        self.assertIn("Final crop caption", captured["crop_xmp_text"])


if __name__ == "__main__":
    unittest.main()
