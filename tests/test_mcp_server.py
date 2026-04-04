"""Tests for MCP server photoalbums helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_server
from photoalbums.lib import album_sets, xmp_sidecar


class AlbumSetConfigMixin:
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.photos_root = self.root / "photos"
        self.photos_root.mkdir(parents=True)
        self.scan_root = self.root / "incoming"
        self.scan_root.mkdir(parents=True)
        self.manifest_path = self.root / "ai_index_manifest.jsonl"
        self.cast_store = self.root / "cast"
        self.cast_store.mkdir(parents=True)
        self.album_sets_path = self.root / "album_sets.toml"

        self._orig_album_sets_path = album_sets.ALBUM_SETS_PATH
        self._orig_scanwatch_services = mcp_server.scanwatch_services

        album_sets.ALBUM_SETS_PATH = self.album_sets_path
        album_sets.load_album_sets.cache_clear()
        mcp_server.scanwatch_services = {}
        self._write_album_sets_config()

    def tearDown(self) -> None:
        for service in mcp_server.scanwatch_services.values():
            service.stop()
        mcp_server.scanwatch_services = self._orig_scanwatch_services
        album_sets.ALBUM_SETS_PATH = self._orig_album_sets_path
        album_sets.load_album_sets.cache_clear()
        self.tmp.cleanup()
        super().tearDown()

    def _write_album_sets_config(self) -> None:
        content = "\n".join(
            [
                'default_archive_set = "cordell"',
                'default_scan_set = "incoming_scans"',
                "",
                "[sets.cordell]",
                'kind = "archive"',
                'description = "Cordell family photo albums archive"',
                f'photos_root = "{self.photos_root.as_posix()}"',
                f'manifest_path = "{self.manifest_path.as_posix()}"',
                f'cast_store = "{self.cast_store.as_posix()}"',
                'skill = "cordell-photo-albums"',
                "",
                "[sets.incoming_scans]",
                'kind = "scanwatch"',
                'description = "Incoming scan intake workspace"',
                f'photos_root = "{self.scan_root.as_posix()}"',
                "",
            ]
        )
        self.album_sets_path.write_text(content, encoding="utf-8")


class TestPhotoalbumsAlbumSets(AlbumSetConfigMixin, unittest.TestCase):
    def test_photoalbums_list_sets_returns_client_safe_metadata(self):
        result = mcp_server.photoalbums_list_sets()

        self.assertEqual([row["album_set"] for row in result], ["cordell", "incoming_scans"])
        cordell = result[0]
        incoming = result[1]
        self.assertEqual(cordell["kind"], "archive")
        self.assertTrue(cordell["is_default"])
        self.assertIn("photoalbums_ai_index", cordell["supported_tools"])
        self.assertNotIn("photos_root", cordell)
        self.assertEqual(incoming["kind"], "scanwatch")
        self.assertTrue(incoming["is_default"])
        self.assertIn("scanwatch_status", incoming["supported_tools"])
        self.assertNotIn("manifest_path", incoming)

    def test_photoalbums_list_sets_can_filter_by_kind(self):
        result = mcp_server.photoalbums_list_sets(kind="scanwatch")

        self.assertEqual(result, [mcp_server.photoalbums_get_set("incoming_scans")])

    def test_photoalbums_list_sets_rejects_unknown_kind(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_list_sets(kind="unknown")

        self.assertIn("kind must be one of", str(exc.exception))

    def test_scanwatch_status_uses_default_scan_set(self):
        result = mcp_server.scanwatch_status()

        self.assertEqual(result["album_set"], "incoming_scans")
        self.assertFalse(result["running"])
        self.assertEqual(result["event_count"], 0)


class TestPhotoalbumsAiIndexPhotoResolution(AlbumSetConfigMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._orig_runner = mcp_server.runner
        self.runner = mock.Mock()
        self.runner.start.return_value = "job123"
        mcp_server.runner = self.runner

    def tearDown(self) -> None:
        mcp_server.runner = self._orig_runner
        super().tearDown()

    def _started_args(self) -> list[str]:
        self.assertTrue(self.runner.start.called)
        return self.runner.start.call_args.args[1]

    def test_photoalbums_ai_index_passes_through_full_photo_path(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(album_set="cordell", photo=str(image_path))

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path))

    def test_photoalbums_ai_index_resolves_bare_filename(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(album_set="cordell", photo="Photo_01.jpg")

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path.resolve()))

    def test_photoalbums_ai_index_raises_when_filename_not_found(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(album_set="cordell", photo="Missing.jpg")

        self.assertIn("was not found", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_raises_when_filename_is_ambiguous(self):
        first = self.photos_root / "Album_A" / "Photo_01.jpg"
        second = self.photos_root / "Album_B" / "Photo_01.jpg"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.touch()
        second.touch()

        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(album_set="cordell", photo="Photo_01.jpg")

        self.assertIn("ambiguous", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_rejects_photo_lists(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(album_set="cordell", photo=["Photo_01.jpg"])  # type: ignore[arg-type]

        self.assertIn("single filename or path string", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_only_passes_supported_filters(self):
        mcp_server.photoalbums_ai_index(
            album_set="cordell",
            album="Album_A",
            max_images=5,
            reprocess_mode="all",
        )

        args = self._started_args()
        self.assertEqual(args[args.index("--reprocess-mode") + 1], "all")
        self.assertEqual(args[args.index("--album") + 1], "Album_A")
        self.assertEqual(args[args.index("--max-images") + 1], "5")
        self.assertNotIn("--caption-engine", args)
        self.assertNotIn("--ocr-engine", args)
        self.assertNotIn("--disable-people", args)
        self.assertNotIn("--disable-objects", args)
        self.assertNotIn("--disable-ocr", args)
        self.assertNotIn("--include-view", args)
        self.assertNotIn("--photo-offset", args)
        self.assertNotIn("--dry-run", args)

    def test_photoalbums_ai_index_starts_sharded_workers(self):
        self.runner.start.side_effect = ["job_a", "job_b"]

        result = mcp_server.photoalbums_ai_index(
            album_set="cordell",
            album="Album_A",
            workers=2,
            lmstudio_base_urls=["http://lm-a:1234/v1", "http://lm-b:1234/v1"],
        )

        self.assertEqual(self.runner.start.call_count, 2)
        first_name, first_args = self.runner.start.call_args_list[0].args[:2]
        second_name, second_args = self.runner.start.call_args_list[1].args[:2]
        self.assertEqual(first_name, "photoalbums_ai_index:cordell[1/2]")
        self.assertEqual(second_name, "photoalbums_ai_index:cordell[2/2]")
        self.assertEqual(first_args[first_args.index("--shard-count") + 1], "2")
        self.assertEqual(first_args[first_args.index("--shard-index") + 1], "0")
        self.assertEqual(second_args[second_args.index("--shard-index") + 1], "1")
        self.assertEqual(first_args[first_args.index("--lmstudio-base-url") + 1], "http://lm-a:1234/v1")
        self.assertEqual(second_args[second_args.index("--lmstudio-base-url") + 1], "http://lm-b:1234/v1")
        self.assertEqual(result["child_job_ids"], ["job_a", "job_b"])
        self.assertEqual(result["workers"], 2)
        self.assertEqual(result["warning"], mcp_server.PHOTOALBUMS_MULTI_WORKER_DEPRECATION)

    def test_photoalbums_ai_index_defaults_sharded_workers_to_configured_urls(self):
        self.runner.start.side_effect = ["job_a", "job_b"]

        result = mcp_server.photoalbums_ai_index(
            album_set="cordell",
            album="Album_A",
            workers=2,
        )

        self.assertEqual(self.runner.start.call_count, 2)
        first_args = self.runner.start.call_args_list[0].args[1]
        second_args = self.runner.start.call_args_list[1].args[1]
        self.assertEqual(first_args[first_args.index("--lmstudio-base-url") + 1], "http://192.168.4.72:1234")
        self.assertEqual(second_args[second_args.index("--lmstudio-base-url") + 1], "http://192.168.4.21:1234")
        self.assertEqual(result["child_job_ids"], ["job_a", "job_b"])

    def test_photoalbums_ai_index_rejects_worker_photo_mode(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(album_set="cordell", photo=str(image_path), workers=2)

        self.assertIn("workers > 1", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_rejects_mismatched_worker_urls(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(
                album_set="cordell",
                album="Album_A",
                workers=2,
                lmstudio_base_urls=["http://lm-a:1234/v1", "http://lm-b:1234/v1", "http://lm-c:1234/v1"],
            )

        self.assertIn("lmstudio_base_urls", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_rejects_scanwatch_set(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(album_set="incoming_scans")

        self.assertIn("does not support archive operations", str(exc.exception))
        self.runner.start.assert_not_called()


class TestPhotoalbumsLoadXmp(AlbumSetConfigMixin, unittest.TestCase):
    def _write_sidecar(self, image_path: Path) -> Path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        sidecar_path = image_path.with_suffix(".xmp")
        xmp_sidecar.write_xmp_sidecar(
            sidecar_path,
            creator_tool="https://github.com/cove/imago",
            person_names=["Alice Example"],
            subjects=["park", "bench"],
            title="Album page caption",
            description="Alice Example sitting on a bench in the park.",
            album_title="Family Book I",
            source_text="scan_001.tif",
            ocr_text="FAMILY BOOK",
            ocr_authority_source="archive_stitched",
            detections_payload={
                "people": [
                    {
                        "name": "Alice Example",
                        "bbox": [10, 20, 30, 40],
                        "score": 0.98,
                    }
                ],
                "objects": [{"label": "bench", "score": 0.81}],
                "ocr": {"chars": 11},
                "caption": {"effective_engine": "template"},
            },
            subphotos=[
                {
                    "index": 1,
                    "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "description": "Inset photo",
                    "ocr_text": "Inset",
                    "people": ["Alice Example"],
                    "subjects": ["bench"],
                    "detections": {"objects": [{"label": "bench", "score": 0.81}]},
                }
            ],
            stitch_key="Family_1986_B01_P01",
            image_width=100,
            image_height=100,
            ocr_ran=True,
            people_detected=True,
            people_identified=True,
        )
        return sidecar_path

    def test_photoalbums_load_xmp_loads_explicit_sidecar(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        sidecar_path = self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(file_name="Photo_01.xmp", album_set="cordell")

        self.assertEqual(result["resolved_from"], "xmp_path")
        self.assertIsNone(result["photo_path"])
        self.assertEqual(result["sidecar_path"], str(sidecar_path.resolve()))
        self.assertEqual(result["creator_tool"], "https://github.com/cove/imago")
        self.assertEqual(result["title"], "Album page caption")
        self.assertEqual(result["person_names"], ["Alice Example"])
        self.assertEqual(result["subjects"], ["park", "bench"])
        self.assertEqual(result["source_text"], "scan_001.tif")
        self.assertEqual(result["ocr_authority_source"], "archive_stitched")
        self.assertEqual(result["stitch_key"], "Family_1986_B01_P01")
        self.assertEqual(result["summary"]["people_in_image_count"], 1)
        self.assertEqual(result["summary"]["detected_people_count"], 1)
        self.assertEqual(result["summary"]["detected_object_count"], 1)
        self.assertEqual(result["summary"]["ocr_char_count"], 11)
        self.assertEqual(result["summary"]["subphoto_count"], 1)
        self.assertEqual(result["subphotos"][0]["bounds"]["width"], 0.03)
        self.assertEqual(result["subphotos"][0]["people"], ["Alice Example"])

    def test_photoalbums_load_xmp_resolves_photo_filename(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        sidecar_path = self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(file_name="Photo_01.jpg", album_set="cordell")

        self.assertEqual(result["resolved_from"], "photo")
        self.assertEqual(result["photo_path"], str(image_path.resolve()))
        self.assertEqual(result["sidecar_path"], str(sidecar_path.resolve()))

    def test_photoalbums_load_xmp_includes_raw_xml_when_requested(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(
            file_name="Photo_01.xmp",
            album_set="cordell",
            include_raw_xml=True,
        )

        self.assertIn("<x:xmpmeta", result["raw_xml"])
        self.assertIn("Alice Example", result["raw_xml"])

    def test_photoalbums_load_xmp_raises_when_sidecar_missing_for_photo(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_load_xmp(file_name="Photo_01.jpg", album_set="cordell")

        self.assertIn("No XMP sidecar", str(exc.exception))

    def test_photoalbums_load_xmp_requires_file_name(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_load_xmp(file_name="", album_set="cordell")

        self.assertIn("file_name", str(exc.exception))


class TestPhotoalbumsMcpQueries(AlbumSetConfigMixin, unittest.TestCase):
    def _write_manifest(self, *rows: dict) -> None:
        payload = "\n".join(json.dumps(row) for row in rows)
        self.manifest_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def _write_cast_store(self) -> None:
        (self.cast_store / "people.json").write_text('{"people":[]}', encoding="utf-8")
        (self.cast_store / "faces.json").write_text("[]", encoding="utf-8")
        (self.cast_store / "review_queue.jsonl").write_text("", encoding="utf-8")

    def _write_sidecar(
        self,
        image_path: Path,
        *,
        album_title: str = "",
        ocr_authority_source: str = "",
        people_detected: bool = False,
    ) -> Path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        sidecar_path = image_path.with_suffix(".xmp")
        xmp_sidecar.write_xmp_sidecar(
            sidecar_path,
            creator_tool="https://github.com/cove/imago",
            person_names=["Alice Example"] if people_detected else [],
            subjects=["park"],
            title="Album page caption",
            description="Album page caption",
            album_title=album_title,
            source_text="scan_001.tif",
            ocr_text="FAMILY BOOK",
            ocr_authority_source=ocr_authority_source,
            detections_payload={"caption": {"effective_engine": "template"}},
            ocr_ran=True,
            people_detected=people_detected,
            people_identified=people_detected,
        )
        return sidecar_path

    def test_photoalbums_manifest_query_returns_derived_fields(self):
        image_path = self.photos_root / "Album_Query_Archive" / "Family_1986_B01_P00_S01.tif"
        sidecar_path = self._write_sidecar(image_path, album_title="Family Book I")
        self._write_manifest(
            {
                "image_path": str(image_path.resolve()),
                "sidecar_path": str(sidecar_path.resolve()),
                "state": "done",
                "cast_store_signature": "sig-1",
            }
        )

        result = mcp_server.photoalbums_manifest_query(album_set="cordell", album="Album_Query", state="done")

        self.assertEqual(result["total_matches"], 1)
        row = result["rows"][0]
        self.assertEqual(row["album_dir"], "Album_Query_Archive")
        self.assertEqual(row["file_name"], image_path.name)
        self.assertTrue(row["is_cover_candidate"])
        self.assertTrue(row["sidecar_present"])
        self.assertEqual(row["album_title"], "Family Book I")

    def test_photoalbums_album_status_reports_cover_ready(self):
        cover_path = self.photos_root / "Album_Status_Archive" / "Family_1986_B01_P00_S01.tif"
        page_path = self.photos_root / "Album_Status_Archive" / "Family_1986_B01_P02_S01.tif"
        self._write_sidecar(cover_path, album_title="Family Book I")
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.touch()

        result = mcp_server.photoalbums_album_status(album="Album_Status", album_set="cordell")

        self.assertEqual(result["total_images"], 2)
        self.assertTrue(result["cover_ready"])
        self.assertEqual(result["matched_parent_dirs"], ["Album_Status_Archive"])
        self.assertEqual(len(result["cover_candidates"]), 1)
        self.assertEqual(result["cover_candidates"][0]["album_title"], "Family Book I")

    def test_photoalbums_job_artifacts_filters_by_kind_and_file_name(self):
        artifact_file = self.root / "job.artifacts.jsonl"
        artifact_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "kind": "photoalbums_xmp",
                            "image_path": str(self.photos_root / "Album_A" / "Photo_01.jpg"),
                            "sidecar_path": str(self.photos_root / "Album_A" / "Photo_01.xmp"),
                            "label": "Photo_01.jpg",
                        }
                    ),
                    json.dumps(
                        {
                            "kind": "photoalbums_prompts",
                            "image_path": str(self.photos_root / "Album_A" / "Photo_01.jpg"),
                            "label": "Photo_01.jpg",
                            "steps": [],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        jobs = [{"id": "job123", "name": "photoalbums_ai_index:cordell", "artifact_file": str(artifact_file)}]

        with mock.patch.object(mcp_server.runner, "list_jobs", return_value=jobs):
            result = mcp_server.photoalbums_job_artifacts(
                job_id="job123",
                kind="photoalbums_xmp",
                file_name="Photo_01.jpg",
            )

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["artifacts"][0]["kind"], "photoalbums_xmp")

    def test_photoalbums_reprocess_audit_reports_stitched_and_cast_reasons(self):
        self._write_cast_store()

        stitched_one = self.photos_root / "Audit_Archive" / "Family_1986_B01_P03_S01.tif"
        stitched_two = self.photos_root / "Audit_Archive" / "Family_1986_B01_P03_S02.tif"
        stale_people = self.photos_root / "Audit_Archive" / "Family_1986_B01_P04_S01.tif"

        stitched_sidecar = self._write_sidecar(stitched_one)
        self._write_sidecar(stitched_two)
        stale_sidecar = self._write_sidecar(stale_people, people_detected=True)

        self._write_manifest(
            {
                "image_path": str(stitched_one.resolve()),
                "sidecar_path": str(stitched_sidecar.resolve()),
                "cast_store_signature": "unchanged",
            },
            {
                "image_path": str(stitched_two.resolve()),
                "sidecar_path": str(stitched_two.with_suffix(".xmp").resolve()),
                "cast_store_signature": "unchanged",
            },
            {
                "image_path": str(stale_people.resolve()),
                "sidecar_path": str(stale_sidecar.resolve()),
                "cast_store_signature": "old-cast-signature",
            },
        )

        result = mcp_server.photoalbums_reprocess_audit(album_set="cordell", album="Audit", limit=10)

        reasons_by_name = {row["file_name"]: set(row["reprocess_reasons"]) for row in result["rows"]}
        self.assertIn("missing_stitched_authority", reasons_by_name[stitched_one.name])
        self.assertIn("missing_stitched_authority", reasons_by_name[stitched_two.name])
        self.assertIn("cast_store_signature_changed", reasons_by_name[stale_people.name])


if __name__ == "__main__":
    unittest.main()
