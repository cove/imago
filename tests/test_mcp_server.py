"""Tests for MCP server photoalbums helpers."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_server
from photoalbums.lib import xmp_sidecar


class TestPhotoalbumsAiIndexPhotoResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.photos_root = Path(self.tmp.name) / "photos"
        self.photos_root.mkdir(parents=True)

        self._orig_runner = mcp_server.runner
        self._orig_photos_root = mcp_server.PHOTOS_ROOT_DEFAULT
        self.runner = mock.Mock()
        self.runner.start.return_value = "job123"
        mcp_server.runner = self.runner
        mcp_server.PHOTOS_ROOT_DEFAULT = str(self.photos_root)

    def tearDown(self):
        mcp_server.runner = self._orig_runner
        mcp_server.PHOTOS_ROOT_DEFAULT = self._orig_photos_root
        self.tmp.cleanup()

    def _started_args(self) -> list[str]:
        self.assertTrue(self.runner.start.called)
        return self.runner.start.call_args.args[1]

    def test_photoalbums_ai_index_passes_through_full_photo_path(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(photo=str(image_path))

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path))

    def test_photoalbums_ai_index_resolves_bare_filename(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(photo="Photo_01.jpg")

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path.resolve()))

    def test_photoalbums_ai_index_raises_when_filename_not_found(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(photo="Missing.jpg")

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
            mcp_server.photoalbums_ai_index(photo="Photo_01.jpg")

        self.assertIn("ambiguous", str(exc.exception))
        self.runner.start.assert_not_called()

    def test_photoalbums_ai_index_only_passes_supported_filters(self):
        mcp_server.photoalbums_ai_index(
            album="Album_A",
            max_images=5,
            process_all_photos=True,
        )

        args = self._started_args()
        self.assertIn("--force", args)
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


class TestPhotoalbumsLoadXmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.photos_root = Path(self.tmp.name) / "photos"
        self.photos_root.mkdir(parents=True)

        self._orig_photos_root = mcp_server.PHOTOS_ROOT_DEFAULT
        mcp_server.PHOTOS_ROOT_DEFAULT = str(self.photos_root)

    def tearDown(self):
        mcp_server.PHOTOS_ROOT_DEFAULT = self._orig_photos_root
        self.tmp.cleanup()

    def _write_sidecar(self, image_path: Path) -> Path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        sidecar_path = image_path.with_suffix(".xmp")
        xmp_sidecar.write_xmp_sidecar(
            sidecar_path,
            creator_tool="https://github.com/cove/imago",
            person_names=["Alice Example"],
            subjects=["park", "bench"],
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
            ocr_ran=True,
            people_detected=True,
            people_identified=True,
        )
        return sidecar_path

    def test_photoalbums_load_xmp_loads_explicit_sidecar(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        sidecar_path = self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(file_name="Photo_01.xmp")

        self.assertEqual(result["resolved_from"], "xmp_path")
        self.assertIsNone(result["photo_path"])
        self.assertEqual(result["sidecar_path"], str(sidecar_path.resolve()))
        self.assertEqual(result["creator_tool"], "https://github.com/cove/imago")
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
        self.assertEqual(result["subphotos"][0]["bounds"]["width"], 3)
        self.assertEqual(result["subphotos"][0]["people"], ["Alice Example"])

    def test_photoalbums_load_xmp_resolves_photo_filename(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        sidecar_path = self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(file_name="Photo_01.jpg")

        self.assertEqual(result["resolved_from"], "photo")
        self.assertEqual(result["photo_path"], str(image_path.resolve()))
        self.assertEqual(result["sidecar_path"], str(sidecar_path.resolve()))

    def test_photoalbums_load_xmp_includes_raw_xml_when_requested(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        sidecar_path = self._write_sidecar(image_path)

        result = mcp_server.photoalbums_load_xmp(
            file_name="Photo_01.xmp",
            include_raw_xml=True,
        )

        self.assertIn("<x:xmpmeta", result["raw_xml"])
        self.assertIn("Alice Example", result["raw_xml"])

    def test_photoalbums_load_xmp_raises_when_sidecar_missing_for_photo(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_load_xmp(file_name="Photo_01.jpg")

        self.assertIn("No XMP sidecar", str(exc.exception))

    def test_photoalbums_load_xmp_requires_file_name(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_load_xmp(file_name="")

        self.assertIn("file_name", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
