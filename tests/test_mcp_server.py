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


class TestPhotoalbumsAiIndexPhotoResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.photos_root = Path(self.tmp.name) / "photos"
        self.photos_root.mkdir(parents=True)

        self._orig_runner = mcp_server.runner
        self.runner = mock.Mock()
        self.runner.start.return_value = "job123"
        mcp_server.runner = self.runner

    def tearDown(self):
        mcp_server.runner = self._orig_runner
        self.tmp.cleanup()

    def _started_args(self) -> list[str]:
        self.assertTrue(self.runner.start.called)
        return self.runner.start.call_args.args[1]

    def test_photoalbums_ai_index_passes_through_full_photo_path(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(
            photos_root=str(self.photos_root),
            photo=str(image_path),
        )

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path))

    def test_photoalbums_ai_index_resolves_bare_filename(self):
        image_path = self.photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        mcp_server.photoalbums_ai_index(
            photos_root=str(self.photos_root),
            photo="Photo_01.jpg",
        )

        args = self._started_args()
        self.assertEqual(args[args.index("--photo") + 1], str(image_path.resolve()))

    def test_photoalbums_ai_index_raises_when_filename_not_found(self):
        with self.assertRaises(ValueError) as exc:
            mcp_server.photoalbums_ai_index(
                photos_root=str(self.photos_root),
                photo="Missing.jpg",
            )

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
            mcp_server.photoalbums_ai_index(
                photos_root=str(self.photos_root),
                photo="Photo_01.jpg",
            )

        self.assertIn("ambiguous", str(exc.exception))
        self.runner.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
