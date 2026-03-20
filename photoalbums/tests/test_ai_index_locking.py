import io
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

from photoalbums.lib import ai_index


class TestAIIndexLocking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _single_photo_args(self, image_path: Path, manifest_path: Path) -> list[str]:
        return [
            "--photos-root",
            str(self.root),
            "--cast-store",
            str(self.root / "cast"),
            "--manifest",
            str(manifest_path),
            "--photo",
            str(image_path),
            "--disable-people",
            "--disable-objects",
            "--ocr-engine",
            "none",
            "--caption-engine",
            "none",
            "--ignore-render-settings",
        ]

    def _batch_args(self, manifest_path: Path) -> list[str]:
        return [
            "--photos-root",
            str(self.root),
            "--cast-store",
            str(self.root / "cast"),
            "--manifest",
            str(manifest_path),
            "--disable-people",
            "--disable-objects",
            "--ocr-engine",
            "none",
            "--caption-engine",
            "none",
            "--ignore-render-settings",
        ]

    def test_image_processing_lock_rejects_active_lock(self):
        image_path = self.root / "Photo_01.jpg"
        image_path.touch()

        lock_path = ai_index._acquire_image_processing_lock(image_path)
        try:
            with self.assertRaises(RuntimeError) as exc:
                ai_index._acquire_image_processing_lock(image_path)
            self.assertIn("already processing", str(exc.exception))
        finally:
            ai_index._release_image_processing_lock(lock_path)

    def test_image_processing_lock_clears_stale_lock(self):
        image_path = self.root / "Photo_01.jpg"
        image_path.touch()
        lock_path = ai_index._processing_lock_path(image_path)
        lock_path.write_text('{"pid": 999999, "job_id": "old"}', encoding="utf-8")

        with mock.patch.object(ai_index, "_pid_alive", return_value=False):
            acquired = ai_index._acquire_image_processing_lock(image_path)
        try:
            self.assertEqual(acquired, lock_path)
            payload = ai_index._read_processing_lock(lock_path)
            self.assertEqual(payload["pid"], ai_index.os.getpid())
        finally:
            ai_index._release_image_processing_lock(acquired)

    def test_run_returns_error_when_single_photo_is_locked(self):
        image_path = self.root / "Photo_01.jpg"
        image_path.touch()
        manifest_path = self.root / "manifest.jsonl"
        lock_path = ai_index._acquire_image_processing_lock(image_path)
        try:
            with mock.patch("sys.stdout", new=io.StringIO()), mock.patch("sys.stderr", new=io.StringIO()):
                result = ai_index.run(self._single_photo_args(image_path, manifest_path))
            self.assertEqual(result, 1)
        finally:
            ai_index._release_image_processing_lock(lock_path)

    def test_run_returns_error_when_batch_lock_exists(self):
        view_dir = self.root / "Album_View"
        view_dir.mkdir(parents=True)
        (view_dir / "Photo_01.jpg").touch()
        manifest_path = self.root / "manifest.jsonl"
        batch_lock_path = ai_index._acquire_batch_processing_lock(self.root)
        try:
            stdout = io.StringIO()
            with mock.patch("sys.stdout", new=stdout), mock.patch("sys.stderr", new=io.StringIO()):
                result = ai_index.run(self._batch_args(manifest_path))
            self.assertEqual(result, 1)
            self.assertIn("another photoalbums ai batch run is already active", stdout.getvalue())
        finally:
            ai_index._release_batch_processing_lock(batch_lock_path)


if __name__ == "__main__":
    unittest.main()
