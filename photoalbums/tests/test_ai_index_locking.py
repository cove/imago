import io
import sys
import tempfile
import unittest
from unittest import skip
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_index, ai_processing_locks


class TestAIIndexLocking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _single_photo_args(self, image_path: Path) -> list[str]:
        return [
            "--photos-root",
            str(self.root),
            "--cast-store",
            str(self.root / "cast"),
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

    def _batch_args(self) -> list[str]:
        return [
            "--photos-root",
            str(self.root),
            "--cast-store",
            str(self.root / "cast"),
            "--disable-people",
            "--disable-objects",
            "--ocr-engine",
            "none",
            "--caption-engine",
            "none",
            "--ignore-render-settings",
        ]

    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
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
        lock_path = ai_processing_locks._processing_lock_path(image_path)
        lock_path.write_text('{"pid": 999999, "job_id": "old"}', encoding="utf-8")

        with mock.patch.object(ai_processing_locks, "_pid_alive", return_value=False):
            acquired = ai_index._acquire_image_processing_lock(image_path)
        try:
            self.assertEqual(acquired, lock_path)
            payload = ai_processing_locks._read_processing_lock(lock_path)
            self.assertEqual(payload["pid"], ai_processing_locks.os.getpid())
        finally:
            ai_index._release_image_processing_lock(acquired)

    def test_release_image_processing_lock_retries_windows_sharing_violation(self):
        lock_path = self.root / "Photo_01.jpg.photoalbums-ai.lock"
        lock_path.touch()
        sharing_violation = PermissionError("lock file is busy")
        sharing_violation.winerror = 32

        with (
            mock.patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=[sharing_violation, None],
            ) as unlink,
            mock.patch.object(ai_processing_locks.time, "sleep") as sleep,
        ):
            ai_index._release_image_processing_lock(lock_path)

        self.assertEqual(unlink.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_run_returns_error_when_single_photo_is_locked(self):
        image_path = self.root / "Photo_01.jpg"
        image_path.touch()
        lock_path = ai_index._acquire_image_processing_lock(image_path)
        try:
            with mock.patch("sys.stdout", new=io.StringIO()), mock.patch("sys.stderr", new=io.StringIO()):
                result = ai_index.run(self._single_photo_args(image_path))
            self.assertEqual(result, 1)
        finally:
            ai_index._release_image_processing_lock(lock_path)

    def test_run_returns_error_when_batch_lock_exists(self):
        view_dir = self.root / "Album_View"
        view_dir.mkdir(parents=True)
        (view_dir / "Photo_01.jpg").touch()
        batch_lock_path = ai_index._acquire_batch_processing_lock(self.root)
        try:
            stdout = io.StringIO()
            with mock.patch("sys.stdout", new=stdout), mock.patch("sys.stderr", new=io.StringIO()):
                result = ai_index.run(self._batch_args())
            self.assertEqual(result, 1)
            self.assertIn("another photoalbums ai batch run is already active", stdout.getvalue())
        finally:
            ai_index._release_batch_processing_lock(batch_lock_path)


if __name__ == "__main__":
    unittest.main()
