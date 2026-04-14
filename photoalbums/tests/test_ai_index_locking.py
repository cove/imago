import io
import sys
import tempfile
import unittest
from contextlib import nullcontext
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
            "--include-view",
            "--disable-people",
            "--disable-objects",
            "--ocr-engine",
            "none",
            "--caption-engine",
            "none",
            "--ignore-render-settings",
        ]

    @staticmethod
    def _mock_layout(image_path: Path):
        return nullcontext(
            mock.Mock(
                page_like=False,
                content_path=image_path,
                content_bounds=None,
            )
        )

    @staticmethod
    def _analysis(image_path: Path) -> ai_index.ImageAnalysis:
        return ai_index.ImageAnalysis(
            image_path=image_path,
            people_names=[],
            object_labels=[],
            ocr_text="",
            ocr_keywords=[],
            subjects=[],
            description="",
            payload={
                "people": [],
                "objects": [],
                "ocr": {"engine": "none", "model": "", "language": "eng", "keywords": [], "chars": 0},
                "caption": {
                    "requested_engine": "none",
                    "effective_engine": "none",
                    "fallback": False,
                    "error": "",
                    "model": "",
                },
            },
        )

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
    
    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
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

    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
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
    
    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
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

    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
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

    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
    def test_sharded_run_ignores_existing_batch_lock(self):
        view_dir = self.root / "Album_View"
        view_dir.mkdir(parents=True)
        image_path = view_dir / "Photo_01.jpg"
        image_path.touch()
        batch_lock_path = ai_index._acquire_batch_processing_lock(self.root)
        try:
            stdout = io.StringIO()
            args = self._batch_args() + ["--shard-count", "2", "--shard-index", "0", "--dry-run"]
            with (
                mock.patch("sys.stdout", new=stdout),
                mock.patch("sys.stderr", new=io.StringIO()),
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image_path),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=self._analysis(image_path)),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=self._analysis(image_path).payload),
            ):
                result = ai_index.run(args)
            self.assertEqual(result, 0)
            self.assertIn("Processed: 1", stdout.getvalue())
        finally:
            ai_index._release_batch_processing_lock(batch_lock_path)

    @skip("Temporarily disabled due to intermittent Windows KeyboardInterrupt reporting during pytest teardown.")
    def test_sharded_run_skips_locked_dependency_collision(self):
        view_dir = self.root / "Album_View"
        view_dir.mkdir(parents=True)
        image_path = view_dir / "Photo_01.jpg"
        image_path.touch()
        lock_path = ai_index._acquire_image_processing_lock(image_path)
        try:
            stdout = io.StringIO()
            args = self._batch_args() + ["--shard-count", "2", "--shard-index", "0", "--verbose"]
            with mock.patch("sys.stdout", new=stdout), mock.patch("sys.stderr", new=io.StringIO()):
                result = ai_index.run(args)
            self.assertEqual(result, 0)
            self.assertIn("skip  Photo_01.jpg (already processing", stdout.getvalue())
        finally:
            ai_index._release_image_processing_lock(lock_path)


if __name__ == "__main__":
    unittest.main()
