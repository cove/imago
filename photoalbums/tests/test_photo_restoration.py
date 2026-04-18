"""Tests for photo_restoration module."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def _reset_pipeline():
    """Reset the module-level pipeline singleton between tests."""
    import photoalbums.lib.photo_restoration as mod

    mod._pipeline = None


class TestRestorePhoto(unittest.TestCase):
    def setUp(self):
        _reset_pipeline()

    def tearDown(self):
        _reset_pipeline()

    def test_returns_restored_image(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10), color=(100, 100, 100))
        restored = Image.new("RGB", (10, 10), color=(200, 200, 200))

        mock_result = mock.MagicMock()
        mock_result.images = [restored]
        mock_pipe = mock.MagicMock(return_value=mock_result)

        mock_pipeline_cls = mock.MagicMock(return_value=mock_pipe)
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=64_000_000_000),
        ):
            result = restore_photo(original)

        self.assertIs(result, restored)
        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "RealRestorer/RealRestorer",
            torch_dtype=fake_torch.bfloat16,
        )
        mock_pipe.enable_model_cpu_offload.assert_called_once_with()
        mock_pipe.to.assert_not_called()

    def test_import_error_returns_original(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10))

        with mock.patch.dict("sys.modules", {"diffusers": None, "torch": None}):
            with self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"):
                result = restore_photo(original)

        self.assertIs(result, original)

    def test_import_error_is_no_op_on_subsequent_calls(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10))

        with mock.patch.dict("sys.modules", {"diffusers": None, "torch": None}):
            with self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"):
                restore_photo(original)
            result = restore_photo(original)

        self.assertIs(result, original)

    def test_model_load_refused_when_model_size_meets_or_exceeds_installed_ram(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10))
        mock_pipeline_cls = mock.MagicMock()
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=41_800_000_000),
            self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"),
        ):
            result = restore_photo(original)

        self.assertIs(result, original)
        mock_pipeline_cls.from_pretrained.assert_not_called()

    def test_inference_failure_returns_original(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10))

        mock_pipe = mock.MagicMock(side_effect=RuntimeError("GPU OOM"))
        mock_pipeline_cls = mock.MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=64_000_000_000),
        ):
            with self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"):
                result = restore_photo(original)

        self.assertIs(result, original)

    def test_cpu_only_setup_uses_cpu_device_without_offload(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo

        original = Image.new("RGB", (10, 10), color=(100, 100, 100))
        restored = Image.new("RGB", (10, 10), color=(200, 200, 200))

        mock_result = mock.MagicMock()
        mock_result.images = [restored]
        mock_pipe = mock.MagicMock(return_value=mock_result)

        mock_pipeline_cls = mock.MagicMock(return_value=mock_pipe)
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=64_000_000_000),
        ):
            result = restore_photo(original)

        self.assertIs(result, restored)
        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "RealRestorer/RealRestorer",
            torch_dtype=fake_torch.float32,
        )
        mock_pipe.to.assert_called_once_with("cpu")
        mock_pipe.enable_model_cpu_offload.assert_not_called()

    def test_restore_photo_with_result_reports_success(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import REAL_RESTORER_MODEL_NAME, restore_photo_with_result

        original = Image.new("RGB", (10, 10), color=(100, 100, 100))
        restored = Image.new("RGB", (10, 10), color=(200, 200, 200))

        mock_result = mock.MagicMock()
        mock_result.images = [restored]
        mock_pipe = mock.MagicMock(return_value=mock_result)

        mock_pipeline_cls = mock.MagicMock(return_value=mock_pipe)
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=64_000_000_000),
        ):
            result, status = restore_photo_with_result(original)

        self.assertIs(result, restored)
        self.assertEqual(status, "restored")
        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            REAL_RESTORER_MODEL_NAME,
            torch_dtype=fake_torch.bfloat16,
        )

    def test_restore_photo_with_result_reports_unavailable(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo_with_result

        original = Image.new("RGB", (10, 10))

        with mock.patch.dict("sys.modules", {"diffusers": None, "torch": None}):
            with self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"):
                result, status = restore_photo_with_result(original)

        self.assertIs(result, original)
        self.assertEqual(status, "unavailable")

    def test_restore_photo_with_result_reports_failure(self):
        from PIL import Image
        from photoalbums.lib.photo_restoration import restore_photo_with_result

        original = Image.new("RGB", (10, 10))

        mock_pipe = mock.MagicMock(side_effect=RuntimeError("GPU OOM"))
        mock_pipeline_cls = mock.MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        fake_diffusers = types.SimpleNamespace(RealRestorerPipeline=mock_pipeline_cls)
        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with (
            mock.patch.dict("sys.modules", {"diffusers": fake_diffusers, "torch": fake_torch}),
            mock.patch("photoalbums.lib.photo_restoration._installed_ram_bytes", return_value=64_000_000_000),
        ):
            with self.assertLogs("photoalbums.lib.photo_restoration", level="WARNING"):
                result, status = restore_photo_with_result(original)

        self.assertIs(result, original)
        self.assertEqual(status, "failed")


class TestCropPageRegionsRestoration(unittest.TestCase):
    def test_restore_photo_called_before_save(self):
        """crop_page_regions passes the in-memory PIL Image to restore_photo before saving."""
        import tempfile
        from pathlib import Path

        from PIL import Image

        from photoalbums.lib.ai_photo_crops import crop_page_regions
        from photoalbums.lib.xmp_sidecar import read_pipeline_step
        from photoalbums.tests.test_ai_photo_crops import _make_minimal_jpeg, _write_region_xmp

        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Album_Pages"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Album_Photos"
            view_jpg = view_dir / "Album_B00_P02_V.jpg"
            img_w, img_h = 200, 100
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_jpg.with_suffix(".xmp"),
                [{"index": 0, "x": 0, "y": 0, "width": img_w, "height": img_h, "caption": "Test"}],
                img_w,
                img_h,
            )

            captured = []

            def fake_restore(img):
                captured.append(img)
                return img, "restored"

            with mock.patch("photoalbums.lib.photo_restoration.restore_photo_with_result", side_effect=fake_restore):
                count = crop_page_regions(view_jpg, photos_dir, skip_restoration=False)

            self.assertEqual(count, 1)
            self.assertEqual(len(captured), 1)
            self.assertIsInstance(captured[0], Image.Image)
            state = read_pipeline_step(photos_dir / "Album_B00_P02_D01-00_V.xmp", "photo_restoration")
            self.assertEqual(state["result"], "restored")
            self.assertEqual(state["model"], "RealRestorer/RealRestorer")

    def test_skip_restoration_skips_restore_photo(self):
        """crop_page_regions does not call restore_photo when skip_restoration=True."""
        import tempfile
        from pathlib import Path

        from photoalbums.lib.ai_photo_crops import crop_page_regions
        from photoalbums.lib.xmp_sidecar import read_pipeline_step
        from photoalbums.tests.test_ai_photo_crops import _make_minimal_jpeg, _write_region_xmp

        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Album_Pages"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Album_Photos"
            view_jpg = view_dir / "Album_B00_P02_V.jpg"
            img_w, img_h = 200, 100
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_jpg.with_suffix(".xmp"),
                [{"index": 0, "x": 0, "y": 0, "width": img_w, "height": img_h, "caption": "Test"}],
                img_w,
                img_h,
            )

            with mock.patch("photoalbums.lib.photo_restoration.restore_photo_with_result") as mock_restore:
                count = crop_page_regions(view_jpg, photos_dir, skip_restoration=True)

            self.assertEqual(count, 1)
            mock_restore.assert_not_called()
            state = read_pipeline_step(photos_dir / "Album_B00_P02_D01-00_V.xmp", "photo_restoration")
            self.assertEqual(state["result"], "skipped")
            self.assertNotIn("model", state)

    def test_force_restoration_reruns_on_existing_crop(self):
        import tempfile
        from pathlib import Path

        from photoalbums.lib.ai_photo_crops import crop_page_regions
        from photoalbums.lib.xmp_sidecar import read_pipeline_step
        from photoalbums.tests.test_ai_photo_crops import _make_minimal_jpeg, _write_region_xmp

        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "Album_Pages"
            view_dir.mkdir()
            photos_dir = Path(tmp) / "Album_Photos"
            view_jpg = view_dir / "Album_B00_P02_V.jpg"
            img_w, img_h = 200, 100
            _make_minimal_jpeg(view_jpg, img_w, img_h)
            _write_region_xmp(
                view_jpg.with_suffix(".xmp"),
                [{"index": 0, "x": 0, "y": 0, "width": img_w, "height": img_h, "caption": "Test"}],
                img_w,
                img_h,
            )

            with mock.patch(
                "photoalbums.lib.photo_restoration.restore_photo_with_result",
                side_effect=lambda image: (image, "restored"),
            ) as first_restore:
                count = crop_page_regions(view_jpg, photos_dir, skip_restoration=False)

            self.assertEqual(count, 1)
            self.assertEqual(first_restore.call_count, 1)

            with mock.patch(
                "photoalbums.lib.photo_restoration.restore_photo_with_result",
                side_effect=lambda image: (image, "restored"),
            ) as second_restore:
                count = crop_page_regions(view_jpg, photos_dir, force_restoration=True)

            self.assertEqual(count, 1)
            self.assertEqual(second_restore.call_count, 1)
            state = read_pipeline_step(photos_dir / "Album_B00_P02_D01-00_V.xmp", "photo_restoration")
            self.assertEqual(state["result"], "restored")

