import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import textwrap

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_model_settings


class TestAIModelSettings(unittest.TestCase):
    def tearDown(self):
        ai_model_settings.load_ai_model_settings.cache_clear()

    def test_load_ai_model_settings_reads_model_aliases_and_selected_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_models.toml"
            path.write_text(
                textwrap.dedent(
                    """
                    selected_ocr_model = "fast"
                    selected_caption_model = "big"
                    view_region_model = "docling"
                    lmstudio_base_url = "http://lmstudio.local:1234/v1"

                    [docling_pipeline]
                    preset = "granite_docling"
                    backend = "auto_inline"
                    device = "auto"
                    retries = 4

                    [models]
                    big = ["qwen/qwen3-vl-30b", "qwen/qwen3-vl-32b"]
                    fast = ["qwen/qwen3.5-9b"]
                    docling = ["granite-docling-258m"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                loaded = ai_model_settings.load_ai_model_settings()
                self.assertEqual(loaded["selected_ocr_model"], "fast")
                self.assertEqual(loaded["selected_caption_model"], "big")
                self.assertEqual(loaded["ocr_model"], "qwen/qwen3.5-9b")
                self.assertEqual(loaded["caption_model"], "qwen/qwen3-vl-30b")
                self.assertEqual(loaded["ocr_models"], ["qwen/qwen3.5-9b"])
                self.assertEqual(loaded["caption_models"], ["qwen/qwen3-vl-30b", "qwen/qwen3-vl-32b"])
                self.assertEqual(
                    loaded["view_region_models"],
                    ["granite-docling-258m"],
                )
                self.assertEqual(loaded["view_region_model"], "granite-docling-258m")
                self.assertEqual(loaded["lmstudio_base_url"], "http://lmstudio.local:1234/v1")
                self.assertEqual(loaded["docling_preset"], "granite_docling")
                self.assertEqual(loaded["docling_backend"], "auto_inline")
                self.assertEqual(loaded["docling_device"], "auto")
                self.assertEqual(loaded["docling_retries"], 4)
                self.assertEqual(
                    ai_model_settings.default_ocr_model(),
                    "qwen/qwen3.5-9b",
                )
                self.assertEqual(ai_model_settings.default_ocr_models(), ["qwen/qwen3.5-9b"])
                self.assertEqual(
                    ai_model_settings.default_caption_model(),
                    "qwen/qwen3-vl-30b",
                )
                self.assertEqual(
                    ai_model_settings.default_caption_models(),
                    ["qwen/qwen3-vl-30b", "qwen/qwen3-vl-32b"],
                )
                self.assertEqual(
                    ai_model_settings.default_lmstudio_base_url(),
                    "http://lmstudio.local:1234/v1",
                )
                self.assertEqual(ai_model_settings.default_view_region_model(), "granite-docling-258m")
                self.assertEqual(
                    ai_model_settings.default_view_region_models(),
                    ["granite-docling-258m"],
                )
                self.assertEqual(ai_model_settings.default_docling_preset(), "granite_docling")
                self.assertEqual(ai_model_settings.default_docling_backend(), "auto_inline")
                self.assertEqual(ai_model_settings.default_docling_device(), "auto")
                self.assertEqual(ai_model_settings.default_docling_retries(), 4)

        self.assertEqual(
            loaded["models"],
            {
                "big": ["qwen/qwen3-vl-30b", "qwen/qwen3-vl-32b"],
                "docling": ["granite-docling-258m"],
                "fast": ["qwen/qwen3.5-9b"],
            },
        )

    def test_load_ai_model_settings_rejects_selected_alias_not_in_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_models.toml"
            path.write_text(
                textwrap.dedent(
                    """
                    selected_ocr_model = "fast"
                    selected_caption_model = "big"

                    [models]
                    big = ["qwen/qwen3-vl-30b"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                with self.assertRaises(RuntimeError) as exc:
                    ai_model_settings.load_ai_model_settings()

        self.assertIn("selected_ocr_model", str(exc.exception))

    def test_load_ai_model_settings_requires_models_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_models.toml"
            path.write_text(
                textwrap.dedent(
                    """
                    selected_ocr_model = "big"
                    selected_caption_model = "big"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                with self.assertRaises(RuntimeError) as exc:
                    ai_model_settings.load_ai_model_settings()

        self.assertIn("'models' must be a TOML table", str(exc.exception))

    def test_load_ai_model_settings_defaults_lmstudio_base_url_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_models.toml"
            path.write_text(
                textwrap.dedent(
                    """
                    selected_ocr_model = "big"
                    selected_caption_model = "big"

                    [models]
                    big = ["qwen/qwen3-vl-30b"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                loaded = ai_model_settings.load_ai_model_settings()

        self.assertEqual(loaded["lmstudio_base_url"], "http://localhost:1234/v1")

    def test_load_ai_model_settings_allows_direct_view_region_model_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_models.toml"
            path.write_text(
                textwrap.dedent(
                    """
                    selected_ocr_model = "big"
                    selected_caption_model = "big"
                    view_region_model = "granite-docling-258m"

                    [models]
                    big = ["qwen/qwen3-vl-30b"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                loaded = ai_model_settings.load_ai_model_settings()

        self.assertEqual(loaded["view_region_model"], "granite-docling-258m")
        self.assertEqual(loaded["view_region_models"], ["granite-docling-258m"])


if __name__ == "__main__":
    unittest.main()
