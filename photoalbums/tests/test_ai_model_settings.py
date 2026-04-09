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
                    selected_ctm_model = "restoration"
                    lmstudio_base_url = "http://lmstudio.local:1234/v1"

                    [ctm_validation]
                    min_confidence = 0.4
                    max_abs_coefficient = 3.5
                    max_row_sum = 4.5
                    max_clipping_ratio = 0.25

                    [models]
                    big = "qwen/qwen3-vl-30b"
                    fast = "qwen/qwen3.5-9b"
                    restoration = "google/gemma-4-31b-it"
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
                self.assertEqual(loaded["selected_ctm_model"], "restoration")
                self.assertEqual(loaded["ocr_model"], "qwen/qwen3.5-9b")
                self.assertEqual(loaded["caption_model"], "qwen/qwen3-vl-30b")
                self.assertEqual(loaded["ctm_model"], "google/gemma-4-31b-it")
                self.assertEqual(loaded["lmstudio_base_url"], "http://lmstudio.local:1234/v1")
                self.assertEqual(
                    ai_model_settings.default_ocr_model(),
                    "qwen/qwen3.5-9b",
                )
                self.assertEqual(
                    ai_model_settings.default_caption_model(),
                    "qwen/qwen3-vl-30b",
                )
                self.assertEqual(
                    ai_model_settings.default_lmstudio_base_url(),
                    "http://lmstudio.local:1234/v1",
                )
                self.assertEqual(ai_model_settings.default_ctm_model(), "google/gemma-4-31b-it")
                self.assertEqual(
                    ai_model_settings.default_ctm_validation_settings(),
                    {
                        "min_confidence": 0.4,
                        "max_abs_coefficient": 3.5,
                        "max_row_sum": 4.5,
                        "max_clipping_ratio": 0.25,
                    },
                )

        self.assertEqual(
            loaded["models"],
            {
                "big": "qwen/qwen3-vl-30b",
                "fast": "qwen/qwen3.5-9b",
                "restoration": "google/gemma-4-31b-it",
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
                    big = "qwen/qwen3-vl-30b"
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
                    big = "qwen/qwen3-vl-30b"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ai_model_settings, "AI_MODEL_SETTINGS_PATH", path):
                ai_model_settings.load_ai_model_settings.cache_clear()
                loaded = ai_model_settings.load_ai_model_settings()

        self.assertEqual(loaded["lmstudio_base_url"], "http://localhost:1234/v1")


if __name__ == "__main__":
    unittest.main()
