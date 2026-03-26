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

                    [models]
                    big = "qwen/qwen3-vl-30b"
                    fast = "qwen/qwen3.5-9b"
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
                self.assertEqual(
                    ai_model_settings.default_ocr_model(),
                    "qwen/qwen3.5-9b",
                )
                self.assertEqual(
                    ai_model_settings.default_caption_model(),
                    "qwen/qwen3-vl-30b",
                )

        self.assertEqual(
            loaded["models"],
            {
                "big": "qwen/qwen3-vl-30b",
                "fast": "qwen/qwen3.5-9b",
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


if __name__ == "__main__":
    unittest.main()
