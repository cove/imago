import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_caption


class TestAICaption(unittest.TestCase):
    def test_template_caption_includes_expected_sections(self):
        text = ai_caption.build_template_caption(
            people=["Alice", "Bob"],
            objects=["bus", "chair"],
            ocr_text="Welcome to Beijing station",
        )
        self.assertIn("Alice", text)
        self.assertIn("bus", text)
        self.assertIn("Visible text reads:", text)

    def test_caption_engine_none_returns_empty_text(self):
        engine = ai_caption.CaptionEngine(engine="none")
        out = engine.generate(
            image_path="sample.jpg",
            people=["Alice"],
            objects=["car"],
            ocr_text="",
        )
        self.assertEqual(out.text, "")
        self.assertEqual(out.engine, "none")

    def test_qwen_falls_back_to_template_on_error(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.side_effect = RuntimeError("model offline")
        with mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen):
            engine = ai_caption.CaptionEngine(engine="qwen")
            out = engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
            )
        self.assertEqual(out.engine, "template")
        self.assertTrue(out.fallback)
        self.assertIn("model offline", out.error)
        self.assertIn("Alice", out.text)


if __name__ == "__main__":
    unittest.main()
