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

    def test_page_caption_mentions_photo_count_and_page_text(self):
        text = ai_caption.build_page_caption(
            photo_count=2,
            people=["Alice", "Bob"],
            objects=["bus", "chair"],
            ocr_text="Welcome to Beijing station",
        )
        self.assertIn("contains 2 photo(s)", text)
        self.assertIn("Across the page", text)
        self.assertIn("Visible text on the page reads:", text)

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

    def test_blip_falls_back_to_template_on_error(self):
        fake_blip = mock.Mock()
        fake_blip.describe.side_effect = RuntimeError("cpu path failed")
        with mock.patch("photoalbums.lib.ai_caption.BlipLocalCaptioner", return_value=fake_blip):
            engine = ai_caption.CaptionEngine(engine="blip")
            out = engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
            )
        self.assertEqual(out.engine, "template")
        self.assertTrue(out.fallback)
        self.assertIn("cpu path failed", out.error)
        self.assertIn("Alice", out.text)

    def test_qwen_engine_forwards_cpu_tuning_settings(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.return_value = "caption text"
        with mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen) as ctor:
            engine = ai_caption.CaptionEngine(
                engine="qwen",
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                max_tokens=64,
                temperature=0.1,
                qwen_attn_implementation="sdpa",
                qwen_min_pixels=131072,
                qwen_max_pixels=524288,
                max_image_edge=1024,
            )
            out = engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
            )
        ctor.assert_called_once_with(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            max_new_tokens=64,
            temperature=0.1,
            attn_implementation="sdpa",
            min_pixels=131072,
            max_pixels=524288,
            max_image_edge=1024,
        )
        self.assertEqual(out.engine, "qwen")
        self.assertEqual(out.text, "caption text")


if __name__ == "__main__":
    unittest.main()
