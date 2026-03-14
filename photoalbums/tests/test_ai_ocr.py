import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_ocr


class TestAIOcr(unittest.TestCase):
    def test_legacy_docstrange_ocr_engine_alias_resolves_to_qwen(self):
        self.assertEqual(ai_ocr._normalize_ocr_engine("docstrange"), "qwen")

    def test_qwen_ocr_requires_local_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with mock.patch.object(ai_ocr, "HF_MODEL_CACHE_DIR", cache_dir):
                ocr = ai_ocr.OCREngine(engine="qwen")
                with self.assertRaises(RuntimeError) as exc:
                    ocr._ensure_loaded()
        self.assertIn("local-only inference", str(exc.exception))

    def test_qwen_ocr_uses_local_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            snapshot = cache_dir / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")

            fake_processor_cls = mock.Mock()
            fake_processor = mock.Mock()
            fake_processor_cls.from_pretrained.return_value = fake_processor

            fake_model_cls = mock.Mock()
            fake_model = mock.Mock()
            fake_model_cls.from_pretrained.return_value = fake_model

            fake_torch = mock.Mock()

            with (
                mock.patch.object(ai_ocr, "HF_MODEL_CACHE_DIR", cache_dir),
                mock.patch.object(
                    ai_ocr,
                    "_load_qwen_transformers",
                    return_value=(fake_torch, fake_processor_cls, fake_model_cls),
                ),
            ):
                ocr = ai_ocr.OCREngine(engine="qwen")
                ocr._ensure_loaded()

            processor_kwargs = fake_processor_cls.from_pretrained.call_args.kwargs
            self.assertTrue(processor_kwargs["local_files_only"])
            self.assertEqual(processor_kwargs["max_pixels"], ai_ocr.DEFAULT_QWEN_OCR_MAX_PIXELS)

            model_kwargs = fake_model_cls.from_pretrained.call_args.kwargs
            self.assertTrue(model_kwargs["local_files_only"])

    def test_qwen_ocr_reads_text_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            snapshot = cache_dir / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")

            image_path = cache_dir / "sample.jpg"
            Image.new("RGB", (320, 240), color="white").save(image_path)

            fake_input_ids = mock.Mock()
            fake_input_ids.shape = (1, 12)
            fake_input_ids.to.return_value = fake_input_ids

            fake_processor = mock.Mock()
            fake_processor.apply_chat_template.return_value = "OCR PROMPT"
            fake_processor.return_value = {"input_ids": fake_input_ids}
            fake_processor.batch_decode.return_value = ["assistant: MAINLAND CHINA\n1986 BOOK 11"]

            fake_processor_cls = mock.Mock()
            fake_processor_cls.from_pretrained.return_value = fake_processor

            fake_model = mock.Mock()
            fake_model.device = "cpu"
            fake_model.generate.return_value = object()

            fake_model_cls = mock.Mock()
            fake_model_cls.from_pretrained.return_value = fake_model

            fake_torch = mock.Mock()
            fake_torch.inference_mode.return_value = contextlib.nullcontext()

            with (
                mock.patch.object(ai_ocr, "HF_MODEL_CACHE_DIR", cache_dir),
                mock.patch.object(
                    ai_ocr,
                    "_load_qwen_transformers",
                    return_value=(fake_torch, fake_processor_cls, fake_model_cls),
                ),
            ):
                ocr = ai_ocr.OCREngine(engine="qwen")
                text = ocr.read_text(image_path)

            self.assertEqual(text, "MAINLAND CHINA\n1986 BOOK 11")
            fake_processor.apply_chat_template.assert_called_once()
            prompt_messages = fake_processor.apply_chat_template.call_args.args[0]
            self.assertEqual(prompt_messages[0]["content"][1]["text"], ai_ocr.DEFAULT_QWEN_OCR_PROMPT)

    def test_qwen_ocr_normalizes_no_text_response(self):
        self.assertEqual(ai_ocr._normalize_ocr_text("No visible text"), "")

    def test_ocr_normalization_rejects_reasoning_dump(self):
        text = ai_ocr._normalize_ocr_text(
            "The user wants me to extract text from the provided image.\n"
            "1. **Analyze the image:**\n"
            "2. **Transcribe the text found:**"
        )
        self.assertEqual(text, "")

    def test_parse_lmstudio_structured_ocr_extracts_json_after_think_prefix(self):
        text = ai_ocr._parse_lmstudio_structured_ocr(
            '<think>{ "text": "WELCOME TO\\n敦煌之夏" }',
            finish_reason="stop",
        )
        self.assertEqual(text, "WELCOME TO\n敦煌之夏")

    def test_lmstudio_ocr_uses_json_schema(self):
        response_payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"text": "MAINLAND CHINA\n1986\nBOOK 11"}),
                    },
                }
            ]
        }

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            self.assertTrue(request.full_url.endswith("/chat/completions"))
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(payload["response_format"]["json_schema"]["name"], "ocr_payload")
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (320, 240), color="white").save(image_path)
            with (
                mock.patch.object(ai_ocr, "_build_ocr_data_url", return_value="data:image/jpeg;base64,abc123"),
                mock.patch.object(ai_ocr, "_lmstudio_ocr_select_model", return_value="qwen2.5-vl"),
                mock.patch.object(ai_ocr.urllib.request, "urlopen", side_effect=fake_urlopen),
            ):
                ocr = ai_ocr.OCREngine(engine="lmstudio", base_url="http://127.0.0.1:1234")
                text = ocr.read_text(image_path)

        self.assertEqual(text, "MAINLAND CHINA\n1986\nBOOK 11")


if __name__ == "__main__":
    unittest.main()
