import json
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

from photoalbums.lib import ai_caption


class TestAICaption(unittest.TestCase):
    def test_normalize_caption_strips_qwen_thinking_output(self):
        text = ai_caption._normalize_caption(
            "assistant: <think>Reason through the image.</think> Two people are standing beside a car."
        )
        self.assertEqual(text, "Two people are standing beside a car.")

    def test_normalize_caption_strips_model_request_preamble(self):
        text = ai_caption._normalize_caption(
            "The user wants a detailed description of the provided image.\n\n"
            "**1. Visual Analysis:**\n"
            "- Blue textured cover."
        )
        self.assertEqual(text, "**1. Visual Analysis:** - Blue textured cover.")

    def test_template_caption_includes_expected_sections(self):
        text = ai_caption.build_template_caption(
            people=["Alice", "Bob"],
            objects=["bus", "chair"],
            ocr_text="Welcome to Beijing station",
        )
        self.assertIn("Alice", text)
        self.assertIn("bus", text)
        self.assertIn("Visible text reads:", text)

    def test_template_caption_returns_empty_when_no_signals_exist(self):
        text = ai_caption.build_template_caption(
            people=[],
            objects=[],
            ocr_text="",
        )
        self.assertEqual(text, "")

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

    def test_page_caption_omits_no_match_sentence(self):
        text = ai_caption.build_page_caption(
            photo_count=1,
            people=[],
            objects=[],
            ocr_text="",
        )
        self.assertEqual(text, "This album page contains 1 photo(s).")

    def test_cover_page_caption_mentions_cover_and_title_text(self):
        text = ai_caption.build_cover_page_caption(
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
        )
        self.assertIn("cover or title page", text)
        self.assertIn("Visible title text reads:", text)
        self.assertIn("MAINLAND CHINA 1986 BOOK 11", text)

    def test_infer_album_context_detects_family_album_from_path(self):
        context = ai_caption.infer_album_context(
            image_path=Path("Family_View") / "Family_1980-1985_B08_P01.jpg",
            allow_ocr=False,
        )
        self.assertEqual(context.kind, ai_caption.ALBUM_KIND_FAMILY)
        self.assertEqual(context.label, "Family Photo Album")

    def test_infer_album_context_detects_photo_essay_from_collection_name(self):
        context = ai_caption.infer_album_context(
            image_path=Path("Photo Albums") / "EasternEuropeSpainMorocco_1988_B00_View" / "EasternEuropeSpainMorocco_1988_B00_P01.jpg",
            allow_ocr=False,
        )
        self.assertEqual(context.kind, ai_caption.ALBUM_KIND_PHOTO_ESSAY)
        self.assertEqual(context.label, "Photo Essay")
        self.assertIn("Eastern Europe", context.focus)
        self.assertIn("Spain", context.focus)
        self.assertIn("Morocco", context.focus)

    def test_cover_page_caption_mentions_photo_essay_when_context_present(self):
        context = ai_caption.AlbumContext(
            kind=ai_caption.ALBUM_KIND_PHOTO_ESSAY,
            label="Photo Essay",
            focus="China",
        )
        text = ai_caption.build_cover_page_caption(
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
            album_context=context,
        )
        self.assertIn("Photo Essay", text)
        self.assertIn("China", text)

    def test_build_qwen_prompt_includes_cordell_album_rules(self):
        prompt = ai_caption._build_qwen_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_View" / "Family_1980-1985_B08_P01.jpg",
        )
        self.assertIn("Cordell Photo Albums rules:", prompt)
        self.assertIn("Family Photo Album", prompt)
        self.assertIn("Photo Essay", prompt)

    def test_looks_like_album_cover_detects_blue_title_page(self):
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "China_1986_B02_P01.jpg"
            image = np.full((360, 260, 3), (185, 125, 70), dtype=np.uint8)
            cv2.putText(
                image,
                "CHINA",
                (35, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (245, 245, 245),
                3,
                cv2.LINE_AA,
            )
            self.assertTrue(cv2.imwrite(str(path), image))
            self.assertTrue(
                ai_caption.looks_like_album_cover(
                    path,
                    ocr_text="CHINA 1986",
                )
            )

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

    def test_legacy_blip_alias_routes_to_qwen_and_falls_back_to_template_on_error(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.side_effect = RuntimeError("model offline")
        with mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen) as ctor:
            engine = ai_caption.CaptionEngine(engine="blip")
            out = engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
            )
        ctor.assert_called_once_with(
            model_name=ai_caption.DEFAULT_QWEN_CAPTION_MODEL,
            prompt_text="",
            max_new_tokens=96,
            temperature=0.2,
            attn_implementation="auto",
            min_pixels=0,
            max_pixels=0,
            max_image_edge=0,
        )
        self.assertEqual(engine.engine, "qwen")
        self.assertEqual(out.engine, "template")
        self.assertTrue(out.fallback)
        self.assertIn("model offline", out.error)
        self.assertIn("Alice", out.text)

    def test_qwen_engine_forwards_cpu_tuning_settings(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.return_value = "caption text"
        with mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen) as ctor:
            engine = ai_caption.CaptionEngine(
                engine="qwen",
                model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                caption_prompt="Describe this exact image",
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
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            prompt_text="Describe this exact image",
            max_new_tokens=64,
            temperature=0.1,
            attn_implementation="sdpa",
            min_pixels=131072,
            max_pixels=524288,
            max_image_edge=1024,
        )
        self.assertEqual(out.engine, "qwen")
        self.assertEqual(out.text, "caption text")

    def test_qwen_loader_uses_local_snapshot_and_safe_max_pixels(self):
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
            fake_torch.cuda.is_available.return_value = False

            with (
                mock.patch.object(ai_caption, "HF_MODEL_CACHE_DIR", cache_dir),
                mock.patch.object(
                    ai_caption,
                    "_load_qwen_transformers",
                    return_value=(fake_torch, fake_processor_cls, None, fake_model_cls),
                ),
            ):
                captioner = ai_caption.QwenLocalCaptioner(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
                captioner._ensure_loaded()

            processor_kwargs = fake_processor_cls.from_pretrained.call_args.kwargs
            self.assertTrue(processor_kwargs["local_files_only"])
            self.assertEqual(processor_kwargs["max_pixels"], ai_caption.DEFAULT_QWEN_AUTO_MAX_PIXELS)

            model_kwargs = fake_model_cls.from_pretrained.call_args.kwargs
            self.assertTrue(model_kwargs["local_files_only"])

    def test_legacy_qwen_model_alias_resolves_to_supported_default(self):
        self.assertEqual(
            ai_caption.resolve_caption_model("qwen", "Qwen/Qwen3.5-4B"),
            ai_caption.DEFAULT_QWEN_CAPTION_MODEL,
        )

    def test_lmstudio_captioner_posts_chat_completion_request(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": "A crowded collage of travel snapshots."
                    }
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
            self.assertEqual(timeout, ai_caption.DEFAULT_LMSTUDIO_TIMEOUT_SECONDS)
            self.assertTrue(request.full_url.endswith("/chat/completions"))
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "qwen2.5-vl")
            self.assertEqual(payload["messages"][0]["content"][0]["text"], "Describe this exact image")
            self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"not-a-real-jpeg")
            with (
                mock.patch.object(ai_caption, "_build_data_url", return_value="data:image/jpeg;base64,abc123"),
                mock.patch.object(ai_caption, "_select_lmstudio_model", return_value="qwen2.5-vl"),
                mock.patch.object(ai_caption.urllib.request, "urlopen", side_effect=fake_urlopen),
            ):
                captioner = ai_caption.LMStudioCaptioner(
                    prompt_text="Describe this exact image",
                    base_url="http://127.0.0.1:1234",
                )
                text = captioner.describe(
                    image_path=image_path,
                    people=[],
                    objects=[],
                    ocr_text="",
                )

        self.assertEqual(text, "A crowded collage of travel snapshots.")


if __name__ == "__main__":
    unittest.main()
