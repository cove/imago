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

from photoalbums.lib import ai_caption, _caption_lmstudio, _caption_qwen, ai_ocr


class TestAICaption(unittest.TestCase):
    def test_infer_album_title_prefers_cover_text_and_romanizes_book_number(self):
        text = ai_caption.infer_album_title(
            image_path=Path("Photo Albums") / "China_1986_B02_View" / "China_1986_B02_P01.jpg",
            ocr_text="MAINLAND CHINA\n1986\nBOOK 11",
        )
        self.assertEqual(text, "Mainland China Book II")

    def test_infer_printed_album_title_preserves_cover_book_label(self):
        text = ai_caption.infer_printed_album_title(
            ocr_text="MAINLAND CHINA\n1986\nBOOK 11",
        )
        self.assertEqual(text, "Mainland China Book 11")

    def test_infer_album_context_detects_family_album_from_path(self):
        context = ai_caption.infer_album_context(
            image_path=Path("Family_View") / "Family_1980-1985_B08_P01.jpg",
            allow_ocr=False,
        )
        self.assertEqual(context.kind, ai_caption.ALBUM_KIND_FAMILY)
        self.assertEqual(context.label, "Family Photo Album")

    def test_infer_album_context_detects_photo_essay_from_collection_name(self):
        context = ai_caption.infer_album_context(
            image_path=Path("Photo Albums")
            / "EasternEuropeSpainMorocco_1988_B00_View"
            / "EasternEuropeSpainMorocco_1988_B00_P01.jpg",
            allow_ocr=False,
        )
        self.assertEqual(context.kind, ai_caption.ALBUM_KIND_PHOTO_ESSAY)
        self.assertEqual(context.label, "Photo Essay")
        self.assertIn("Eastern Europe", context.focus)
        self.assertIn("Spain", context.focus)
        self.assertIn("Morocco", context.focus)

    def test_build_qwen_prompt_includes_cordell_album_rules(self):
        prompt = ai_caption._build_qwen_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_View" / "Family_1980-1985_B08_P01.jpg",
        )
        self.assertIn("Album title hint:", prompt)
        self.assertIn(
            "Treat album title hints and classification hints as supporting context",
            prompt,
        )
        self.assertIn("Family Photo Album", prompt)
        self.assertIn("Preserve visible cover labels exactly as shown", prompt)
        self.assertIn("Treat `BOOK 11` as a visible printed label in `ocr_text`", prompt)
        self.assertIn(
            "Treat `BOOK 11` as Book II only when reasoning about album identity",
            prompt,
        )
        self.assertIn("confidence is greater than 95%", prompt)
        self.assertIn(
            "Do not translate, normalize, complete, or correct text inside `ocr_text`",
            prompt,
        )
        self.assertIn("Use English translation only in caption or location reasoning", prompt)
        self.assertIn("Output GPS coordinates only when they are literally visible", prompt)
        self.assertNotIn("Filename hint:", prompt)
        self.assertNotIn("Folder hint:", prompt)

    def test_build_qwen_prompt_prefers_printed_cover_title_when_available(self):
        prompt = ai_caption._build_qwen_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "China_1986_B02_View" / "China_1986_B02_P02.jpg",
            album_title="Mainland China Book II",
            printed_album_title="Mainland China Book 11",
        )
        self.assertIn("Album title hint: Mainland China Book 11.", prompt)
        self.assertIn("Canonical album title hint: Mainland China Book II.", prompt)
        self.assertIn("Prefer the printed cover title over the normalized title", prompt)

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

    def test_qwen_returns_empty_on_error(self):
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
        self.assertEqual(out.engine, "qwen")
        self.assertTrue(out.fallback)
        self.assertIn("model offline", out.error)
        self.assertEqual(out.text, "")

    def test_legacy_blip_alias_routes_to_qwen_and_returns_empty_on_error(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.side_effect = RuntimeError("model offline")
        with (
            mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen) as ctor,
            mock.patch("photoalbums.lib.ai_caption.default_caption_model", return_value=""),
        ):
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
            stream=False,
        )
        self.assertEqual(engine.engine, "qwen")
        self.assertEqual(out.engine, "qwen")
        self.assertTrue(out.fallback)
        self.assertIn("model offline", out.error)
        self.assertEqual(out.text, "")

    def test_qwen_engine_forwards_cpu_tuning_settings(self):
        fake_qwen = mock.Mock()
        fake_qwen.describe.return_value = ai_caption.CaptionDetails(text="caption text")
        with mock.patch("photoalbums.lib.ai_caption.QwenLocalCaptioner", return_value=fake_qwen) as ctor:
            engine = ai_caption.CaptionEngine(
                engine="qwen",
                model_name="qwen/qwen3.5-9b",
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
            model_name="qwen/qwen3.5-9b",
            prompt_text="Describe this exact image",
            max_new_tokens=64,
            temperature=0.1,
            attn_implementation="sdpa",
            min_pixels=131072,
            max_pixels=524288,
            max_image_edge=1024,
            stream=False,
        )
        self.assertEqual(out.engine, "qwen")
        self.assertEqual(out.text, "caption text")

    def test_qwen_loader_uses_local_snapshot_and_safe_max_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            snapshot = cache_dir / "models--qwen--qwen3.5-9b" / "snapshots" / "abc123"
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
                mock.patch.object(_caption_qwen, "HF_MODEL_CACHE_DIR", cache_dir),
                mock.patch.object(ai_ocr, "HF_MODEL_CACHE_DIR", cache_dir),
                mock.patch.object(
                    _caption_qwen,
                    "_load_qwen_transformers",
                    return_value=(fake_torch, fake_processor_cls, fake_model_cls),
                ),
            ):
                captioner = ai_caption.QwenLocalCaptioner(model_name="qwen/qwen3.5-9b")
                captioner._ensure_loaded()

            processor_kwargs = fake_processor_cls.from_pretrained.call_args.kwargs
            self.assertTrue(processor_kwargs["local_files_only"])
            self.assertEqual(processor_kwargs["max_pixels"], ai_caption.DEFAULT_QWEN_AUTO_MAX_PIXELS)

            model_kwargs = fake_model_cls.from_pretrained.call_args.kwargs
            self.assertTrue(model_kwargs["local_files_only"])

    def test_lmstudio_captioner_posts_chat_completion_request(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "caption": "A crowded collage of travel snapshots.",
                                "location_name": "",
                            }
                        )
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
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(payload["response_format"]["json_schema"]["name"], "caption_payload")
            self.assertEqual(payload["response_format"]["json_schema"]["strict"], "true")
            self.assertEqual(
                payload["messages"][1]["content"][0]["text"],
                "Describe this exact image",
            )
            self.assertTrue(
                payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            )
            self.assertNotIn(
                "translations",
                payload["response_format"]["json_schema"]["schema"]["properties"],
            )
            self.assertIn(
                "location_name",
                payload["response_format"]["json_schema"]["schema"]["properties"],
            )
            self.assertNotIn(
                "gps_latitude",
                payload["response_format"]["json_schema"]["schema"]["properties"],
            )
            self.assertNotIn(
                "gps_longitude",
                payload["response_format"]["json_schema"]["schema"]["properties"],
            )
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"not-a-real-jpeg")
            with (
                mock.patch.object(
                    _caption_lmstudio,
                    "_build_data_url",
                    return_value="data:image/jpeg;base64,abc123",
                ),
                mock.patch.object(
                    _caption_lmstudio,
                    "_select_lmstudio_model",
                    return_value="qwen2.5-vl",
                ),
                mock.patch.object(
                    _caption_lmstudio.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
            ):
                captioner = ai_caption.LMStudioCaptioner(
                    prompt_text="Describe this exact image",
                    base_url="http://127.0.0.1:1234",
                )
                details = captioner.describe(
                    image_path=image_path,
                    prompt="Describe this exact image",
                )

        self.assertEqual(details.text, "A crowded collage of travel snapshots.")
        self.assertEqual(details.gps_latitude, "")
        self.assertEqual(details.gps_longitude, "")
        self.assertEqual(details.location_name, "")

    def test_lmstudio_captioner_posts_people_count_request(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "people_present": True,
                                "estimated_people_count": 4,
                            }
                        )
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
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "people_count_payload",
            )
            self.assertEqual(
                payload["messages"][1]["content"][0]["text"],
                "Count the visible people",
            )
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"not-a-real-jpeg")
            with (
                mock.patch.object(
                    _caption_lmstudio,
                    "_build_data_url",
                    return_value="data:image/jpeg;base64,abc123",
                ),
                mock.patch.object(
                    _caption_lmstudio,
                    "_select_lmstudio_model",
                    return_value="qwen2.5-vl",
                ),
                mock.patch.object(
                    _caption_lmstudio.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
            ):
                captioner = ai_caption.LMStudioCaptioner(
                    prompt_text="Describe this exact image",
                    base_url="http://127.0.0.1:1234",
                )
                details = captioner.estimate_people(
                    image_path=image_path,
                    prompt="Count the visible people",
                )

        self.assertTrue(details.people_present)
        self.assertEqual(details.estimated_people_count, 4)

    def test_lmstudio_captioner_posts_location_request(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "location_name": "Mogao Caves, Dunhuang, Gansu, China",
                                "gps_latitude": "39.9361",
                                "gps_longitude": "94.8076",
                            }
                        )
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
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "location_payload",
            )
            self.assertEqual(
                payload["messages"][1]["content"][0]["text"],
                "Resolve the location",
            )
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"not-a-real-jpeg")
            with (
                mock.patch.object(
                    _caption_lmstudio,
                    "_build_data_url",
                    return_value="data:image/jpeg;base64,abc123",
                ),
                mock.patch.object(
                    _caption_lmstudio,
                    "_select_lmstudio_model",
                    return_value="qwen2.5-vl",
                ),
                mock.patch.object(
                    _caption_lmstudio.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
            ):
                captioner = ai_caption.LMStudioCaptioner(
                    prompt_text="Describe this exact image",
                    base_url="http://127.0.0.1:1234",
                )
                details = captioner.estimate_location(
                    image_path=image_path,
                    prompt="Resolve the location",
                )

        self.assertEqual(details.location_name, "Mogao Caves, Dunhuang, Gansu, China")
        self.assertEqual(details.gps_latitude, "39.9361")
        self.assertEqual(details.gps_longitude, "94.8076")

    def test_parse_lmstudio_structured_caption_rejects_invalid_json(self):
        with self.assertRaises(RuntimeError) as exc:
            ai_caption._parse_lmstudio_structured_caption("not json", finish_reason="stop")
        self.assertIn("raw='not json'", str(exc.exception))
        self.assertIn("finish_reason=stop", str(exc.exception))

    def test_parse_lmstudio_structured_caption_rejects_empty_content(self):
        with self.assertRaises(RuntimeError) as exc:
            ai_caption._parse_lmstudio_structured_caption("", finish_reason="length")
        self.assertIn("finish_reason=length", str(exc.exception))

    def test_parse_lmstudio_structured_caption_extracts_json_after_think_prefix(self):
        details = ai_caption._parse_lmstudio_structured_caption(
            '<think>{ "caption": "A blue album cover labeled MAINLAND CHINA 1986 BOOK 11.", "gps_latitude": "", "gps_longitude": "", "location_name": "" }',
            finish_reason="stop",
        )
        self.assertEqual(details.text, "A blue album cover labeled MAINLAND CHINA 1986 BOOK 11.")

    def test_parse_lmstudio_structured_caption_strips_closed_think_block(self):
        # Thinking models like QwQ emit <think>reasoning</think> before the JSON payload.
        details = ai_caption._parse_lmstudio_structured_caption(
            "<think>Let me analyze the image carefully. I can see several photographs of a Chinese Opera performance in Lanzhou and travel scenes.</think>"
            '{ "caption": "A photo album page from Mainland China Book 11 displays five distinct photographs documenting a Chinese Opera performance in Lanzhou.", '
            '"gps_latitude": "", "gps_longitude": "", "location_name": "Lanzhou, Gansu, China" }',
            finish_reason="stop",
        )
        self.assertIn("Lanzhou", details.text)
        self.assertEqual(details.location_name, "Lanzhou, Gansu, China")
        self.assertNotIn("<think>", details.text)

    def test_parse_lmstudio_structured_caption_prefers_last_valid_payload(self):
        details = ai_caption._parse_lmstudio_structured_caption(
            '{ "caption": {"effective_engine": "lmstudio"} }'
            '\n{"caption": "Four people stand together outdoors in front of a brick house.", "location_name": ""}',
            finish_reason="stop",
        )
        self.assertEqual(
            details.text,
            "Four people stand together outdoors in front of a brick house.",
        )

    def test_parse_lmstudio_structured_caption_prefers_structured_gps_fields(self):
        details = ai_caption._parse_lmstudio_structured_caption(
            json.dumps(
                {
                    "caption": "The Mogao Caves entrance in Dunhuang.",
                    "gps_latitude": "39.7875",
                    "gps_longitude": "100.307222",
                    "location_name": "Mogao Caves, Dunhuang, Gansu, China",
                }
            )
        )
        self.assertEqual(details.gps_latitude, "39.7875")
        self.assertEqual(details.gps_longitude, "100.307222")
        self.assertEqual(details.location_name, "Mogao Caves, Dunhuang, Gansu, China")

    def test_parse_qwen_json_output_extracts_caption_from_json(self):
        raw = '{"caption": "Two people stand beside a red car.", "location_name": "", "gps_latitude": "", "gps_longitude": ""}'
        details = ai_caption._parse_qwen_json_output(raw)
        self.assertEqual(details.text, "Two people stand beside a red car.")
        self.assertEqual(details.gps_latitude, "")

    def test_parse_qwen_json_output_strips_think_block_before_json(self):
        raw = '<think>Let me analyze this.</think>{"caption": "A mountain valley.", "location_name": "", "gps_latitude": "", "gps_longitude": ""}'
        details = ai_caption._parse_qwen_json_output(raw)
        self.assertEqual(details.text, "A mountain valley.")

    def test_parse_qwen_json_output_falls_back_to_plain_text(self):
        raw = "A plain text caption with no JSON."
        details = ai_caption._parse_qwen_json_output(raw)
        self.assertEqual(details.text, "A plain text caption with no JSON.")


if __name__ == "__main__":
    unittest.main()
