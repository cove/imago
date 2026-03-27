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

from photoalbums.lib import ai_caption, _caption_lmstudio


class TestAICaption(unittest.TestCase):
    def test_infer_album_title_falls_back_to_path_when_no_ocr(self):
        text = ai_caption.infer_album_title(
            image_path=Path("Photo Albums") / "China_1986_B02_View" / "China_1986_B02_P01.jpg",
        )
        self.assertEqual(text, "China 1986 Book II")

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

    def test_build_local_prompt_is_concise_and_context_driven(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_View" / "Family_1980-1985_B08_P01.jpg",
        )
        self.assertIn("Classify visible text into `author_text` and `scene_text`.", prompt)
        self.assertIn("Return empty strings when no applicable text exists for a field.", prompt)
        self.assertIn("classified subsets of `ocr_text`, not replacements for it", prompt)
        self.assertIn("Fill them whenever the classification is supported", prompt)
        self.assertNotIn("Album title hint:", prompt)
        self.assertNotIn("Album classification hint:", prompt)
        self.assertNotIn("Detected objects:", prompt)
        self.assertNotIn("Treat album title hints and classification hints as supporting context", prompt)
        self.assertNotIn("reproduce `BOOK 11` exactly as printed", prompt)
        self.assertNotIn("Filename hint:", prompt)
        self.assertNotIn("Folder hint:", prompt)

    def test_build_local_prompt_omits_album_title_hints(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "China_1986_B02_View" / "China_1986_B02_P02.jpg",
            album_title="Mainland China Book II",
            printed_album_title="Mainland China Book 11",
        )
        self.assertNotIn("Album title hint:", prompt)
        self.assertNotIn("Canonical album title hint:", prompt)
        self.assertNotIn("Prefer the printed cover title over the normalized title", prompt)

    def test_build_local_prompt_groups_runtime_hints_into_single_block(self):
        prompt = ai_caption._build_local_prompt(
            people=["Alice Example"],
            objects=["bench"],
            ocr_text="FAMILY BOOK",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_View" / "Family_1980-1985_B08_P01.jpg",
            album_title="Family Book I",
        )
        self.assertIn("author_text", prompt)
        self.assertIn("scene_text", prompt)
        self.assertNotIn("Detected objects:", prompt)
        self.assertNotIn("OCR text hint:", prompt)

    def test_system_prompts_load_from_skill_sections(self):
        self.assertIn("Count clearly visible real people only.", ai_caption.people_count_system_prompt())
        self.assertIn("leave GPS fields empty", ai_caption.location_system_prompt())

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
                    ocr_text="MAINLAND CHINA 1986",
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

    def test_legacy_local_alias_routes_to_lmstudio_and_returns_empty_on_error(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.side_effect = RuntimeError("model offline")
        with (
            mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio) as ctor,
            mock.patch("photoalbums.lib.ai_caption.default_caption_model", return_value=""),
        ):
            engine = ai_caption.CaptionEngine(engine="local")
            out = engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
            )
        ctor.assert_called_once_with(
            model_name="",
            prompt_text="",
            max_new_tokens=96,
            temperature=0.2,
            base_url=ai_caption.DEFAULT_LMSTUDIO_BASE_URL,
            max_image_edge=0,
            stream=False,
        )
        self.assertEqual(engine.engine, "lmstudio")
        self.assertEqual(out.engine, "lmstudio")
        self.assertTrue(out.fallback)
        self.assertIn("model offline", out.error)
        self.assertEqual(out.text, "")

    def test_lmstudio_engine_forwards_caption_settings(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(text="caption text")
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio) as ctor:
            engine = ai_caption.CaptionEngine(
                engine="lmstudio",
                model_name="qwen/qwen3.5-9b",
                caption_prompt="Describe this exact image",
                max_tokens=64,
                temperature=0.1,
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
            base_url=ai_caption.DEFAULT_LMSTUDIO_BASE_URL,
            max_image_edge=1024,
            stream=False,
        )
        self.assertEqual(out.engine, "lmstudio")
        self.assertEqual(out.text, "caption text")

    def test_generate_records_prompt_debug_metadata(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(text="caption text")
        fake_lmstudio._resolved_model_name = ""
        records = []
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(
                engine="lmstudio",
                model_name="qwen/qwen3.5-9b",
                caption_prompt="Describe this exact image",
            )
            engine.generate(
                image_path="sample.jpg",
                people=["Alice"],
                objects=["car"],
                ocr_text="",
                source_path="sample.jpg",
                debug_recorder=lambda **row: records.append(row),
                debug_step="caption_refresh",
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], "caption_refresh")
        self.assertEqual(records[0]["engine"], "lmstudio")
        self.assertEqual(records[0]["model"], "qwen/qwen3.5-9b")
        self.assertEqual(records[0]["prompt"], "Describe this exact image")
        self.assertEqual(records[0]["prompt_source"], "custom")

    def test_lmstudio_captioner_posts_chat_completion_request(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ocr_text": "Temple of Heaven\nNO SMOKING",
                                "author_text": "Temple of Heaven",
                                "scene_text": "NO SMOKING",
                                "location_name": "",
                                "album_title": "",
                                "ocr_lang": "en",
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
            self.assertEqual(payload["messages"][0]["content"], "")
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

        self.assertEqual(details.text, "Temple of Heaven")
        self.assertEqual(details.ocr_text, "Temple of Heaven\nNO SMOKING")
        self.assertEqual(details.author_text, "Temple of Heaven")
        self.assertEqual(details.scene_text, "NO SMOKING")
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
                payload["messages"][0]["content"],
                ai_caption.people_count_system_prompt(),
            )
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
                payload["messages"][0]["content"],
                ai_caption.location_system_prompt(),
            )
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
            '<think>{ "ocr_text": "MAINLAND CHINA 1986 BOOK 11", "author_text": "MAINLAND CHINA 1986 BOOK 11", "scene_text": "", "gps_latitude": "", "gps_longitude": "", "location_name": "", "album_title": "", "ocr_lang": "en" }',
            finish_reason="stop",
        )
        self.assertEqual(details.text, "MAINLAND CHINA 1986 BOOK 11")
        self.assertEqual(details.ocr_text, "MAINLAND CHINA 1986 BOOK 11")
        self.assertEqual(details.author_text, "MAINLAND CHINA 1986 BOOK 11")

    def test_parse_lmstudio_structured_caption_strips_closed_think_block(self):
        # Thinking models like QwQ emit <think>reasoning</think> before the JSON payload.
        details = ai_caption._parse_lmstudio_structured_caption(
            "<think>Let me analyze the image carefully. I can see several photographs of a Chinese Opera performance in Lanzhou and travel scenes.</think>"
            '{ "ocr_text": "Chinese Opera", "author_text": "Chinese Opera", "scene_text": "", '
            '"gps_latitude": "", "gps_longitude": "", "location_name": "Lanzhou, Gansu, China", "album_title": "", "ocr_lang": "en" }',
            finish_reason="stop",
        )
        self.assertEqual(details.text, "Chinese Opera")
        self.assertEqual(details.ocr_text, "Chinese Opera")
        self.assertEqual(details.location_name, "Lanzhou, Gansu, China")
        self.assertNotIn("<think>", details.text)

    def test_parse_lmstudio_structured_caption_prefers_last_valid_payload(self):
        details = ai_caption._parse_lmstudio_structured_caption(
            '{ "caption": {"effective_engine": "lmstudio"} }'
            '\n{"ocr_text": "Cordell Home", "author_text": "Cordell Home", "scene_text": "", "location_name": "", "album_title": "", "ocr_lang": "en"}',
            finish_reason="stop",
        )
        self.assertEqual(details.text, "Cordell Home")
        self.assertEqual(details.ocr_text, "Cordell Home")

    def test_parse_lmstudio_structured_caption_prefers_structured_gps_fields(self):
        details = ai_caption._parse_lmstudio_structured_caption(
            json.dumps(
                {
                    "ocr_text": "Mogao Caves",
                    "author_text": "Mogao Caves",
                    "scene_text": "",
                    "gps_latitude": "39.7875",
                    "gps_longitude": "100.307222",
                    "location_name": "Mogao Caves, Dunhuang, Gansu, China",
                    "album_title": "",
                    "ocr_lang": "en",
                }
            )
        )
        self.assertEqual(details.author_text, "Mogao Caves")
        self.assertEqual(details.ocr_text, "Mogao Caves")
        self.assertEqual(details.gps_latitude, "39.7875")
        self.assertEqual(details.gps_longitude, "100.307222")
        self.assertEqual(details.location_name, "Mogao Caves, Dunhuang, Gansu, China")


if __name__ == "__main__":
    unittest.main()
