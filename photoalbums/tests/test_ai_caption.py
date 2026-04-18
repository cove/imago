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
    def test_build_local_prompt_is_concise_and_generic(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_Pages" / "Family_1980-1985_B08_P02.jpg",
        )
        self.assertIn("Use `author_text` for typewriter-written Courier text on white paper strips.", prompt)
        self.assertIn("Return empty strings when no applicable text exists for a field.", prompt)
        self.assertIn("classified subsets of `ocr_text`, not replacements for it", prompt)
        self.assertIn("Fill them whenever the classification is supported", prompt)
        self.assertIn("This page contains multiple photographs.", prompt)
        self.assertNotIn("This is an album cover or title page.", prompt)
        self.assertNotIn("Album title hint:", prompt)
        self.assertNotIn("Album classification hint:", prompt)
        self.assertNotIn("Preamble Combined", prompt)

    def test_build_local_prompt_mentions_split_typewritten_strips(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="SHOW AT THE DUNHUANG CULTURAL CENTRE",
            source_path=Path("Photo Albums") / "China_1986_B02_Archive" / "China_1986_B02_P02_S01.tif",
        )
        self.assertIn(
            "Recover the full `author_text` when the strip is visibly present but cropped in this scan", prompt
        )
        self.assertIn("the supplied `ocr_text` contains the missing words", prompt)

    def test_build_local_prompt_groups_runtime_hints_into_single_block(self):
        prompt = ai_caption._build_local_prompt(
            people=["Alice Example"],
            objects=["bench"],
            ocr_text="FAMILY BOOK",
            source_path=Path("Photo Albums") / "Family_1980-1985_B08_Pages" / "Family_1980-1985_B08_P01.jpg",
            album_title="Family Book I",
        )
        self.assertIn("author_text", prompt)
        self.assertIn("scene_text", prompt)
        self.assertNotIn("Detected objects:", prompt)
        self.assertNotIn("OCR text hint:", prompt)

    def test_build_local_prompt_includes_upstream_ocr_context_only_when_supplied(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="",
            source_path=Path("Photo Albums") / "China_1986_B02_Archive" / "China_1986_B02_P02_D01-01.tif",
            context_ocr_text="TEMPLE OF HEAVEN\nBEIJING",
        )
        self.assertIn("The following OCR text comes from the parent album page XMP and is context only", prompt)
        self.assertIn("TEMPLE OF HEAVEN\nBEIJING", prompt)
        self.assertIn("Do not copy words from this context", prompt)

    def test_build_local_prompt_includes_cover_page_prompt_for_title_pages(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
            source_path=Path("Photo Albums") / "China_1986_B02_Pages" / "China_1986_B02_P01.jpg",
        )
        self.assertIn("This is an album cover or title page.", prompt)
        self.assertIn("Use the OCR text from this page as the source of truth for `album_title`.", prompt)

    def test_system_prompts_load_from_skill_sections(self):
        self.assertIn("Count clearly visible real people only.", ai_caption.people_count_system_prompt())
        self.assertIn("leave GPS fields empty", ai_caption.location_system_prompt())

    def test_build_location_prompt_includes_album_and_ocr_hints(self):
        prompt = ai_caption._build_location_prompt(
            ocr_text="TEMPLE OF HEAVEN\nBEIJING",
            album_title="China 1986 Book 11",
        )
        self.assertIn("The album name is: China 1986 Book 11", prompt)
        self.assertIn("The OCR text on the page is: TEMPLE OF HEAVEN\nBEIJING", prompt)
        self.assertIn("prefer a disambiguating geocoding query", prompt)
        self.assertIn("Avoid ambiguous bare place names like `Oxford`", prompt)
        self.assertIn("Include country, state/province, or broad region", prompt)

    def test_build_location_prompt_falls_back_to_printed_album_title(self):
        prompt = ai_caption._build_location_prompt(
            ocr_text="MOGAO CAVES",
            album_title="",
            printed_album_title="Mainland China 1986 Book 11",
        )
        self.assertIn("The album name is: Mainland China 1986 Book 11", prompt)
        self.assertIn("The OCR text on the page is: MOGAO CAVES", prompt)

    def test_build_local_prompt_includes_disambiguating_location_query_guidance(self):
        prompt = ai_caption._build_local_prompt(
            people=[],
            objects=[],
            ocr_text="LONDON GARDENS",
            source_path=Path("Photo Albums") / "Europe_1973_Archive" / "Europe_1973_B01_P02_S01.tif",
        )
        self.assertIn("Prefer `place, city/state, country` style queries", prompt)

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

    def test_generate_does_not_mark_scene_text_page_caption_as_fallback(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(
            text="",
            ocr_text="SHOW AT THE DUNHUANG CULTURAL CENTRE",
            author_text="",
            scene_text="EXHIBIT HISTORICAL RELICS OF DUNHUANG",
            image_regions=[{"author_text": "SHOW AT THE DUNHUANG CULTURAL CENTRE", "scene_text": ""}],
        )
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(engine="lmstudio")
            out = engine.generate(
                image_path="sample.jpg",
                people=[],
                objects=[],
                ocr_text="SHOW AT THE DUNHUANG CULTURAL CENTRE",
            )
        self.assertFalse(out.fallback)
        self.assertEqual(out.error, "")
        self.assertEqual(out.scene_text, "EXHIBIT HISTORICAL RELICS OF DUNHUANG")

    def test_generate_does_not_mark_location_only_caption_as_fallback(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(
            text="",
            ocr_text="",
            author_text="",
            scene_text="",
            location_name="Dubrovnik",
        )
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(engine="lmstudio")
            out = engine.generate(
                image_path="sample.jpg",
                people=[],
                objects=[],
                ocr_text="",
            )
        self.assertFalse(out.fallback)
        self.assertEqual(out.error, "")
        self.assertEqual(out.location_name, "Dubrovnik")

    def test_generate_does_not_mark_photo_region_only_caption_as_fallback(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(
            text="",
            ocr_text="",
            author_text="",
            scene_text="",
            image_regions=[{"x": 0.11, "y": 0.044, "w": 0.469, "h": 0.492}],
        )
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(engine="lmstudio")
            out = engine.generate(
                image_path="sample.jpg",
                people=[],
                objects=[],
                ocr_text="",
            )
        self.assertFalse(out.fallback)
        self.assertEqual(out.error, "")
        self.assertEqual(out.image_regions, [{"x": 0.11, "y": 0.044, "w": 0.469, "h": 0.492}])

    def test_lmstudio_engine_forwards_caption_settings(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.describe_page.return_value = ai_caption.CaptionDetails(text="caption text")
        with (
            mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio) as ctor,
            mock.patch("photoalbums.lib.ai_caption.default_lmstudio_base_url", return_value=ai_caption.DEFAULT_LMSTUDIO_BASE_URL),
        ):
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
            model_name=["qwen/qwen3.5-9b"],
            prompt_text="Describe this exact image",
            max_new_tokens=64,
            temperature=0.1,
            base_url=ai_caption.DEFAULT_LMSTUDIO_BASE_URL,
            max_image_edge=1024,
            stream=False,
        )
        self.assertEqual(out.engine, "lmstudio")
        self.assertEqual(out.text, "caption text")

    def test_lmstudio_captioner_falls_back_to_next_model(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ocr_text": "Temple of Heaven",
                                "author_text": "Temple of Heaven",
                                "scene_text": "",
                                "location_name": "",
                                "album_title": "",
                                "ocr_lang": "en",
                            }
                        )
                    }
                }
            ]
        }

        attempted_models = []

        def fake_request(url, *, payload=None, timeout):
            attempted_models.append(payload["model"])
            if payload["model"] == "bad-model":
                raise RuntimeError("bad-model failed")
            return response_payload

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"not-a-real-jpeg")
            with (
                mock.patch.object(_caption_lmstudio, "_build_data_url", return_value="data:image/jpeg;base64,abc123"),
                mock.patch.object(_caption_lmstudio, "_lmstudio_request_json", side_effect=fake_request),
            ):
                captioner = ai_caption.LMStudioCaptioner(
                    model_name=["bad-model", "good-model"],
                    prompt_text="Describe this exact image",
                    base_url="http://127.0.0.1:1234",
                )
                details = captioner.describe(
                    image_path=image_path,
                    prompt="Describe this exact image",
                )

        self.assertEqual(attempted_models, ["bad-model", "good-model"])
        self.assertEqual(captioner._resolved_model_name, "good-model")
        self.assertEqual(details.author_text, "Temple of Heaven")

   
    def test_estimate_locations_shown_records_shared_location_prompt(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.estimate_locations_shown.return_value = ai_caption.CaptionDetails(
            text="",
            locations_shown=[{"name": "Hassan II Mosque", "country_name": "Morocco"}],
        )
        fake_lmstudio._resolved_model_name = ""
        fake_lmstudio.last_response_text = '{"locations_shown":[{"name":"Hassan II Mosque","country_name":"Morocco"}]}'
        fake_lmstudio.last_finish_reason = "stop"
        records = []
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(
                engine="lmstudio",
                model_name="qwen/qwen3.5-9b",
            )
            engine.estimate_locations_shown(
                image_path="sample.jpg",
                ocr_text="EASTERN EUROPE SPAIN AND MOROCCO 1988",
                album_title="Spain and Morocco 1988",
                source_path="sample.jpg",
                debug_recorder=lambda **row: records.append(row),
                debug_step="locations_shown_refresh",
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], "locations_shown_refresh")
        self.assertEqual(records[0]["system_prompt"], ai_caption.location_system_prompt())
        self.assertIn("If the response schema asks for `locations_shown`", records[0]["prompt"])
        self.assertIn("The album name is: Spain and Morocco 1988", records[0]["prompt"])
        self.assertIn("The OCR text on the page is: EASTERN EUROPE SPAIN AND MOROCCO 1988", records[0]["prompt"])
        self.assertIn("EASTERN EUROPE SPAIN AND MOROCCO 1988", records[0]["prompt"])
        self.assertEqual(records[0]["response"], fake_lmstudio.last_response_text)
        self.assertEqual(records[0]["finish_reason"], "stop")

    def test_estimate_location_records_album_and_ocr_hints_in_prompt(self):
        fake_lmstudio = mock.Mock()
        fake_lmstudio.estimate_location.return_value = ai_caption.CaptionDetails(
            text="",
            location_name="Mogao Caves, Dunhuang, Gansu, China",
            gps_latitude="",
            gps_longitude="",
        )
        fake_lmstudio._resolved_model_name = ""
        fake_lmstudio.last_response_text = (
            '{"location_name":"Mogao Caves, Dunhuang, Gansu, China","gps_latitude":"","gps_longitude":""}'
        )
        fake_lmstudio.last_finish_reason = "stop"
        records = []
        with mock.patch("photoalbums.lib.ai_caption.LMStudioCaptioner", return_value=fake_lmstudio):
            engine = ai_caption.CaptionEngine(
                engine="lmstudio",
                model_name="qwen/qwen3.5-9b",
            )
            engine.estimate_location(
                image_path="sample.jpg",
                people=[],
                objects=[],
                ocr_text="TEMPLE OF HEAVEN\nBEIJING",
                source_path="sample.jpg",
                album_title="China 1986 Book 11",
                debug_recorder=lambda **row: records.append(row),
                debug_step="location_refresh",
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], "location_refresh")
        self.assertEqual(records[0]["system_prompt"], ai_caption.location_system_prompt())
        self.assertIn("The album name is: China 1986 Book 11", records[0]["prompt"])
        self.assertIn("The OCR text on the page is: TEMPLE OF HEAVEN\nBEIJING", records[0]["prompt"])
        self.assertEqual(records[0]["response"], fake_lmstudio.last_response_text)
        self.assertEqual(records[0]["finish_reason"], "stop")

    def test_location_shown_system_prompt_aliases_shared_location_prompt(self):
        self.assertEqual(ai_caption.location_shown_system_prompt(), ai_caption.location_system_prompt())

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
            self.assertEqual(payload["response_format"]["json_schema"]["strict"], True)
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
                    _caption_lmstudio.LMStudioCaptioner,
                    "_select_model_name",
                    staticmethod(lambda *_: "qwen2.5-vl"),
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
                    _caption_lmstudio.LMStudioCaptioner,
                    "_select_model_name",
                    staticmethod(lambda *_: "qwen2.5-vl"),
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

    def test_lmstudio_stream_tokens_raises_sse_error_message_verbatim(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                yield b"event: error\n"
                yield (
                    b'data: {"error":{"message":"request (4196 tokens) exceeds the available context size '
                    b'(4096 tokens), try increasing it"},"message":"request (4196 tokens) exceeds the '
                    b'available context size (4096 tokens), try increasing it"}\n'
                )
                yield b"\n"

        with mock.patch.object(
            _caption_lmstudio.urllib.request,
            "urlopen",
            return_value=_FakeResponse(),
        ):
            with self.assertRaises(RuntimeError) as exc:
                list(_caption_lmstudio._lmstudio_stream_tokens("http://127.0.0.1:1234/v1/chat/completions", {}, 30))

        self.assertEqual(
            str(exc.exception),
            "LM Studio request failed: request (4196 tokens) exceeds the available context size "
            "(4096 tokens), try increasing it",
        )

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
                    _caption_lmstudio.LMStudioCaptioner,
                    "_select_model_name",
                    staticmethod(lambda *_: "qwen2.5-vl"),
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

    def test_lmstudio_captioner_posts_locations_shown_request_with_ocr_hints(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "locations_shown": [
                                    {
                                        "name": "Hassan II Mosque",
                                        "world_region": "Africa",
                                        "country_name": "Morocco",
                                        "country_code": "MA",
                                        "province_or_state": "",
                                        "city": "",
                                        "sublocation": "",
                                    }
                                ]
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
                "locations_shown_payload",
            )
            self.assertIn(
                "name",
                payload["response_format"]["json_schema"]["schema"]["properties"]["locations_shown"]["items"][
                    "properties"
                ],
            )
            self.assertIn(
                "If the response schema asks for `locations_shown`",
                payload["messages"][1]["content"][0]["text"],
            )
            self.assertIn(
                "EASTERN EUROPE SPAIN AND MOROCCO 1988",
                payload["messages"][1]["content"][0]["text"],
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
                    _caption_lmstudio.LMStudioCaptioner,
                    "_select_model_name",
                    staticmethod(lambda *_: "qwen2.5-vl"),
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
                captioner._resolved_model_name = "qwen2.5-vl"
                details = captioner.estimate_locations_shown(
                    image_path=image_path,
                    prompt=(
                        "If the response schema asks for `locations_shown`, identify distinct famous locations visible in the photographs on this page.\n"
                        "- The OCR text on the page is: EASTERN EUROPE SPAIN AND MOROCCO 1988\n"
                        "EASTERN EUROPE SPAIN AND MOROCCO 1988"
                    ),
                )

        self.assertEqual(
            details.locations_shown,
            [
                {
                    "name": "Hassan II Mosque",
                    "world_region": "Africa",
                    "country_name": "Morocco",
                    "country_code": "MA",
                    "province_or_state": "",
                    "city": "",
                    "sublocation": "",
                }
            ],
        )

    def test_parse_lmstudio_structured_caption_rejects_invalid_json(self):
        with self.assertRaises(RuntimeError) as exc:
            ai_caption._parse_lmstudio_structured_caption("not json", finish_reason="stop")
        self.assertIn("raw='not json'", str(exc.exception))
        self.assertIn("finish_reason=stop", str(exc.exception))

    def test_parse_lmstudio_structured_caption_rejects_empty_content(self):
        with self.assertRaises(RuntimeError) as exc:
            ai_caption._parse_lmstudio_structured_caption("", finish_reason="length")
        self.assertIn("finish_reason=length", str(exc.exception))

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

