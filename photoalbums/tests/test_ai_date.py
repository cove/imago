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

from photoalbums.lib import ai_date


class TestAIDate(unittest.TestCase):
    def test_date_estimate_prompt_includes_filename_year(self):
        prompt = ai_date._build_date_estimate_prompt(
            ocr_text="BILL GOODWIN IN THE PARK",
            album_title="Mainland China Book II",
            source_path=Path("MainlandChina_1986_B01_P04_D01-00_V.jpg"),
        )

        self.assertIn("Image file: MainlandChina_1986_B01_P04_D01-00_V.jpg", prompt)
        self.assertIn("Filename year: 1986", prompt)
        self.assertIn("use the filename year as the fallback year", prompt)

    def test_parse_date_estimate_coerces_zero_day_to_month_precision(self):
        self.assertEqual(
            ai_date._parse_date_estimate('{"date":"1988-01-00"}'),
            "1988-01",
        )

    def test_parse_date_estimate_coerces_zero_month_and_day_to_year_precision(self):
        self.assertEqual(
            ai_date._parse_date_estimate('{"date":"1988-00-00"}'),
            "1988",
        )

    def test_parse_date_estimate_rejects_invalid_nonzero_month(self):
        with self.assertRaisesRegex(RuntimeError, "invalid dc:date value"):
            ai_date._parse_date_estimate('{"date":"1988-13-00"}')

    def test_date_estimate_engine_falls_back_to_next_model(self):
        attempted_models = []

        def fake_request(url, *, payload=None, timeout):
            attempted_models.append(payload["model"])
            if payload["model"] == "bad-model":
                raise RuntimeError("bad-model failed")
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"date":"1988-01-00"}'},
                    }
                ]
            }

        with (
            mock.patch.object(ai_date, "default_caption_models", return_value=["bad-model", "good-model"]),
            mock.patch.object(ai_date, "default_caption_model", return_value="bad-model"),
            mock.patch.object(ai_date, "_lmstudio_request_json", side_effect=fake_request),
        ):
            engine = ai_date.DateEstimateEngine(engine="lmstudio", lmstudio_base_url="http://127.0.0.1:1234")
            result = engine.estimate(ocr_text="JAN 1988", album_title="Album")

        self.assertFalse(result.fallback)
        self.assertEqual(result.date, "1988-01")
        self.assertEqual(attempted_models, ["bad-model", "good-model"])
        self.assertEqual(engine.effective_model_name, "good-model")

    def test_default_date_token_budget_exceeds_legacy_cap(self):
        # Reasoning models truncated at the old 128-token cap before emitting the date.
        self.assertGreater(ai_date.DEFAULT_DATE_MAX_TOKENS, 128)

    def test_date_request_uses_full_token_budget_without_clamp(self):
        """The configured max_tokens must reach the request unchanged (no 128 clamp),
        so reasoning models have room to emit the date JSON after their preamble."""
        captured: dict[str, object] = {}

        def fake_request(url, *, payload=None, timeout):
            captured["max_tokens"] = payload["max_tokens"]
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"date":"1988"}'}}
                ]
            }

        with (
            mock.patch.object(ai_date, "default_caption_models", return_value=["m"]),
            mock.patch.object(ai_date, "default_caption_model", return_value="m"),
            mock.patch.object(ai_date, "_lmstudio_request_json", side_effect=fake_request),
        ):
            engine = ai_date.DateEstimateEngine(
                engine="lmstudio", lmstudio_base_url="http://127.0.0.1:1234", max_tokens=4096
            )
            result = engine.estimate(ocr_text="1988", album_title="Album")

        self.assertEqual(result.date, "1988")
        self.assertEqual(captured["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
