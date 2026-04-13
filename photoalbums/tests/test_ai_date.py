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


if __name__ == "__main__":
    unittest.main()
