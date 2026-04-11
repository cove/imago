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

from photoalbums.lib import ai_ctm_restoration


class TestAICTMRestoration(unittest.TestCase):
    def test_parse_ctm_response_accepts_valid_payload(self):
        result = ai_ctm_restoration.parse_ctm_response(
            '{"matrix":[1.05,-0.02,-0.01,-0.04,1.35,-0.01,-0.03,-0.02,1.55],"confidence":0.91,"warnings":[],"reasoning_summary":"conservative"}',
            model_name="gemma",
            source_path="image.jpg",
        )
        self.assertEqual(len(result.matrix), 9)
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertEqual(result.model_name, "gemma")
        self.assertEqual(result.source_path, "image.jpg")

    def test_parse_ctm_response_rejects_invalid_json(self):
        with self.assertRaises(ai_ctm_restoration.CTMValidationError):
            ai_ctm_restoration.parse_ctm_response("not-json")

    def test_validate_ctm_result_rejects_excessive_coefficients(self):
        result = ai_ctm_restoration.CTMResult(
            matrix=[10.0, 0, 0, 0, 1, 0, 0, 0, 1],
            confidence=1.0,
        )
        with self.assertRaises(ai_ctm_restoration.CTMValidationError):
            ai_ctm_restoration.validate_ctm_result(result)

    def test_generate_and_store_ctm_retries_until_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.jpg"
            image.write_bytes(b"fake-jpeg")
            responses = [
                {"choices": [{"message": {"content": "oops"}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"matrix":[1.0,0,0,0,1.0,0,0,0,1.0],"confidence":0.8,"warnings":[],"reasoning_summary":"ok"}'
                            }
                        }
                    ]
                },
            ]
            with (
                mock.patch.object(ai_ctm_restoration, "_post_json", side_effect=responses),
                mock.patch.object(ai_ctm_restoration, "default_ctm_model", return_value="gemma"),
                mock.patch.object(
                    ai_ctm_restoration, "default_lmstudio_base_url", return_value="http://localhost:1234/v1"
                ),
            ):
                sidecar, result = ai_ctm_restoration.generate_and_store_ctm(image, force=True)

            self.assertTrue(sidecar.exists())
            self.assertEqual(len(result.matrix), 9)


if __name__ == "__main__":
    unittest.main()
