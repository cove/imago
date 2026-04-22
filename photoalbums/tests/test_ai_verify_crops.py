from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_verify_crops, xmp_sidecar


class TestAIVerifyCrops(unittest.TestCase):
    def test_build_verification_prompt_includes_carry_over_and_month_year_guidance(self):
        prompt = ai_verify_crops.build_verification_prompt(
            page_image_name="Album_1988_B01_P03_V.jpg",
            crop_image_name="Album_1988_B01_P03_D01-00_V.jpg",
            page_xmp_text="author_text: AUG. 1988",
            crop_xmp_text="description: their new condominium",
        )

        self.assertIn("nearby-caption carry-over", prompt)
        self.assertIn("AUG. 1988", prompt)
        self.assertIn("1988-08", prompt)

    def test_parse_verification_payload_normalizes_good_result(self):
        payload = {
            "caption": {"verdict": "good", "reasoning": "Caption matches page context.", "failure_reason": "ignored"},
            "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
            "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
            "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
            "overall": {"verdict": "good", "reasoning": "Everything belongs together.", "failure_reason": ""},
            "human_inference": "",
            "needs_another_pass": ["caption"],
            "needs_human_review": ["date"],
        }

        result = ai_verify_crops.parse_verification_payload(payload)

        self.assertEqual(result["caption"]["failure_reason"], "")
        self.assertEqual(result["needs_another_pass"], [])
        self.assertEqual(result["needs_human_review"], [])

    def test_parse_verification_payload_requires_human_inference_for_bad_or_uncertain(self):
        payload = {
            "caption": {"verdict": "bad", "reasoning": "Caption belongs to nearby photo.", "failure_reason": "Wrong caption."},
            "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
            "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
            "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
            "overall": {"verdict": "bad", "reasoning": "Set does not belong together.", "failure_reason": "Caption mismatch."},
            "human_inference": "",
            "needs_another_pass": ["caption"],
            "needs_human_review": [],
        }

        with self.assertRaises(ValueError):
            ai_verify_crops.parse_verification_payload(payload)

    def test_parse_verification_payload_keeps_bad_and_uncertain_routing(self):
        payload = {
            "caption": {"verdict": "bad", "reasoning": "Caption belongs to nearby photo.", "failure_reason": "Wrong caption."},
            "gps": {"verdict": "uncertain", "reasoning": "Page context too vague.", "failure_reason": "Not enough place evidence."},
            "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
            "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
            "overall": {"verdict": "uncertain", "reasoning": "Some evidence conflicts.", "failure_reason": "Caption and GPS unresolved."},
            "human_inference": "Human would read this as a neighboring caption and a vague place reference.",
            "needs_another_pass": ["caption", "gps", "date"],
            "needs_human_review": ["gps", "shown_location"],
        }

        result = ai_verify_crops.parse_verification_payload(payload)

        self.assertEqual(result["needs_another_pass"], ["caption", "gps"])
        self.assertEqual(result["needs_human_review"], ["gps"])

    def test_load_page_verifier_inputs_reports_missing_page_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            page_path = Path(tmp) / "Album_1988_B01_P03_V.jpg"
            page_path.write_bytes(b"jpg")

            result = ai_verify_crops.load_page_verifier_inputs(page_path)

            self.assertEqual(result["missing_context"], ["page_xmp"])
            self.assertEqual(result["crops"], [])

    def test_persist_verify_crops_state_writes_page_and_crop_pipeline_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages_dir = root / "Album_1988_B01_Pages"
            photos_dir = root / "Album_1988_B01_Photos"
            pages_dir.mkdir()
            photos_dir.mkdir()

            page_image = pages_dir / "Album_1988_B01_P03_V.jpg"
            crop_image = photos_dir / "Album_1988_B01_P03_D01-00_V.jpg"
            page_image.write_bytes(b"jpg")
            crop_image.write_bytes(b"jpg")

            xmp_sidecar.write_xmp_sidecar(
                page_image.with_suffix(".xmp"),
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="Page caption",
                source_text="Album_1988_B01_P03_S01.tif",
                ocr_text="AUG. 1988",
                detections_payload={},
            )
            xmp_sidecar.write_xmp_sidecar(
                crop_image.with_suffix(".xmp"),
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="Crop caption",
                source_text="Album_1988_B01_P03_S01.tif",
                ocr_text="AUG. 1988",
                detections_payload={},
            )

            verify_result = {
                "status": "ok",
                "page_input_hash": "hash123",
                "missing_context": [],
                "artifact_path": str(root / "_debug" / "verify-crops" / "Album_1988_B01_P03_V.json"),
                "results": [
                    {
                        "crop_image_path": str(crop_image),
                        "crop_xmp_path": str(crop_image.with_suffix(".xmp")),
                        "model": "glm-test",
                        "review": {
                            "caption": {"verdict": "bad", "reasoning": "Caption shifted right.", "failure_reason": "Caption belongs to neighbor."},
                            "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                            "shown_location": {"verdict": "uncertain", "reasoning": "Page only says Spain.", "failure_reason": "Too vague for city."},
                            "date": {"verdict": "good", "reasoning": "Date matches page.", "failure_reason": ""},
                            "overall": {"verdict": "uncertain", "reasoning": "Caption and shown location unresolved.", "failure_reason": "Two concerns unresolved."},
                            "human_inference": "Human would infer neighboring caption carry-over and only country-level place evidence.",
                            "needs_another_pass": ["caption", "shown_location"],
                            "needs_human_review": [],
                        },
                    }
                ],
            }

            ai_verify_crops.persist_verify_crops_state(page_image, verify_result)

            page_state = xmp_sidecar.read_pipeline_step(page_image.with_suffix(".xmp"), "verify-crops")
            crop_state = xmp_sidecar.read_pipeline_step(crop_image.with_suffix(".xmp"), "verify-crops")
            assert page_state is not None
            assert crop_state is not None
            self.assertEqual(page_state["input_hash"], "hash123")
            self.assertEqual(page_state["reviewed_crop_count"], 1)
            self.assertEqual(page_state["needs_another_pass"], ["caption", "shown_location"])
            self.assertTrue(page_state["page_verification_ran"])
            concerns = dict(crop_state["concerns"])
            self.assertEqual(concerns["caption"]["status"], "bad")
            self.assertEqual(concerns["caption"]["failure_reason"], "Caption belongs to neighbor.")
            self.assertEqual(concerns["shown_location"]["status"], "uncertain")
            self.assertEqual(concerns["overall"]["status"], "uncertain")
            self.assertEqual(crop_state["human_inference"], verify_result["results"][0]["review"]["human_inference"])

    def test_run_verify_crops_page_runs_pass2_retry_before_returning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages_dir = root / "Album_1988_B01_Pages"
            photos_dir = root / "Album_1988_B01_Photos"
            pages_dir.mkdir()
            photos_dir.mkdir()

            page_image = pages_dir / "Album_1988_B01_P03_V.jpg"
            crop_image = photos_dir / "Album_1988_B01_P03_D01-00_V.jpg"
            page_image.write_bytes(b"jpg")
            crop_image.write_bytes(b"jpg")

            verifier_inputs = {
                "page_image_path": str(page_image.resolve()),
                "page_xmp_path": str(page_image.with_suffix(".xmp").resolve()),
                "page_image_exists": True,
                "page_xmp_exists": True,
                "page_xmp_text": "page context",
                "missing_context": [],
                "crops": [
                    {
                        "crop_image_path": str(crop_image.resolve()),
                        "crop_xmp_path": str(crop_image.with_suffix(".xmp").resolve()),
                        "crop_xmp_exists": True,
                        "crop_xmp_text": "crop context",
                    }
                ],
            }
            parsed_review = {
                "caption": {"verdict": "bad", "reasoning": "Caption drifted to a neighboring crop.", "failure_reason": "Caption belongs to the crop on the left."},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "bad", "reasoning": "Caption mismatch makes the set unreliable.", "failure_reason": "Caption mismatch."},
                "human_inference": "Human would attach the nearby caption to the left crop instead.",
                "needs_another_pass": ["caption"],
                "needs_human_review": [],
            }
            follow_up_review = {
                "caption": {"verdict": "good", "reasoning": "Caption now matches the page context.", "failure_reason": ""},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "good", "reasoning": "Everything now belongs together.", "failure_reason": ""},
                "human_inference": "",
                "needs_another_pass": [],
                "needs_human_review": [],
            }

            with (
                patch.object(ai_verify_crops, "load_page_verifier_inputs", return_value=verifier_inputs),
                patch.object(
                    ai_verify_crops,
                    "_call_structured_vision_request",
                    side_effect=[
                        (parsed_review, "glm-test", "{\"ok\": true}", "stop"),
                        (follow_up_review, "glm-test", "{\"ok\": true}", "stop"),
                    ],
                ),
                patch.object(
                    ai_verify_crops,
                    "_run_pass2_retry",
                    return_value={
                        "pass": 2,
                        "concern": "caption",
                        "prompt_variant": "retry",
                        "issue": "Caption drifted to a neighboring crop.",
                        "failure_reason": "Caption belongs to the crop on the left.",
                        "before_value": "Wrong caption",
                        "after_value": "Fixed caption",
                        "changed": True,
                    },
                ) as retry_mock,
            ):
                result = ai_verify_crops.run_verify_crops_page(page_image)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["results"][0]["retry_attempts"][0]["concern"], "caption")
            self.assertEqual(result["results"][0]["review"]["caption"]["verdict"], "good")
            retry_mock.assert_called_once()
            artifact = Path(result["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("\"retry_attempts\"", artifact)

    def test_run_verify_crops_page_logs_before_after_retry_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages_dir = root / "Album_1988_B01_Pages"
            photos_dir = root / "Album_1988_B01_Photos"
            pages_dir.mkdir()
            photos_dir.mkdir()

            page_image = pages_dir / "Album_1988_B01_P03_V.jpg"
            crop_image = photos_dir / "Album_1988_B01_P03_D01-00_V.jpg"
            page_image.write_bytes(b"jpg")
            crop_image.write_bytes(b"jpg")

            verifier_inputs = {
                "page_image_path": str(page_image.resolve()),
                "page_xmp_path": str(page_image.with_suffix(".xmp").resolve()),
                "page_image_exists": True,
                "page_xmp_exists": True,
                "page_xmp_text": "page context",
                "missing_context": [],
                "crops": [
                    {
                        "crop_image_path": str(crop_image.resolve()),
                        "crop_xmp_path": str(crop_image.with_suffix(".xmp").resolve()),
                        "crop_xmp_exists": True,
                        "crop_xmp_text": "crop context",
                    }
                ],
            }
            parsed_review = {
                "caption": {"verdict": "bad", "reasoning": "Caption drifted to a neighboring crop.", "failure_reason": "Caption belongs to the crop on the left."},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "bad", "reasoning": "Caption mismatch makes the set unreliable.", "failure_reason": "Caption mismatch."},
                "human_inference": "Human would attach the nearby caption to the left crop instead.",
                "needs_another_pass": ["caption"],
                "needs_human_review": [],
            }
            messages: list[str] = []

            with (
                patch.object(ai_verify_crops, "load_page_verifier_inputs", return_value=verifier_inputs),
                patch.object(
                    ai_verify_crops,
                    "_call_structured_vision_request",
                    side_effect=[
                        (parsed_review, "glm-test", "{\"ok\": true}", "stop"),
                        (
                            {
                                "caption_max_tokens": 144,
                                "caption_temperature": 0.1,
                                "caption_max_edge": 2048,
                                "reason": "Try a tighter caption rerun.",
                            },
                            "glm-test",
                            "{\"caption_max_tokens\": 144}",
                            "stop",
                        ),
                    ],
                ),
                patch.object(
                    ai_verify_crops,
                    "_run_pass2_retry",
                    side_effect=[
                        {
                            "pass": 2,
                            "concern": "caption",
                            "prompt_variant": "retry",
                            "issue": "Caption drifted to a neighboring crop.",
                            "failure_reason": "Caption belongs to the crop on the left.",
                            "before_value": "Wrong caption",
                            "after_value": "Wrong caption",
                            "changed": False,
                        },
                        {
                            "pass": 3,
                            "concern": "caption",
                            "prompt_variant": "parameter-suggestion",
                            "issue": "Caption drifted to a neighboring crop.",
                            "failure_reason": "Caption belongs to the crop on the left.",
                            "before_value": "Wrong caption",
                            "after_value": "Wrong caption",
                            "changed": False,
                        },
                    ],
                ),
            ):
                ai_verify_crops.run_verify_crops_page(page_image, logger=messages.append)

            self.assertTrue(any("Wrong caption -> Wrong caption (unchanged)" in message for message in messages))

    def test_run_verify_crops_page_runs_pass3_parameter_suggestion_with_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages_dir = root / "Album_1988_B01_Pages"
            photos_dir = root / "Album_1988_B01_Photos"
            pages_dir.mkdir()
            photos_dir.mkdir()

            page_image = pages_dir / "Album_1988_B01_P03_V.jpg"
            crop_image = photos_dir / "Album_1988_B01_P03_D01-00_V.jpg"
            page_image.write_bytes(b"jpg")
            crop_image.write_bytes(b"jpg")

            verifier_inputs = {
                "page_image_path": str(page_image.resolve()),
                "page_xmp_path": str(page_image.with_suffix(".xmp").resolve()),
                "page_image_exists": True,
                "page_xmp_exists": True,
                "page_xmp_text": "page context AUG. 1988",
                "missing_context": [],
                "crops": [
                    {
                        "crop_image_path": str(crop_image.resolve()),
                        "crop_xmp_path": str(crop_image.with_suffix(".xmp").resolve()),
                        "crop_xmp_exists": True,
                        "crop_xmp_text": "crop context wrong caption",
                    }
                ],
            }
            first_review = {
                "caption": {"verdict": "bad", "reasoning": "Caption drifted to a neighboring crop.", "failure_reason": "Caption belongs to the crop on the left."},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "bad", "reasoning": "Caption mismatch makes the set unreliable.", "failure_reason": "Caption mismatch."},
                "human_inference": "Human would attach the nearby caption to the left crop instead.",
                "needs_another_pass": ["caption"],
                "needs_human_review": [],
            }
            second_review = {
                "caption": {"verdict": "bad", "reasoning": "Caption is still mismatched.", "failure_reason": "Caption still belongs to the neighboring crop."},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "bad", "reasoning": "Caption remains unresolved.", "failure_reason": "Caption mismatch remains."},
                "human_inference": "Human would still attach the caption to the neighboring crop instead.",
                "needs_another_pass": ["caption"],
                "needs_human_review": [],
            }
            third_review = {
                "caption": {"verdict": "uncertain", "reasoning": "Caption is still not trustworthy after the third pass.", "failure_reason": "Caption remains ambiguous between neighboring crops."},
                "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                "date": {"verdict": "good", "reasoning": "Date matches page text.", "failure_reason": ""},
                "overall": {"verdict": "uncertain", "reasoning": "Caption is still unresolved overall.", "failure_reason": "Caption ambiguity remains."},
                "human_inference": "Human would still see the caption as ambiguous between adjacent crops.",
                "needs_another_pass": ["caption"],
                "needs_human_review": [],
            }

            with (
                patch.object(ai_verify_crops, "load_page_verifier_inputs", return_value=verifier_inputs),
                patch.object(
                    ai_verify_crops,
                    "_call_structured_vision_request",
                    side_effect=[
                        (first_review, "glm-test", "{\"ok\": true}", "stop"),
                        (second_review, "glm-test", "{\"ok\": true}", "stop"),
                        (
                            {
                                "caption_max_tokens": 144,
                                "caption_temperature": 0.1,
                                "caption_max_edge": 2048,
                                "reason": "Use a slightly larger crop context window.",
                            },
                            "glm-test",
                            "{\"caption_max_tokens\": 144}",
                            "stop",
                        ),
                        (third_review, "glm-test", "{\"ok\": true}", "stop"),
                    ],
                ) as request_mock,
                patch.object(ai_verify_crops, "_current_crop_xmp_text", return_value="crop context revised"),
                patch.object(
                    ai_verify_crops,
                    "_run_pass2_retry",
                    side_effect=[
                        {
                            "pass": 2,
                            "concern": "caption",
                            "prompt_variant": "retry",
                            "issue": "Caption drifted to a neighboring crop.",
                            "failure_reason": "Caption belongs to the crop on the left.",
                            "before_value": "Wrong caption",
                            "after_value": "Still wrong caption",
                            "changed": True,
                            "model": "glm-test",
                            "tuning_params": {
                                "caption_max_tokens": 96,
                                "caption_temperature": 0.2,
                                "caption_max_edge": 0,
                            },
                        },
                        {
                            "pass": 3,
                            "concern": "caption",
                            "prompt_variant": "parameter-suggestion",
                            "issue": "Caption is still mismatched.",
                            "failure_reason": "Caption still belongs to the neighboring crop.",
                            "before_value": "Still wrong caption",
                            "after_value": "Different but unresolved caption",
                            "changed": True,
                            "model": "glm-test",
                            "tuning_params": {
                                "caption_max_tokens": 144,
                                "caption_temperature": 0.1,
                                "caption_max_edge": 2048,
                            },
                        },
                    ],
                ),
            ):
                result = ai_verify_crops.run_verify_crops_page(page_image)

            row = result["results"][0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(row["retry_attempts"]), 2)
            self.assertEqual(row["retry_attempts"][1]["prompt_variant"], "parameter-suggestion")
            self.assertEqual(row["review"]["needs_another_pass"], [])
            self.assertEqual(row["review"]["needs_human_review"], ["caption"])
            self.assertEqual(row["review_provenance"]["caption"]["retry_count"], 2)
            self.assertEqual(row["review_provenance"]["caption"]["tuning_params"]["caption_max_tokens"], 144)

            suggest_kwargs = request_mock.call_args_list[2].kwargs
            self.assertEqual(suggest_kwargs["page_image_path"], page_image)
            self.assertEqual(suggest_kwargs["crop_image_path"], crop_image.resolve())
            self.assertIn("page context AUG. 1988", suggest_kwargs["prompt"])
            self.assertIn("crop context revised", suggest_kwargs["prompt"])

    def test_persist_verify_crops_state_records_retry_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages_dir = root / "Album_1988_B01_Pages"
            photos_dir = root / "Album_1988_B01_Photos"
            pages_dir.mkdir()
            photos_dir.mkdir()

            page_image = pages_dir / "Album_1988_B01_P03_V.jpg"
            crop_image = photos_dir / "Album_1988_B01_P03_D01-00_V.jpg"
            page_image.write_bytes(b"jpg")
            crop_image.write_bytes(b"jpg")

            xmp_sidecar.write_xmp_sidecar(
                page_image.with_suffix(".xmp"),
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="Page caption",
                source_text="Album_1988_B01_P03_S01.tif",
                ocr_text="AUG. 1988",
                detections_payload={},
            )
            xmp_sidecar.write_xmp_sidecar(
                crop_image.with_suffix(".xmp"),
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="Crop caption",
                source_text="Album_1988_B01_P03_S01.tif",
                ocr_text="AUG. 1988",
                detections_payload={},
            )

            verify_result = {
                "status": "ok",
                "page_input_hash": "hash123",
                "missing_context": [],
                "artifact_path": str(root / "_debug" / "verify-crops" / "Album_1988_B01_P03_V.json"),
                "results": [
                    {
                        "crop_image_path": str(crop_image),
                        "crop_xmp_path": str(crop_image.with_suffix(".xmp")),
                        "model": "glm-test",
                        "review": {
                            "caption": {"verdict": "good", "reasoning": "Caption matches after pass 3.", "failure_reason": ""},
                            "gps": {"verdict": "good", "reasoning": "GPS is supported.", "failure_reason": ""},
                            "shown_location": {"verdict": "good", "reasoning": "Shown location matches.", "failure_reason": ""},
                            "date": {"verdict": "good", "reasoning": "Date matches page.", "failure_reason": ""},
                            "overall": {"verdict": "good", "reasoning": "Everything is now aligned.", "failure_reason": ""},
                            "human_inference": "",
                            "needs_another_pass": [],
                            "needs_human_review": [],
                        },
                        "review_provenance": {
                            "caption": {
                                "prompt_variant": "parameter-suggestion",
                                "model": "glm-test",
                                "tuning_params": {
                                    "caption_max_tokens": 144,
                                    "caption_temperature": 0.1,
                                    "caption_max_edge": 2048,
                                },
                                "retry_count": 2,
                            },
                            "gps": {"prompt_variant": "base", "model": "glm-test", "tuning_params": {}, "retry_count": 0},
                            "shown_location": {"prompt_variant": "base", "model": "glm-test", "tuning_params": {}, "retry_count": 0},
                            "date": {"prompt_variant": "base", "model": "glm-test", "tuning_params": {}, "retry_count": 0},
                            "overall": {"prompt_variant": "base", "model": "glm-test", "tuning_params": {}, "retry_count": 0},
                        },
                    }
                ],
            }

            ai_verify_crops.persist_verify_crops_state(page_image, verify_result)

            crop_state = xmp_sidecar.read_pipeline_step(crop_image.with_suffix(".xmp"), "verify-crops")
            assert crop_state is not None
            concerns = dict(crop_state["concerns"])
            self.assertEqual(concerns["caption"]["provenance"]["prompt_variant"], "parameter-suggestion")
            self.assertEqual(concerns["caption"]["provenance"]["retry_count"], 2)
            self.assertEqual(concerns["caption"]["provenance"]["tuning_params"]["caption_max_edge"], 2048)


if __name__ == "__main__":
    unittest.main()
