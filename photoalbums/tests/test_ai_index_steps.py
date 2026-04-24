"""Tests for ai_index_steps.py: step graph, input hashes, and StepRunner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.ai_index_steps import (
    STEP_ORDER,
    STEPS,
    StepRunner,
    caption_input_hash,
    date_estimate_input_hash,
    ocr_input_hash,
    objects_input_hash,
    people_input_hash,
    propagate_to_crops_input_hash,
)


class TestStepDependencyOrder(unittest.TestCase):
    def test_step_order_covers_all_steps(self):
        self.assertEqual(set(STEP_ORDER), set(STEPS.keys()))

    def test_dependencies_appear_before_dependents_in_order(self):
        pos = {name: i for i, name in enumerate(STEP_ORDER)}
        for step_name, step in STEPS.items():
            for dep in step.depends_on:
                self.assertLess(
                    pos[dep],
                    pos[step_name],
                    f"{dep} must come before {step_name} in STEP_ORDER",
                )

    def test_ocr_has_no_dependencies(self):
        self.assertEqual(STEPS["ocr"].depends_on, [])

    def test_people_has_no_dependencies(self):
        self.assertEqual(STEPS["people"].depends_on, [])

    def test_caption_depends_on_ocr_and_people(self):
        self.assertIn("ocr", STEPS["caption"].depends_on)
        self.assertIn("people", STEPS["caption"].depends_on)

    def test_locations_depends_on_caption(self):
        self.assertIn("caption", STEPS["locations"].depends_on)

    def test_propagate_depends_on_locations_and_people(self):
        self.assertIn("locations", STEPS["propagate-to-crops"].depends_on)
        self.assertIn("people", STEPS["propagate-to-crops"].depends_on)


class TestInputHashIsolation(unittest.TestCase):
    """Verify that changing settings for one step does not affect another step's hash."""

    BASE_SETTINGS = {
        "ocr_engine": "local",
        "ocr_model": "qwen-ocr",
        "ocr_lang": "eng",
        "scan_group_signature": "",
        "cast_store_signature": "abc123",
        "caption_engine": "lmstudio",
        "caption_model": "qwen-vl-chat",
        "nominatim_base_url": "http://nominatim.local",
        "model": "yolo11n.pt",
        "enable_objects": True,
        "crop_paths_signature": "xyz",
    }

    def test_ocr_hash_ignores_caption_model(self):
        settings_a = {**self.BASE_SETTINGS, "caption_model": "qwen-vl-chat"}
        settings_b = {**self.BASE_SETTINGS, "caption_model": "different-caption-model"}
        self.assertEqual(
            ocr_input_hash(settings_a, {}),
            ocr_input_hash(settings_b, {}),
            "OCR hash must not change when caption_model changes",
        )

    def test_ocr_hash_ignores_cast_store_signature(self):
        settings_a = {**self.BASE_SETTINGS, "cast_store_signature": "sig-a"}
        settings_b = {**self.BASE_SETTINGS, "cast_store_signature": "sig-b"}
        self.assertEqual(ocr_input_hash(settings_a, {}), ocr_input_hash(settings_b, {}))

    def test_people_hash_ignores_ocr_model(self):
        settings_a = {**self.BASE_SETTINGS, "ocr_model": "ocr-a"}
        settings_b = {**self.BASE_SETTINGS, "ocr_model": "ocr-b"}
        self.assertEqual(people_input_hash(settings_a, {}), people_input_hash(settings_b, {}))

    def test_people_hash_changes_when_reviewed_identity_signature_changes(self):
        settings_a = {**self.BASE_SETTINGS, "cast_store_signature": "reviewed-a"}
        settings_b = {**self.BASE_SETTINGS, "cast_store_signature": "reviewed-b"}
        self.assertNotEqual(people_input_hash(settings_a, {}), people_input_hash(settings_b, {}))

    def test_caption_hash_includes_people_output_hash(self):
        h_without = caption_input_hash(self.BASE_SETTINGS, {})
        h_with = caption_input_hash(self.BASE_SETTINGS, {"people": "people-hash-abc"})
        self.assertNotEqual(h_without, h_with)

    def test_caption_hash_ignores_ocr_model(self):
        settings_a = {**self.BASE_SETTINGS, "ocr_model": "ocr-a"}
        settings_b = {**self.BASE_SETTINGS, "ocr_model": "ocr-b"}
        self.assertEqual(
            caption_input_hash(settings_a, {}),
            caption_input_hash(settings_b, {}),
        )

    def test_objects_hash_empty_when_disabled(self):
        settings = {**self.BASE_SETTINGS, "enable_objects": False}
        self.assertEqual(objects_input_hash(settings, {}), "")

    def test_date_estimate_hash_empty_when_non_lmstudio(self):
        settings = {**self.BASE_SETTINGS, "caption_engine": "none", "caption_model": ""}
        self.assertEqual(date_estimate_input_hash(settings, {}), "")

    def test_propagate_hash_includes_crop_paths_signature(self):
        settings_a = {**self.BASE_SETTINGS, "crop_paths_signature": "crops-a"}
        settings_b = {**self.BASE_SETTINGS, "crop_paths_signature": "crops-b"}
        self.assertNotEqual(
            propagate_to_crops_input_hash(settings_a, {}),
            propagate_to_crops_input_hash(settings_b, {}),
        )


class TestStepRunner(unittest.TestCase):
    SETTINGS = {
        "ocr_engine": "local",
        "ocr_model": "qwen-ocr",
        "ocr_lang": "eng",
        "scan_group_signature": "",
        "cast_store_signature": "abc",
        "caption_engine": "lmstudio",
        "caption_model": "qwen-vl",
        "nominatim_base_url": "",
        "model": "yolo11n.pt",
        "enable_objects": True,
        "crop_paths_signature": "",
    }

    def _make_runner(
        self,
        pipeline_state: dict | None = None,
        detections: dict | None = None,
        forced_steps: set | None = None,
    ) -> StepRunner:
        return StepRunner(
            settings=self.SETTINGS,
            existing_pipeline_state=pipeline_state or {},
            existing_detections=detections or {},
            forced_steps=forced_steps or set(),
        )

    def test_step_is_stale_when_no_pipeline_state(self):
        runner = self._make_runner()
        called = []

        def do_ocr():
            called.append("ocr")
            return {"ocr": {"text": "hello"}}

        runner.run("ocr", do_ocr)
        self.assertIn("ocr", called)
        self.assertTrue(runner.reran["ocr"])

    def test_step_skipped_when_hash_matches(self):
        # Compute the expected hash for OCR
        runner = self._make_runner()
        ocr_hash = runner._compute_input_hash("ocr")

        pipeline_state = {
            "ai-index/ocr": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": ocr_hash},
        }
        runner = self._make_runner(pipeline_state=pipeline_state, detections={"ocr": {"text": "cached"}})

        called = []

        def do_ocr():
            called.append("ocr")
            return {"ocr": {"text": "fresh"}}

        result = runner.run("ocr", do_ocr)
        self.assertEqual(called, [], "OCR should be skipped when hash matches")
        self.assertFalse(runner.reran["ocr"])
        self.assertEqual(result["ocr"], {"text": "cached"})

    def test_upstream_reran_forces_downstream_stale(self):
        # People step has matching hash, but we force it stale
        runner = self._make_runner(forced_steps={"people"})

        people_call_count = [0]
        caption_call_count = [0]

        def do_people():
            people_call_count[0] += 1
            return {"people": [{"name": "Alice"}]}

        def do_caption():
            caption_call_count[0] += 1
            return {"caption": {"description": "Photo of Alice"}}

        runner.run("ocr", lambda: {"ocr": {"text": "hello"}})
        runner.run("people", do_people)
        runner.run("caption", do_caption)

        self.assertEqual(people_call_count[0], 1, "people must run when forced")
        self.assertEqual(caption_call_count[0], 1, "caption must run when people reran")
        self.assertTrue(runner.reran["people"])
        self.assertTrue(runner.reran["caption"])

    def test_hash_match_skips_step_and_cascades_skip_to_downstream(self):
        """When all hashes match, all steps should be skipped."""
        runner = self._make_runner()
        ocr_hash = runner._compute_input_hash("ocr")
        people_hash = runner._compute_input_hash("people")
        # Caption hash needs ocr and people hashes in output_hashes, so just use empty for now
        # and set up state as if previously run
        pipeline_state = {
            "ai-index/ocr": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": ocr_hash},
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        runner2 = self._make_runner(
            pipeline_state=pipeline_state,
            detections={"ocr": {"text": "cached"}, "people": [{"name": "Bob"}]},
        )

        ocr_called = [False]
        people_called = [False]

        def do_ocr():
            ocr_called[0] = True
            return {"ocr": {"text": "fresh"}}

        def do_people():
            people_called[0] = True
            return {"people": [{"name": "Fresh"}]}

        runner2.run("ocr", do_ocr)
        runner2.run("people", do_people)

        self.assertFalse(ocr_called[0])
        self.assertFalse(people_called[0])
        self.assertFalse(runner2.reran["ocr"])
        self.assertFalse(runner2.reran["people"])

    def test_pending_records_populated_for_run_steps(self):
        runner = self._make_runner()
        runner.run("ocr", lambda: {"ocr": {"text": "hello"}})
        records = runner.get_pending_records()
        self.assertIn("ai-index/ocr", records)
        self.assertEqual(records["ai-index/ocr"]["result"], "ok")
        self.assertIn("timestamp", records["ai-index/ocr"])
        self.assertIn("input_hash", records["ai-index/ocr"])

    def test_pending_records_empty_for_skipped_steps(self):
        runner = self._make_runner()
        ocr_hash = runner._compute_input_hash("ocr")
        pipeline_state = {
            "ai-index/ocr": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": ocr_hash},
        }
        runner2 = self._make_runner(pipeline_state=pipeline_state, detections={"ocr": {}})
        runner2.run("ocr", lambda: {"ocr": {"text": "fresh"}})
        records = runner2.get_pending_records()
        self.assertNotIn("ai-index/ocr", records)

    def test_forced_step_runs_even_when_hash_matches(self):
        runner = self._make_runner()
        ocr_hash = runner._compute_input_hash("ocr")
        pipeline_state = {
            "ai-index/ocr": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": ocr_hash},
        }
        runner2 = self._make_runner(
            pipeline_state=pipeline_state,
            detections={"ocr": {"text": "cached"}},
            forced_steps={"ocr"},
        )
        called = [False]

        def do_ocr():
            called[0] = True
            return {"ocr": {"text": "fresh"}}

        runner2.run("ocr", do_ocr)
        self.assertTrue(called[0])
        self.assertTrue(runner2.reran["ocr"])


class TestForcedStepsPropagation(unittest.TestCase):
    """Task 7.3/7.4: Verify --steps CLI arg propagates forced staleness correctly."""

    SETTINGS = {
        "ocr_engine": "local",
        "ocr_model": "qwen-ocr",
        "ocr_lang": "eng",
        "scan_group_signature": "",
        "cast_store_signature": "abc",
        "caption_engine": "lmstudio",
        "caption_model": "qwen-vl",
        "nominatim_base_url": "",
        "model": "yolo11n.pt",
        "enable_objects": True,
        "crop_paths_signature": "",
    }

    def test_forced_caption_step_skips_ocr_and_people_but_runs_caption_and_downstream(self):
        """--steps caption: ocr and people skip (hash match), caption and downstream run."""
        # Build pipeline state with matching hashes for ocr and people
        base_runner = StepRunner(
            settings=self.SETTINGS,
            existing_pipeline_state={},
            existing_detections={},
            forced_steps=set(),
        )
        ocr_hash = base_runner._compute_input_hash("ocr")
        people_hash = base_runner._compute_input_hash("people")

        pipeline_state = {
            "ai-index/ocr": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": ocr_hash},
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        existing_detections = {
            "ocr": {"text": "cached-ocr"},
            "people": [{"name": "CachedPerson"}],
        }

        runner = StepRunner(
            settings=self.SETTINGS,
            existing_pipeline_state=pipeline_state,
            existing_detections=existing_detections,
            forced_steps={"caption"},  # --steps caption
        )

        ocr_called = [False]
        people_called = [False]
        caption_called = [False]
        locations_called = [False]

        runner.run("ocr", lambda: (ocr_called.__setitem__(0, True) or {"ocr": {"text": "fresh"}}))
        runner.run("people", lambda: (people_called.__setitem__(0, True) or {"people": []}))
        runner.run("caption", lambda: (caption_called.__setitem__(0, True) or {"caption": {"desc": "fresh"}}))
        runner.run("locations", lambda: (locations_called.__setitem__(0, True) or {"location": {}, "locations_shown": [], "location_shown_ran": True}))

        # OCR and people should be skipped (hash matches, not forced)
        self.assertFalse(ocr_called[0], "OCR must be skipped when hash matches and not forced")
        self.assertFalse(people_called[0], "people must be skipped when hash matches and not forced")
        # Caption must run (forced)
        self.assertTrue(caption_called[0], "caption must run when forced by --steps")
        # Locations must run (caption reran, and locations depends on caption)
        self.assertTrue(locations_called[0], "locations must run when upstream caption reran")
        self.assertTrue(runner.reran["caption"])
        self.assertTrue(runner.reran["locations"])
        self.assertFalse(runner.reran["ocr"])
        self.assertFalse(runner.reran["people"])


if __name__ == "__main__":
    unittest.main()
