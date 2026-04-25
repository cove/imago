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
    metadata_input_hash,
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

    def test_metadata_has_no_dependencies(self):
        self.assertEqual(STEPS["metadata"].depends_on, [])

    def test_people_has_no_dependencies(self):
        self.assertEqual(STEPS["people"].depends_on, [])

    def test_propagate_depends_on_metadata_and_people(self):
        self.assertIn("metadata", STEPS["propagate-to-crops"].depends_on)
        self.assertIn("people", STEPS["propagate-to-crops"].depends_on)

    def test_metadata_output_keys(self):
        keys = STEPS["metadata"].output_keys
        self.assertIn("ocr", keys)
        self.assertIn("caption", keys)
        self.assertIn("location", keys)
        self.assertIn("locations_shown", keys)


class TestInputHashIsolation(unittest.TestCase):
    """Verify that changing settings for one step does not affect another step's hash."""

    BASE_SETTINGS = {
        "cast_store_signature": "abc123",
        "caption_engine": "lmstudio",
        "caption_model": "qwen-vl-chat",
        "nominatim_base_url": "http://nominatim.local",
        "model": "yolo11n.pt",
        "enable_objects": True,
        "crop_paths_signature": "xyz",
    }

    def test_metadata_hash_empty_when_non_lmstudio(self):
        settings = {**self.BASE_SETTINGS, "caption_engine": "none", "caption_model": ""}
        self.assertEqual(metadata_input_hash(settings, {}), "")

    def test_metadata_hash_changes_with_caption_model(self):
        settings_a = {**self.BASE_SETTINGS, "caption_model": "model-a"}
        settings_b = {**self.BASE_SETTINGS, "caption_model": "model-b"}
        self.assertNotEqual(metadata_input_hash(settings_a, {}), metadata_input_hash(settings_b, {}))

    def test_metadata_hash_changes_with_nominatim_url(self):
        settings_a = {**self.BASE_SETTINGS, "nominatim_base_url": "http://host-a"}
        settings_b = {**self.BASE_SETTINGS, "nominatim_base_url": "http://host-b"}
        self.assertNotEqual(metadata_input_hash(settings_a, {}), metadata_input_hash(settings_b, {}))

    def test_metadata_hash_ignores_cast_store_signature(self):
        settings_a = {**self.BASE_SETTINGS, "cast_store_signature": "sig-a"}
        settings_b = {**self.BASE_SETTINGS, "cast_store_signature": "sig-b"}
        self.assertEqual(metadata_input_hash(settings_a, {}), metadata_input_hash(settings_b, {}))

    def test_people_hash_ignores_caption_model(self):
        settings_a = {**self.BASE_SETTINGS, "caption_model": "model-a"}
        settings_b = {**self.BASE_SETTINGS, "caption_model": "model-b"}
        self.assertEqual(people_input_hash(settings_a, {}), people_input_hash(settings_b, {}))

    def test_people_hash_changes_when_cast_store_signature_changes(self):
        settings_a = {**self.BASE_SETTINGS, "cast_store_signature": "reviewed-a"}
        settings_b = {**self.BASE_SETTINGS, "cast_store_signature": "reviewed-b"}
        self.assertNotEqual(people_input_hash(settings_a, {}), people_input_hash(settings_b, {}))

    def test_objects_hash_empty_when_disabled(self):
        settings = {**self.BASE_SETTINGS, "enable_objects": False}
        self.assertEqual(objects_input_hash(settings, {}), "")

    def test_propagate_hash_includes_crop_paths_signature(self):
        settings_a = {**self.BASE_SETTINGS, "crop_paths_signature": "crops-a"}
        settings_b = {**self.BASE_SETTINGS, "crop_paths_signature": "crops-b"}
        self.assertNotEqual(
            propagate_to_crops_input_hash(settings_a, {}),
            propagate_to_crops_input_hash(settings_b, {}),
        )

    def test_propagate_hash_includes_metadata_output_hash(self):
        h_without = propagate_to_crops_input_hash(self.BASE_SETTINGS, {})
        h_with = propagate_to_crops_input_hash(self.BASE_SETTINGS, {"metadata": "metadata-hash-abc"})
        self.assertNotEqual(h_without, h_with)


class TestStepRunner(unittest.TestCase):
    SETTINGS = {
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

        def do_people():
            called.append("people")
            return {"people": []}

        runner.run("people", do_people)
        self.assertIn("people", called)
        self.assertTrue(runner.reran["people"])

    def test_step_skipped_when_hash_matches(self):
        runner = self._make_runner()
        people_hash = runner._compute_input_hash("people")

        pipeline_state = {
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        runner = self._make_runner(pipeline_state=pipeline_state, detections={"people": [{"name": "cached"}]})

        called = []

        def do_people():
            called.append("people")
            return {"people": [{"name": "fresh"}]}

        result = runner.run("people", do_people)
        self.assertEqual(called, [], "people should be skipped when hash matches")
        self.assertFalse(runner.reran["people"])
        self.assertEqual(result["people"], [{"name": "cached"}])

    def test_upstream_reran_forces_downstream_stale(self):
        runner = self._make_runner(forced_steps={"people"})

        people_call_count = [0]
        propagate_call_count = [0]

        def do_people():
            people_call_count[0] += 1
            return {"people": [{"name": "Alice"}]}

        def do_propagate():
            propagate_call_count[0] += 1
            return {"crops_updated": 1}

        runner.run("metadata", lambda: {"ocr": {}, "caption": {}, "location": {}, "locations_shown": [], "location_shown_ran": False})
        runner.run("people", do_people)
        runner.run("propagate-to-crops", do_propagate)

        self.assertEqual(people_call_count[0], 1, "people must run when forced")
        self.assertEqual(propagate_call_count[0], 1, "propagate must run when people reran")
        self.assertTrue(runner.reran["people"])
        self.assertTrue(runner.reran["propagate-to-crops"])

    def test_hash_match_skips_step_and_cascades_skip_to_downstream(self):
        """When all hashes match, all steps should be skipped."""
        runner = self._make_runner()
        people_hash = runner._compute_input_hash("people")
        pipeline_state = {
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        runner2 = self._make_runner(
            pipeline_state=pipeline_state,
            detections={"people": [{"name": "Bob"}]},
        )

        people_called = [False]

        def do_people():
            people_called[0] = True
            return {"people": [{"name": "Fresh"}]}

        runner2.run("people", do_people)

        self.assertFalse(people_called[0])
        self.assertFalse(runner2.reran["people"])

    def test_pending_records_populated_for_run_steps(self):
        runner = self._make_runner()
        runner.run("people", lambda: {"people": []})
        records = runner.get_pending_records()
        self.assertIn("ai-index/people", records)
        self.assertEqual(records["ai-index/people"]["result"], "ok")
        self.assertIn("timestamp", records["ai-index/people"])
        self.assertIn("input_hash", records["ai-index/people"])

    def test_pending_records_empty_for_skipped_steps(self):
        runner = self._make_runner()
        people_hash = runner._compute_input_hash("people")
        pipeline_state = {
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        runner2 = self._make_runner(pipeline_state=pipeline_state, detections={"people": []})
        runner2.run("people", lambda: {"people": [{"name": "fresh"}]})
        records = runner2.get_pending_records()
        self.assertNotIn("ai-index/people", records)

    def test_forced_step_runs_even_when_hash_matches(self):
        runner = self._make_runner()
        people_hash = runner._compute_input_hash("people")
        pipeline_state = {
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        runner2 = self._make_runner(
            pipeline_state=pipeline_state,
            detections={"people": [{"name": "cached"}]},
            forced_steps={"people"},
        )
        called = [False]

        def do_people():
            called[0] = True
            return {"people": [{"name": "fresh"}]}

        runner2.run("people", do_people)
        self.assertTrue(called[0])
        self.assertTrue(runner2.reran["people"])


class TestForcedStepsPropagation(unittest.TestCase):
    """Verify --steps CLI arg propagates forced staleness correctly."""

    SETTINGS = {
        "cast_store_signature": "abc",
        "caption_engine": "lmstudio",
        "caption_model": "qwen-vl",
        "nominatim_base_url": "",
        "model": "yolo11n.pt",
        "enable_objects": True,
        "crop_paths_signature": "",
    }

    def test_forced_metadata_step_causes_propagate_to_run(self):
        """--steps metadata: people skips (hash match), metadata and propagate run."""
        base_runner = StepRunner(
            settings=self.SETTINGS,
            existing_pipeline_state={},
            existing_detections={},
            forced_steps=set(),
        )
        people_hash = base_runner._compute_input_hash("people")

        pipeline_state = {
            "ai-index/people": {"timestamp": "2026-04-11T00:00:00Z", "result": "ok", "input_hash": people_hash},
        }
        existing_detections = {
            "people": [{"name": "CachedPerson"}],
        }

        runner = StepRunner(
            settings=self.SETTINGS,
            existing_pipeline_state=pipeline_state,
            existing_detections=existing_detections,
            forced_steps={"metadata"},
        )

        metadata_called = [False]
        people_called = [False]
        propagate_called = [False]

        runner.run("metadata", lambda: (metadata_called.__setitem__(0, True) or {"ocr": {}, "caption": {}, "location": {}, "locations_shown": [], "location_shown_ran": True}))
        runner.run("people", lambda: (people_called.__setitem__(0, True) or {"people": []}))
        runner.run("propagate-to-crops", lambda: (propagate_called.__setitem__(0, True) or {"crops_updated": 0}))

        self.assertTrue(metadata_called[0], "metadata must run when forced")
        self.assertFalse(people_called[0], "people must be skipped when hash matches and not forced")
        self.assertTrue(propagate_called[0], "propagate must run when metadata reran")
        self.assertTrue(runner.reran["metadata"])
        self.assertFalse(runner.reran["people"])
        self.assertTrue(runner.reran["propagate-to-crops"])


if __name__ == "__main__":
    unittest.main()
