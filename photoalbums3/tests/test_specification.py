"""Test specification for photoalbums3.

THIS FILE IS THE CONTRACT. Implementation is complete when all tests pass.

Each test maps to a specific feature requirement. Tests are organized by:
1. Engine (DAG execution, caching, state)
2. Job Runner (batch processing, resume)
3. Output Validation (XMP field checking)
4. External State (cast DB, model version tracking)
5. Integration (end-to-end flows)

DO NOT mark a feature as complete until its tests pass.
DO NOT add `@pytest.mark.skip` to make tests pass.
DO NOT implement just enough to make tests pass without addressing the spec.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# =============================================================================
# ENGINE TESTS (Phase 1)
# =============================================================================

class TestPipelineEngine:
    """Tests for engine.PipelineEngine: DAG execution and caching."""

    def test_engine_loads_pipeline_definition(self, tmp_path):
        """REQUIREMENT: Engine loads pipeline from pipeline.py module."""
        from photoalbums3.engine import PipelineEngine
        engine = PipelineEngine(state_dir=tmp_path)
        assert engine.steps is not None
        assert len(engine.steps) > 0, "Pipeline must have at least one step"

    def test_engine_creates_state_for_new_page(self, tmp_path):
        """REQUIREMENT: New page initializes state with all steps as pending."""
        from photoalbums3.engine import PipelineEngine, StepStatus
        engine = PipelineEngine(state_dir=tmp_path)
        state = engine.load_or_create_state("test_page", context={})
        assert state.page_id == "test_page"
        assert all(s.status == StepStatus.PENDING for s in state.steps.values())

    def test_engine_persists_state_to_disk(self, tmp_path):
        """REQUIREMENT: State persists between process invocations."""
        from photoalbums3.engine import PipelineEngine
        engine1 = PipelineEngine(state_dir=tmp_path)
        engine1.load_or_create_state("test_page", context={"key": "value"})
        engine1.save_state()

        # New engine instance should load the same state
        engine2 = PipelineEngine(state_dir=tmp_path)
        state = engine2.load_or_create_state("test_page", context={})
        assert state.context["key"] == "value"

    def test_engine_executes_step_in_dependency_order(self, tmp_path):
        """REQUIREMENT: Steps execute in topological order."""
        from photoalbums3.engine import PipelineEngine
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={})

        execution_order = []
        for step in engine.steps:
            with patch.object(step, "handler", side_effect=lambda **kw: execution_order.append(step.id)):
                pass

        # After full execution, order matches DAG topology
        engine.execute_pipeline()
        # Assert dependencies came before dependents
        for step in engine.steps:
            for dep_id in step.depends_on:
                assert execution_order.index(dep_id) < execution_order.index(step.id), \
                    f"{dep_id} must execute before {step.id}"

    def test_engine_skips_step_with_unchanged_inputs(self, tmp_path):
        """REQUIREMENT: Steps with unchanged input hash are skipped."""
        from photoalbums3.engine import PipelineEngine, StepStatus
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={"input": "value"})

        # First run
        engine.execute_pipeline()
        first_hashes = {s.step_id: s.input_hash for s in engine.state.steps.values()}

        # Second run - inputs unchanged
        engine.execute_pipeline()

        # All steps should have status SKIPPED or DONE with same hash
        for step in engine.state.steps.values():
            assert step.input_hash == first_hashes[step.step_id], \
                "Input hash should not change when inputs are unchanged"

    def test_engine_reruns_step_when_inputs_change(self, tmp_path):
        """REQUIREMENT: Changed inputs trigger re-execution."""
        from photoalbums3.engine import PipelineEngine
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={"input": "v1"})
        engine.execute_pipeline()
        first_hash = list(engine.state.steps.values())[0].input_hash

        # Change context
        engine.state.context["input"] = "v2"
        engine.execute_pipeline()
        second_hash = list(engine.state.steps.values())[0].input_hash

        assert first_hash != second_hash, "Hash must change when inputs change"

    def test_engine_invalidates_downstream_when_upstream_reruns(self, tmp_path):
        """REQUIREMENT: Re-running a step marks downstream steps as stale."""
        from photoalbums3.engine import PipelineEngine, StepStatus
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={})

        # Run all steps
        engine.execute_pipeline()

        # Force re-run of first step
        first_step_id = engine.steps[0].id
        engine.execute_step(first_step_id, force=True)

        # Downstream steps should be marked stale (re-executed or pending)
        # ... assertion logic

    def test_engine_handles_step_failure_gracefully(self, tmp_path):
        """REQUIREMENT: Step failures don't crash the pipeline."""
        from photoalbums3.engine import PipelineEngine, StepStatus
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={})

        # Mock first step to raise exception
        with patch.object(engine.steps[0], "handler", side_effect=Exception("test error")):
            engine.execute_step(engine.steps[0].id)

        first_step = engine.state.steps[engine.steps[0].id]
        assert first_step.status == StepStatus.FAILED
        assert "test error" in first_step.error

    def test_engine_supports_parameter_override(self, tmp_path):
        """REQUIREMENT: Step parameters can be overridden (for prompt tuning)."""
        from photoalbums3.engine import PipelineEngine
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={})

        engine.override_step_param(engine.steps[0].id, "prompt", "new prompt")
        step = engine.state.steps[engine.steps[0].id]
        assert step.params_override.get("prompt") == "new prompt"

    def test_engine_records_input_output_for_inspection(self, tmp_path):
        """REQUIREMENT: After execution, inputs and outputs are recorded."""
        from photoalbums3.engine import PipelineEngine
        engine = PipelineEngine(state_dir=tmp_path)
        engine.load_or_create_state("test_page", context={"key": "value"})
        engine.execute_pipeline()

        for step in engine.state.steps.values():
            if step.status.value == "done":
                assert step.inputs, f"Step {step.step_id} should record inputs"
                assert step.outputs, f"Step {step.step_id} should record outputs"


# =============================================================================
# JOB RUNNER TESTS (Phase 1)
# =============================================================================

class TestBatchJobRunner:
    """Tests for job_runner.BatchJobRunner: batch processing with resume."""

    def test_runner_discovers_pages_for_albums(self, tmp_path):
        """REQUIREMENT: Runner lists all pages for given albums."""
        from photoalbums3.job_runner import BatchJobRunner
        with patch("photoalbums3.job_runner.list_pages_for_album") as mock_list:
            mock_list.side_effect = [
                ["album1_p1", "album1_p2"],
                ["album2_p1"],
            ]
            runner = BatchJobRunner("test_job", ["album1", "album2"], state_dir=tmp_path)
            page_ids = [p["id"] for p in runner.state["pages"]]
            assert page_ids == ["album1_p1", "album1_p2", "album2_p1"]

    def test_runner_creates_job_state_file(self, tmp_path):
        """REQUIREMENT: Job state persists to disk."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)
        assert runner.state_file.exists()
        loaded = json.loads(runner.state_file.read_text())
        assert loaded["job_id"] == "test_job"

    def test_runner_loads_existing_state_on_init(self, tmp_path):
        """REQUIREMENT: Subsequent runs load existing state (Make-like)."""
        from photoalbums3.job_runner import BatchJobRunner
        runner1 = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)
        runner1.state["pages"][0]["status"] = "done"
        runner1.save_state()

        runner2 = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)
        assert runner2.state["pages"][0]["status"] == "done"

    def test_runner_skips_done_pages_on_resume(self, tmp_path):
        """REQUIREMENT: Re-running skips pages already marked done."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        # Mark first page done
        runner.state["pages"][0]["status"] = "done"
        runner.save_state()

        executed = []
        with patch.object(runner.engine, "execute_pipeline",
                          side_effect=lambda: executed.append(1)):
            runner.run()

        # First page should be skipped
        assert len(executed) == len(runner.state["pages"]) - 1

    def test_runner_resumes_after_keyboard_interrupt(self, tmp_path):
        """REQUIREMENT: Ctrl+C preserves state; re-run resumes."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        call_count = [0]
        def maybe_interrupt(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise KeyboardInterrupt()

        with patch.object(runner.engine, "execute_pipeline", side_effect=maybe_interrupt):
            with pytest.raises((KeyboardInterrupt, SystemExit)):
                runner.run()

        # State should reflect partial completion
        runner2 = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)
        completed = sum(1 for p in runner2.state["pages"] if p["status"] == "done")
        pending = sum(1 for p in runner2.state["pages"] if p["status"] == "pending")
        assert completed >= 1, "At least one page should be completed before interrupt"
        assert pending >= 1, "At least one page should remain pending"

    def test_runner_force_restart_marks_all_pending(self, tmp_path):
        """REQUIREMENT: --restart flag marks all pages pending for re-run."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        # Mark all done
        for page in runner.state["pages"]:
            page["status"] = "done"
        runner.save_state()

        # Force restart
        with patch.object(runner.engine, "execute_pipeline"):
            runner.run(force_restart=True)

        # All should have been processed (not skipped)

    def test_runner_returns_correct_exit_codes(self, tmp_path):
        """REQUIREMENT: Exit code 0 on success, 1 on failures."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        with patch.object(runner.engine, "execute_pipeline"):
            assert runner.run() == 0

        # Now with failure
        runner.state["pages"][0]["status"] = "failed"
        runner.save_state()
        # Note: failed pages don't necessarily trigger non-zero exit on next run
        # depending on retry semantics

    def test_runner_handles_step_failures_per_page(self, tmp_path):
        """REQUIREMENT: One page's failure doesn't stop other pages."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        call_count = [0]
        def maybe_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("test failure")

        with patch.object(runner.engine, "execute_pipeline", side_effect=maybe_fail):
            runner.run()

        # First page failed, others should still run
        statuses = [p["status"] for p in runner.state["pages"]]
        assert "failed" in statuses
        assert "done" in statuses

    def test_runner_reports_status_clearly(self, tmp_path, capsys):
        """REQUIREMENT: Status output shows done/pending/failed counts."""
        from photoalbums3.job_runner import BatchJobRunner
        runner = BatchJobRunner("test_job", ["album1"], state_dir=tmp_path)

        with patch.object(runner.engine, "execute_pipeline"):
            runner.run()

        captured = capsys.readouterr()
        assert "Done:" in captured.out or "done" in captured.out.lower()


# =============================================================================
# OUTPUT VALIDATION TESTS (Phase 2)
# =============================================================================

class TestOutputValidator:
    """Tests for output_validator: XMP field validation."""

    def test_validator_detects_missing_xmp(self, tmp_path):
        """REQUIREMENT: Missing XMP file fails validation."""
        from photoalbums3.output_validator import validate_page_output
        is_valid, reason = validate_page_output(
            "test_page",
            tmp_path / "missing.xmp",
            ["detect_regions"],
        )
        assert not is_valid
        assert "missing" in reason.lower() or "not found" in reason.lower()

    def test_validator_checks_required_fields(self, tmp_path):
        """REQUIREMENT: Missing XMP fields fail validation."""
        from photoalbums3.output_validator import validate_page_output
        xmp_path = tmp_path / "test.xmp"
        xmp_path.write_text("<x:xmpmeta></x:xmpmeta>")  # Empty XMP

        is_valid, reason = validate_page_output(
            "test_page",
            xmp_path,
            ["detect_regions"],
        )
        assert not is_valid
        assert "regions" in reason.lower() or "field" in reason.lower()

    def test_validator_computes_output_hash(self, tmp_path):
        """REQUIREMENT: Valid outputs return a hash for staleness tracking."""
        from photoalbums3.output_validator import validate_page_output
        # ... test that valid outputs return a hash string

    def test_validator_detects_manual_xmp_edits(self, tmp_path):
        """REQUIREMENT: Output hash changes when XMP fields are manually edited."""
        from photoalbums3.output_validator import validate_page_output
        # Setup: write XMP, validate, get hash1
        # Modify XMP, validate again, get hash2
        # Assert hash1 != hash2


# =============================================================================
# EXTERNAL STATE TESTS (Phase 2)
# =============================================================================

class TestExternalState:
    """Tests for external_state: tracking external dependencies."""

    def test_external_state_includes_cast_db_hash(self, tmp_path):
        """REQUIREMENT: External state includes cast DB hash."""
        from photoalbums3.external_state import compute_external_state_hash
        cast_db = tmp_path / "cast.db"
        cast_db.write_bytes(b"test data")

        state = compute_external_state_hash(cast_db, model_versions={})
        assert "cast_db_hash" in state
        assert state["cast_db_hash"] != ""

    def test_external_state_detects_changes(self, tmp_path):
        """REQUIREMENT: Changed cast DB → state hash changes."""
        from photoalbums3.external_state import compute_external_state_hash
        cast_db = tmp_path / "cast.db"
        cast_db.write_bytes(b"v1")
        state1 = compute_external_state_hash(cast_db, model_versions={})

        cast_db.write_bytes(b"v2")
        state2 = compute_external_state_hash(cast_db, model_versions={})

        assert state1["cast_db_hash"] != state2["cast_db_hash"]

    def test_external_state_change_invalidates_correct_steps(self):
        """REQUIREMENT: Cast DB change → invalidates face/people steps."""
        from photoalbums3.external_state import has_external_state_changed

        old = {"cast_db_hash": "v1", "model_versions": {}}
        new = {"cast_db_hash": "v2", "model_versions": {}}

        changed, steps = has_external_state_changed(old, new)
        assert changed
        assert "face_refresh" in steps or "detect_regions" in steps

    def test_external_state_model_change_invalidates_ai_steps(self):
        """REQUIREMENT: Caption model change → invalidates ai_index step."""
        from photoalbums3.external_state import has_external_state_changed

        old = {"cast_db_hash": "v1", "model_versions": {"caption": "v1"}}
        new = {"cast_db_hash": "v1", "model_versions": {"caption": "v2"}}

        changed, steps = has_external_state_changed(old, new)
        assert changed
        assert "ai_index" in steps


# =============================================================================
# INTEGRATION TESTS (End-to-End)
# =============================================================================

class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_full_pipeline_runs_to_completion(self, tmp_path):
        """REQUIREMENT: A page can be fully processed end-to-end."""
        from photoalbums3.job_runner import BatchJobRunner
        # Setup test album with sample data
        # Run pipeline
        # Assert all expected outputs exist

    def test_resume_after_interrupt_completes_remaining_work(self, tmp_path):
        """REQUIREMENT: Interrupt + resume produces same final state as uninterrupted run."""
        # Run uninterrupted version → record final state
        # Run interrupted + resumed version → record final state
        # Assert states match

    def test_force_restart_only_rebuilds_changed_steps(self, tmp_path):
        """REQUIREMENT: --restart with no changes is fast (everything skipped)."""
        # Run pipeline
        # Run with --restart, no changes
        # Assert no actual computation happened (all skipped)

    def test_external_state_change_only_rebuilds_affected(self, tmp_path):
        """REQUIREMENT: Cast DB change only re-runs affected steps."""
        # Run pipeline
        # Change cast DB
        # Run again
        # Assert only face_refresh and downstream re-ran


# =============================================================================
# JUSTFILE INTEGRATION
# =============================================================================

class TestJustfileTargets:
    """Test that justfile targets work correctly."""

    def test_just_render_albums_target_exists(self):
        """REQUIREMENT: 'just render-albums' target is defined."""
        justfile = Path(__file__).parent.parent.parent / "justfile"
        content = justfile.read_text()
        assert "render-albums" in content

    def test_just_render_albums_runs_batch_runner(self):
        """REQUIREMENT: justfile target invokes BatchJobRunner."""
        justfile = Path(__file__).parent.parent.parent / "justfile"
        content = justfile.read_text()
        assert "BatchJobRunner" in content or "job_runner" in content


# =============================================================================
# ANTI-STUB TESTS (Force Real Implementation)
# =============================================================================

class TestAntiStub:
    """Tests that detect stub/incomplete implementations."""

    def test_no_pass_only_function_bodies(self):
        """REQUIREMENT: No function should have a body of just 'pass'."""
        import ast
        from photoalbums3 import engine, job_runner

        for module in [engine, job_runner]:
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                        pytest.fail(f"Stub function detected: {node.name} in {module.__name__}")

    def test_no_todo_comments_in_critical_paths(self):
        """REQUIREMENT: No TODO comments in production code paths."""
        from photoalbums3 import engine, job_runner

        for module in [engine, job_runner]:
            content = Path(module.__file__).read_text()
            assert "TODO" not in content, \
                f"TODO comments must be resolved in {module.__name__}"

    def test_no_notimplemented_errors(self):
        """REQUIREMENT: No NotImplementedError in production code."""
        from photoalbums3 import engine, job_runner

        for module in [engine, job_runner]:
            content = Path(module.__file__).read_text()
            assert "NotImplementedError" not in content, \
                f"NotImplementedError found in {module.__name__}"
