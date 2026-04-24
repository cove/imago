"""Pipeline execution engine with caching and staleness tracking.

Handles DAG execution, input/output caching, and smart re-runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Callable
from enum import Enum

from pipeline import PIPELINE_STEPS, STEPS_BY_ID


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepExecution:
    """Record of a single step execution."""
    step_id: str
    status: StepStatus = StepStatus.PENDING
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    error: Optional[str] = None
    timestamp: float = 0.0
    duration: float = 0.0
    params_override: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "input_hash": self.input_hash,
            "error": self.error,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "params_override": self.params_override,
        }


@dataclass
class PipelineState:
    """State of the entire pipeline execution."""
    page_id: str
    steps: dict[str, StepExecution] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # archive_path, view_dir, etc.

    def to_dict(self):
        return {
            "page_id": self.page_id,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "context": self.context,
        }


class PipelineEngine:
    """DAG execution engine with caching and smart re-runs."""

    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or Path("/tmp/photoalbums3_state")
        self.state_dir.mkdir(exist_ok=True, parents=True)
        self.state: Optional[PipelineState] = None

    def load_or_create_state(self, page_id: str, context: dict) -> PipelineState:
        """Load existing state or create new."""
        state_file = self.state_dir / f"{page_id}.json"

        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                state = self._dict_to_state(data)
                # Update context (in case paths changed)
                state.context.update(context)
                self.state = state
                return state
            except Exception as e:
                print(f"Failed to load state: {e}, creating new")

        # Create new state
        state = PipelineState(page_id=page_id, context=context)
        for step in PIPELINE_STEPS:
            state.steps[step.id] = StepExecution(step_id=step.id)

        self.state = state
        return state

    def save_state(self):
        """Persist state to disk."""
        if not self.state:
            return

        state_file = self.state_dir / f"{self.state.page_id}.json"
        state_file.write_text(json.dumps(self.state.to_dict(), indent=2))

    def _compute_input_hash(self, inputs: dict) -> str:
        """Hash the inputs to detect changes."""
        content = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_step_inputs(self, step_id: str) -> dict:
        """Gather inputs from upstream steps."""
        step_def = STEPS_BY_ID[step_id]
        inputs = {}

        # Get base context
        inputs.update(self.state.context)

        # Get outputs from dependency steps
        for dep_id in step_def.depends_on:
            if dep_id in self.state.steps:
                dep_exec = self.state.steps[dep_id]
                inputs[dep_id] = dep_exec.outputs

        # Apply parameter overrides
        step_exec = self.state.steps[step_id]
        if step_exec.params_override:
            inputs.update(step_exec.params_override)

        return inputs

    def is_step_stale(self, step_id: str) -> tuple[bool, str]:
        """Check if a step needs to re-run.

        Returns: (is_stale, reason)
        """
        step_exec = self.state.steps[step_id]

        # Never run: RUNNING or DONE without changes
        if step_exec.status == StepStatus.RUNNING:
            return False, "already running"

        # Check dependencies: if any upstream reran, we're stale
        step_def = STEPS_BY_ID[step_id]
        for dep_id in step_def.depends_on:
            dep_exec = self.state.steps[dep_id]
            # If dep just ran (timestamp = last run time), we're stale
            # For now, if dep has outputs, assume we might be stale
            if dep_exec.status in (StepStatus.RUNNING, StepStatus.DONE):
                # TODO: compare timestamps to see if dep *just* reran
                pass

        # Never run: DONE and inputs unchanged
        if step_exec.status == StepStatus.DONE:
            current_inputs = self._get_step_inputs(step_id)
            current_hash = self._compute_input_hash(current_inputs)

            if current_hash == step_exec.input_hash:
                return False, f"inputs unchanged (hash {current_hash[:8]})"
            else:
                return True, f"inputs changed: {step_exec.input_hash[:8]} → {current_hash[:8]}"

        # Not run yet
        if step_exec.status == StepStatus.PENDING:
            return True, "not yet run"

        # Failed: can retry
        if step_exec.status == StepStatus.FAILED:
            return True, f"previously failed: {step_exec.error}"

        return True, "unknown"

    def execute_step(self, step_id: str, force: bool = False) -> StepExecution:
        """Execute a single step.

        Returns the execution result (may be cached if unchanged).
        """
        step_def = STEPS_BY_ID[step_id]
        step_exec = self.state.steps[step_id]

        # Check staleness
        is_stale, reason = self.is_step_stale(step_id)

        if not force and not is_stale:
            step_exec.status = StepStatus.SKIPPED
            print(f"[{step_id}] SKIP: {reason}")
            return step_exec

        # Mark all downstream as potentially stale
        self._mark_downstream_stale(step_id)

        # Execute
        print(f"[{step_id}] RUN: {reason if not force else 'forced'}")
        step_exec.status = StepStatus.RUNNING

        start_time = time.time()
        try:
            # Get inputs
            inputs = self._get_step_inputs(step_id)
            step_exec.inputs = inputs

            # Compute input hash for next time
            step_exec.input_hash = self._compute_input_hash(inputs)

            # Execute handler
            outputs = step_def.handler(**inputs)

            # Record outputs
            step_exec.outputs = outputs
            step_exec.status = StepStatus.DONE
            step_exec.error = None

            print(f"[{step_id}] DONE ({time.time() - start_time:.2f}s)")

        except Exception as e:
            step_exec.status = StepStatus.FAILED
            step_exec.error = str(e)
            print(f"[{step_id}] FAILED: {e}")

        step_exec.timestamp = time.time()
        step_exec.duration = time.time() - start_time

        # Save state after each step
        self.save_state()

        return step_exec

    def _mark_downstream_stale(self, step_id: str):
        """Mark all downstream steps as stale."""
        for step in PIPELINE_STEPS:
            if step_id in step.depends_on:
                exec = self.state.steps[step.id]
                if exec.status == StepStatus.DONE:
                    exec.status = StepStatus.PENDING  # Force re-run

    def execute_pipeline(self, final_step_id: str = None, force: bool = False) -> PipelineState:
        """Execute pipeline up to final_step_id, executing only stale steps.

        Args:
            final_step_id: Run up to this step (default: last step)
            force: Force all steps to re-run
        """
        if final_step_id is None:
            final_step_id = PIPELINE_STEPS[-1].id

        # Build execution order (topological sort)
        executed = set()
        for step in PIPELINE_STEPS:
            # Execute dependencies first
            for dep_id in step.depends_on:
                if dep_id not in executed:
                    self.execute_step(dep_id, force=force)
                    executed.add(dep_id)

            # Execute this step
            self.execute_step(step.id, force=force)
            executed.add(step.id)

            # Stop if we've reached the target
            if step.id == final_step_id:
                break

        return self.state

    def override_step_param(self, step_id: str, param: str, value: Any):
        """Override a parameter for a step (e.g., change prompt or model)."""
        step_exec = self.state.steps[step_id]
        step_exec.params_override[param] = value

        # Mark step as stale so it re-runs
        if step_exec.status == StepStatus.DONE:
            step_exec.status = StepStatus.PENDING

        self.save_state()
        print(f"[{step_id}] Override {param} = {value}")

    @staticmethod
    def _dict_to_state(data: dict) -> PipelineState:
        """Convert JSON dict back to PipelineState."""
        state = PipelineState(page_id=data["page_id"], context=data.get("context", {}))

        for step_id, step_data in data.get("steps", {}).items():
            state.steps[step_id] = StepExecution(
                step_id=step_id,
                status=StepStatus(step_data["status"]),
                inputs=step_data.get("inputs", {}),
                outputs=step_data.get("outputs", {}),
                input_hash=step_data.get("input_hash", ""),
                error=step_data.get("error"),
                timestamp=step_data.get("timestamp", 0),
                duration=step_data.get("duration", 0),
                params_override=step_data.get("params_override", {}),
            )

        return state
