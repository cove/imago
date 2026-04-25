"""Step graph definition and StepRunner for the ai-index pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .ai_prompt_assets import asset_hashes
from .xmp_sidecar import xmp_datetime_now


def _sha16(*parts: str) -> str:
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _sha16_data(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    return _sha16(serialized)


def _prompt_hash_payload(*paths: str) -> str:
    return _sha16_data(asset_hashes(*paths))


@dataclass(frozen=True)
class StepDef:
    name: str
    depends_on: list[str]
    output_keys: list[str]


# ── Step declarations ──────────────────────────────────────────────────────────

STEPS: dict[str, StepDef] = {
    "ocr": StepDef("ocr", [], ["ocr"]),
    "people": StepDef("people", [], ["people"]),
    "caption": StepDef("caption", ["ocr", "people"], ["caption"]),
    "locations": StepDef("locations", ["caption"], ["location", "locations_shown", "location_shown_ran"]),
    "objects": StepDef("objects", [], ["objects", "object_model"]),
    "date-estimate": StepDef("date-estimate", ["ocr", "caption"], []),
    "propagate-to-crops": StepDef("propagate-to-crops", ["locations", "people"], []),
}

# Topological order for execution
STEP_ORDER = ["ocr", "people", "objects", "caption", "locations", "date-estimate", "propagate-to-crops"]


# ── Input hash functions ───────────────────────────────────────────────────────

def ocr_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    return _sha16(
        settings.get("ocr_engine", ""),
        settings.get("ocr_model", ""),
        settings.get("ocr_lang", ""),
        settings.get("scan_group_signature", ""),
        _prompt_hash_payload(
            "ai-index/ocr/system.md",
            "ai-index/ocr/user.md",
            "ai-index/ocr/params.toml",
        ),
    )


def people_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    return _sha16(
        settings.get("cast_store_signature", ""),
        _prompt_hash_payload(
            "ai-index/people-count/system.md",
            "ai-index/people-count/user.md",
            "ai-index/people-count/output.md",
            "ai-index/people-count/params.toml",
        ),
    )


def caption_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    return _sha16(
        settings.get("caption_engine", ""),
        settings.get("caption_model", ""),
        output_hashes.get("people", ""),
        _prompt_hash_payload(
            "ai-index/caption/user.md",
            "ai-index/caption/upstream-ocr-context.md",
            "ai-index/caption/cover-page.md",
            "ai-index/caption/output-page.md",
            "ai-index/caption/params.toml",
        ),
    )


def locations_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    if str(settings.get("caption_engine", "")).strip().lower() != "lmstudio":
        return ""
    return _sha16(
        settings.get("caption_engine", ""),
        settings.get("caption_model", ""),
        output_hashes.get("caption", ""),
        settings.get("nominatim_base_url", ""),
        _prompt_hash_payload(
            "ai-index/locations/system.md",
            "ai-index/locations/user.md",
            "ai-index/locations/output-location.md",
            "ai-index/locations/output-shown.md",
            "ai-index/locations/params.toml",
        ),
    )


def objects_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    model = str(settings.get("model", "")).strip()
    if not model or not settings.get("enable_objects", True):
        return ""
    return _sha16(model)


def date_estimate_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    if str(settings.get("caption_engine", "")).strip().lower() != "lmstudio":
        return ""
    model = str(settings.get("caption_model", "")).strip()
    if not model:
        return ""
    return _sha16(
        model,
        output_hashes.get("ocr", ""),
        output_hashes.get("caption", ""),
        _prompt_hash_payload(
            "ai-index/date-estimate/system.md",
            "ai-index/date-estimate/user.md",
            "ai-index/date-estimate/output.md",
            "ai-index/date-estimate/params.toml",
        ),
    )


def propagate_to_crops_input_hash(settings: dict[str, Any], output_hashes: dict[str, str]) -> str:
    return _sha16(
        output_hashes.get("locations", ""),
        output_hashes.get("people", ""),
        settings.get("crop_paths_signature", ""),
    )


STEP_HASH_FNS: dict[str, Callable[[dict[str, Any], dict[str, str]], str]] = {
    "ocr": ocr_input_hash,
    "people": people_input_hash,
    "caption": caption_input_hash,
    "locations": locations_input_hash,
    "objects": objects_input_hash,
    "date-estimate": date_estimate_input_hash,
    "propagate-to-crops": propagate_to_crops_input_hash,
}


# ── StepRunner ─────────────────────────────────────────────────────────────────

class StepRunner:
    """Evaluates staleness per step, skips fresh steps, runs stale steps, accumulates records."""

    def __init__(
        self,
        *,
        settings: dict[str, Any],
        existing_pipeline_state: dict[str, dict],
        existing_detections: dict[str, Any],
        forced_steps: set[str],
    ) -> None:
        self.settings = settings
        self.existing_pipeline_state = existing_pipeline_state
        self.existing_detections = existing_detections
        self.forced_steps = forced_steps
        # Track which steps ran this session
        self.reran: dict[str, bool] = {}
        # Computed output hash per step (for downstream input hashes)
        self.output_hashes: dict[str, str] = {}
        # Step records to write atomically with the final XMP write
        self.pending_records: dict[str, dict] = {}

    def _compute_input_hash(self, step_name: str) -> str:
        fn = STEP_HASH_FNS.get(step_name)
        if fn is None:
            return ""
        return fn(self.settings, self.output_hashes)

    def _is_stale(self, step_name: str, current_hash: str) -> bool:
        if step_name in self.forced_steps:
            return True
        step_key = f"ai-index/{step_name}"
        recorded = self.existing_pipeline_state.get(step_key)
        if recorded is None:
            return True
        recorded_hash = str(recorded.get("input_hash", ""))
        if recorded_hash != current_hash or not current_hash:
            return True
        step = STEPS[step_name]
        if any(self.reran.get(dep, False) for dep in step.depends_on):
            return True
        return False

    def run(
        self,
        step_name: str,
        fn: Callable[[], dict[str, Any]],
        *,
        model: str = "",
    ) -> dict[str, Any]:
        """Run or skip a step.

        fn() should return a dict of output payload keys to merge.
        Returns the output dict (from cache or fresh execution).
        """
        step = STEPS[step_name]
        current_hash = self._compute_input_hash(step_name)

        if not self._is_stale(step_name, current_hash):
            # Step is fresh — load cached output from existing detections
            self.reran[step_name] = False
            cached = {k: self.existing_detections.get(k) for k in step.output_keys}
            self.output_hashes[step_name] = _sha16_data({k: cached.get(k) for k in step.output_keys})
            return cached

        # Step is stale — run it
        try:
            output = fn()
            result = "ok"
        except Exception:
            self.reran[step_name] = True
            step_key = f"ai-index/{step_name}"
            self.pending_records[step_key] = {
                "timestamp": xmp_datetime_now(),
                "input_hash": current_hash,
                "result": "error",
                **({"model": str(model)} if model else {}),
            }
            raise

        # Handle not-applicable (fn returns None)
        if output is None:
            self.reran[step_name] = False
            step_key = f"ai-index/{step_name}"
            self.pending_records[step_key] = {
                "timestamp": xmp_datetime_now(),
                "input_hash": "",
                "result": "not-applicable",
            }
            return {}

        self.reran[step_name] = True
        self.output_hashes[step_name] = _sha16_data({k: output.get(k) for k in step.output_keys})

        step_key = f"ai-index/{step_name}"
        self.pending_records[step_key] = {
            "timestamp": xmp_datetime_now(),
            "input_hash": current_hash,
            "result": result,
            **({"model": str(model)} if model else {}),
        }
        return output

    def get_pending_records(self) -> dict[str, dict]:
        """Return step records to be written atomically with the final XMP payload."""
        return dict(self.pending_records)
