from __future__ import annotations

from pathlib import Path
from typing import Callable


def emit_prompt_debug(
    debug_recorder,
    *,
    step: str,
    engine: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    source_path: str | Path | None = None,
    prompt_source: str = "",
    response: str = "",
    finish_reason: str = "",
    metadata: dict | None = None,
) -> None:
    if not callable(debug_recorder):
        return
    debug_recorder(
        step=str(step or "").strip(),
        engine=str(engine or "").strip(),
        model=str(model or "").strip(),
        prompt=str(prompt or ""),
        system_prompt=str(system_prompt or ""),
        source_path=source_path,
        prompt_source=str(prompt_source or "").strip(),
        response=str(response or ""),
        finish_reason=str(finish_reason or "").strip(),
        metadata=dict(metadata or {}),
    )


def single_string_response_format(*, schema_name: str, field_name: str) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": str(schema_name),
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    str(field_name): {"type": "string"},
                },
                "required": [str(field_name)],
                "additionalProperties": False,
            },
        },
    }


def resolve_cached_lmstudio_model_name(
    resolved_name: str,
    resolve_model: Callable[[], str],
) -> str:
    if resolved_name:
        return resolved_name
    return resolve_model()


class LMStudioModelResolverMixin:
    _resolved_model_name: str
    base_url: str
    model_name: str
    timeout_seconds: float
    _select_model_name: Callable[[str, str, float], str]

    def _resolve_model_name(self) -> str:
        self._resolved_model_name = resolve_cached_lmstudio_model_name(
            self._resolved_model_name,
            lambda: self._select_model_name(
                self.base_url,
                self.model_name,
                self.timeout_seconds,
            ),
        )
        return self._resolved_model_name
