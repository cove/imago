from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


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
    return json_schema_response_format(
        schema_name=schema_name,
        properties={
            str(field_name): {"type": "string"},
        },
        required=[str(field_name)],
    )


def json_schema_response_format(
    *,
    schema_name: str,
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": str(schema_name),
            "strict": True,
            "schema": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _normalize_model_name_candidates(model_name: str | Sequence[str]) -> list[str]:
    if isinstance(model_name, str):
        candidates = [model_name]
    else:
        candidates = list(model_name or [])
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_candidate in candidates:
        candidate = str(raw_candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


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
    model_names: list[str]
    timeout_seconds: float
    last_response_text: str
    last_finish_reason: str
    _select_model_name: Callable[[str, str, float], str]

    @property
    def effective_model_name(self) -> str:
        return str(self._resolved_model_name or self.model_name)

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

    def _run_with_model_fallback(self, action):
        errors: list[str] = []
        candidates = _normalize_model_name_candidates(self.model_names or [self.model_name])
        if not candidates:
            candidates = [""]
        last_error: Exception | None = None
        for candidate in candidates:
            self.model_name = candidate
            self._resolved_model_name = ""
            self.last_response_text = ""
            self.last_finish_reason = ""
            try:
                result = action()
            except Exception as exc:
                last_error = exc
                errors.append(f"{candidate}: {exc}")
                continue
            if not self._resolved_model_name:
                self._resolved_model_name = str(candidate or "").strip()
            return result
        if last_error is not None and len(errors) <= 1:
            raise last_error
        attempted = "; ".join(errors)
        raise RuntimeError(f"LM Studio model fallback failed: {attempted}") from last_error
