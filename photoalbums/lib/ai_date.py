from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ._caption_lmstudio import (
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _decode_lmstudio_text,
    _extract_structured_json_payload,
    _format_lmstudio_debug_response,
    _lmstudio_request_json,
    _normalize_model_name_candidates,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from ._lmstudio_helpers import LMStudioModelResolverMixin, emit_prompt_debug, single_string_response_format
from ._prompt_skill import required_section_text
from .ai_model_settings import default_caption_model, default_caption_models, default_lmstudio_base_url
from .xmp_sidecar import _normalize_dc_date


def _date_estimate_response_format() -> dict[str, object]:
    return single_string_response_format(schema_name="date_estimate_payload", field_name="date")


def _is_date_estimate_payload(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("date"), str)


def date_estimate_system_prompt() -> str:
    return required_section_text("System Prompt - Date Estimate")


def _build_date_estimate_prompt(*, ocr_text: str, album_title: str) -> str:
    lines = [required_section_text("Preamble Date Estimate")]
    clean_album_title = str(album_title or "").strip()
    clean_ocr_text = str(ocr_text or "").strip()
    if clean_album_title:
        lines.append(f"Album title: {clean_album_title}")
    if clean_ocr_text:
        lines.append("OCR text:")
        lines.append(clean_ocr_text)
    else:
        lines.append("OCR text: ")
    lines.append(required_section_text("Output Format - Date Estimate"))
    return "\n".join(lines)


def _parse_date_estimate(value: object, *, finish_reason: str = "") -> str:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty date estimate content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(text, is_valid=_is_date_estimate_payload)
        if payload is None:
            raise RuntimeError(f"LM Studio returned invalid date estimate JSON: {text}") from exc
    if not _is_date_estimate_payload(payload):
        raise RuntimeError(f"LM Studio returned invalid date estimate payload: {text}")
    date_text = str(payload.get("date") or "").strip()
    normalized = _normalize_dc_date(date_text)
    if date_text and not normalized:
        raise RuntimeError(f"LM Studio returned invalid dc:date value: {date_text}")
    return normalized


@dataclass
class DateEstimateOutput:
    engine: str
    date: str = ""
    fallback: bool = False
    error: str = ""


class DateEstimateEngine(LMStudioModelResolverMixin):
    _select_model_name = staticmethod(_select_lmstudio_model)

    def __init__(
        self,
        *,
        engine: str = "lmstudio",
        model_name: str = "",
        lmstudio_base_url: str = "",
        max_tokens: int = 96,
        temperature: float = 0.0,
    ) -> None:
        normalized = str(engine or "lmstudio").strip().lower()
        if normalized in {"qwen", "blip", "local"}:
            normalized = "lmstudio"
        if normalized not in {"none", "lmstudio"}:
            raise ValueError(f"Unsupported date estimate engine: {engine}")
        self.engine = normalized
        if str(model_name or "").strip():
            self.model_names = [str(model_name).strip()]
        else:
            self.model_names = default_caption_models() or ([default_caption_model()] if default_caption_model() else [])
        self.model_name = self.model_names[0] if self.model_names else ""
        self.base_url = normalize_lmstudio_base_url(
            str(lmstudio_base_url or "").strip(),
            default=default_lmstudio_base_url(),
        )
        self.max_tokens = max(32, int(max_tokens))
        self.temperature = max(0.0, float(temperature))
        self.timeout_seconds = DEFAULT_LMSTUDIO_TIMEOUT_SECONDS
        self._resolved_model_name = ""
        self.last_response_text = ""
        self.last_finish_reason = ""

    @property
    def effective_model_name(self) -> str:
        return str(self._resolved_model_name or self.model_name)

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

    def estimate(
        self,
        *,
        ocr_text: str,
        album_title: str,
        source_path: str | Path | None = None,
        debug_recorder=None,
        debug_step: str = "date_estimate",
    ) -> DateEstimateOutput:
        prompt = _build_date_estimate_prompt(ocr_text=ocr_text, album_title=album_title)
        system_prompt = date_estimate_system_prompt()
        response = ""
        finish_reason = ""
        error_text = ""
        if self.engine == "none":
            emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model="",
                prompt=prompt,
                system_prompt=system_prompt,
                source_path=source_path,
                prompt_source="skill",
                response="",
                finish_reason="",
                metadata={"skipped": True},
            )
            return DateEstimateOutput(engine=self.engine, fallback=True, error="")
        try:
            def run_request() -> str:
                payload = {
                    "model": self._resolve_model_name(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": _date_estimate_response_format(),
                    "max_tokens": min(128, int(self.max_tokens)),
                    "temperature": float(self.temperature),
                    "stream": False,
                }
                response_payload = _lmstudio_request_json(
                    f"{self.base_url}/chat/completions",
                    payload=payload,
                    timeout=self.timeout_seconds,
                )
                choices = list(response_payload.get("choices") or [])
                if not choices:
                    raise RuntimeError("LM Studio returned no choices.")
                message = dict(choices[0].get("message") or {})
                self.last_finish_reason = str(choices[0].get("finish_reason") or "")
                self.last_response_text = _format_lmstudio_debug_response(message.get("content"))
                if not self.last_response_text:
                    self.last_response_text = _format_lmstudio_debug_response(message)
                return _parse_date_estimate(
                    message.get("content"),
                    finish_reason=self.last_finish_reason,
                )

            estimated_date = self._run_with_model_fallback(run_request)
            response = self.last_response_text
            finish_reason = self.last_finish_reason
            return DateEstimateOutput(engine=self.engine, date=estimated_date, fallback=False, error="")
        except Exception as exc:
            response = str(self.last_response_text or "")
            finish_reason = str(self.last_finish_reason or "")
            error_text = str(exc)
            return DateEstimateOutput(engine=self.engine, fallback=True, error=error_text)
        finally:
            metadata: dict[str, object] = {}
            if error_text:
                metadata["error"] = error_text
            emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name if self.engine != "none" else "",
                prompt=prompt,
                system_prompt=system_prompt,
                source_path=source_path,
                prompt_source="skill",
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )
