from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..naming import parse_album_filename
from ._caption_lmstudio import (
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _decode_lmstudio_text,
    _extract_structured_json_payload,
    _format_lmstudio_debug_response,
    _lmstudio_request_json,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from ._lmstudio_helpers import LMStudioModelResolverMixin, emit_prompt_debug, single_string_response_format
from .ai_model_settings import default_caption_model, default_caption_models, default_lmstudio_base_url
from .xmp_sidecar import _normalize_dc_date


def _date_estimate_response_format() -> dict[str, object]:
    return single_string_response_format(schema_name="date_estimate_payload", field_name="date")


def _is_date_estimate_payload(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("date"), str)


_DATE_SYSTEM_PROMPT = (
    "- You estimate a photo date for XMP `dc:date`.\n"
    "- Return only valid JSON matching the response_format schema.\n"
    "- Use OCR text as the primary source of truth.\n"
    "- If OCR text does not support a date, fall back to the filename year, then the album title.\n"
    "- When OCR text supports a month or day but no year, use the filename year if available, otherwise the album title year.\n"
    "- Do not infer a different year from visual style when a filename year or album title year is available.\n"
    "- Return the most precise supported W3C date value: `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, or an empty string.\n"
    "- Do not invent a month or day unless the supplied text supports it.\n"
    "- If the source only supports an approximate or rounded date, return the nearest honest precision instead of inventing missing parts.\n"
    "- Never use placeholder components like `00` for month or day.\n"
    "- If the day is unknown, return `YYYY-MM` instead of `YYYY-MM-00`.\n"
    "- If the month is unknown, return `YYYY` instead of `YYYY-00` or `YYYY-00-00`.\n"
    "- Do not include reasoning or extra fields."
)
_DATE_USER_PROMPT = (
    "- Estimate a single photo date for XMP `dc:date`.\n"
    "- First look for an explicit or strongly implied date in the OCR text.\n"
    "- If the OCR text does not support a date, use the filename year as the fallback year, then the album title year.\n"
    "- If OCR text gives a month or day without a year, combine that precision with the filename year, then the album title year.\n"
    "- Treat the filename year as album context, not as a technical capture date.\n"
    "- Treat month abbreviations with or without a trailing period as explicit month evidence, even when OCR spacing is imperfect.\n"
    "- When only a year is supported, return the year only.\n"
    "- When a month and year are supported, return `YYYY-MM`.\n"
    "- When a full calendar date is supported, return `YYYY-MM-DD`.\n"
    "- If the visible text implies an approximate date, round to the nearest supported precision without adding unsupported detail.\n"
    "- Example: `AUG. 1988` -> `1988-08`.\n"
    "- Example: `AUG.1988` -> `1988-08`.\n"
    "- Example: `AUGUST 1988` -> `1988-08`.\n"
    "- Example: `about January 1988` -> `1988-01`.\n"
    "- Example: `early 1988` -> `1988`.\n"
    "- Example: `winter 1988` -> `1988`.\n"
    "- Never return `00` for an unknown month or day; reduce precision instead.\n"
    "- Example: `January 1988` -> `1988-01`, not `1988-01-00`.\n"
    "- Example: `1988` -> `1988`, not `1988-00` or `1988-00-00`.\n"
    "- If neither OCR text, filename, nor album title supports any date estimate, return the empty string."
)
_DATE_OUTPUT_PROMPT = (
    '`{"date": ""}`\n'
    '- `date`: estimated W3C date string for `dc:date`, using one of `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, or `""`.\n'
    "- Prefer OCR evidence over filename or album-title fallback.\n"
    "- Just return the JSON without any extra text or explanation."
)


def date_estimate_system_prompt() -> str:
    return _DATE_SYSTEM_PROMPT


# The date JSON is tiny, but reasoning ("thinking") models emit a long preamble
# before the structured answer. A small cap truncates them (finish_reason=length,
# empty content) before they reach the date. Non-reasoning models stop as soon as
# the JSON is emitted, so a generous cap is harmless for them.
DEFAULT_DATE_MAX_TOKENS = 2048


def _date_params() -> dict[str, object]:
    return {"max_tokens": DEFAULT_DATE_MAX_TOKENS, "temperature": 0.0, "timeout_seconds": 300.0}


def _date_prompt_metadata(resolved_params: dict[str, object]) -> dict[str, object]:
    return {}


def _filename_year(source_path: str | Path | None) -> str:
    if source_path is None:
        return ""
    _collection, year, _book, _page = parse_album_filename(Path(source_path).name)
    return year if year != "Unknown" else ""


def _build_date_estimate_prompt(*, ocr_text: str, album_title: str, source_path: str | Path | None = None) -> str:
    lines = [_DATE_USER_PROMPT]
    clean_album_title = str(album_title or "").strip()
    clean_ocr_text = str(ocr_text or "").strip()
    clean_source_name = Path(source_path).name if source_path is not None else ""
    clean_filename_year = _filename_year(source_path)
    if clean_source_name:
        lines.append(f"Image file: {clean_source_name}")
    if clean_filename_year:
        lines.append(f"Filename year: {clean_filename_year}")
    if clean_album_title:
        lines.append(f"Album title: {clean_album_title}")
    if clean_ocr_text:
        lines.append("OCR text:")
        lines.append(clean_ocr_text)
    else:
        lines.append("OCR text: ")
    lines.append(_DATE_OUTPUT_PROMPT)
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
        max_tokens: int = DEFAULT_DATE_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        params = _date_params()
        normalized = str(engine or "lmstudio").strip().lower()
        if normalized in {"qwen", "blip", "local"}:
            normalized = "lmstudio"
        if normalized not in {"none", "lmstudio"}:
            raise ValueError(f"Unsupported date estimate engine: {engine}")
        self.engine = normalized
        if str(model_name or "").strip():
            self.model_names = [str(model_name).strip()]
        else:
            self.model_names = default_caption_models() or (
                [default_caption_model()] if default_caption_model() else []
            )
        self.model_name = self.model_names[0] if self.model_names else ""
        self.base_url = normalize_lmstudio_base_url(
            str(lmstudio_base_url or "").strip(),
            default=default_lmstudio_base_url(),
        )
        self.max_tokens = max(
            32, int(max_tokens if max_tokens is not None else params.get("max_tokens", DEFAULT_DATE_MAX_TOKENS))
        )
        self.temperature = max(0.0, float(temperature if temperature is not None else params.get("temperature", 0.0)))
        self.timeout_seconds = float(params.get("timeout_seconds", DEFAULT_LMSTUDIO_TIMEOUT_SECONDS))
        self._resolved_model_name = ""
        self.last_response_text = ""
        self.last_finish_reason = ""

    def estimate(
        self,
        *,
        ocr_text: str,
        album_title: str,
        source_path: str | Path | None = None,
        prompt_prefix: str = "",
        debug_recorder=None,
        debug_step: str = "date_estimate",
    ) -> DateEstimateOutput:
        base_prompt = _build_date_estimate_prompt(ocr_text=ocr_text, album_title=album_title, source_path=source_path)
        prompt = "\n\n".join(part for part in (str(prompt_prefix or "").strip(), base_prompt) if part).strip()
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
                prompt_source="photoalbums/prompts/ai-index/date-estimate",
                response="",
                finish_reason="",
                metadata={
                    **_date_prompt_metadata(
                        {
                            "max_tokens": int(self.max_tokens),
                            "temperature": float(self.temperature),
                            "timeout_seconds": float(self.timeout_seconds),
                        }
                    ),
                    "skipped": True,
                },
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
                    "max_tokens": int(self.max_tokens),
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
            metadata.update(
                _date_prompt_metadata(
                    {
                        "max_tokens": int(self.max_tokens),
                        "temperature": float(self.temperature),
                        "timeout_seconds": float(self.timeout_seconds),
                    }
                )
            )
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
                prompt_source="photoalbums/prompts/ai-index/date-estimate",
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )
