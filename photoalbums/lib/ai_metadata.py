from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ._caption_lmstudio import (
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _build_data_url,
    _decode_lmstudio_text,
    _extract_structured_json_payload,
    _lmstudio_request_json,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from ._lmstudio_helpers import LMStudioModelResolverMixin, emit_prompt_debug
from .ai_lmstudio_structured import run_engine_with_model_fallback, schema_response_format, step_prompt_assets_metadata
from .ai_model_settings import default_caption_model, default_caption_models, default_lmstudio_base_url
from .ai_prompt_assets import load_params, load_prompt

_DEFAULT_MAX_IMAGE_EDGE = 1920


def _metadata_response_format() -> dict:
    return schema_response_format(schema_name="metadata_payload", schema_path="ai-index/metadata/schema.json")


def _metadata_system_prompt() -> str:
    return load_prompt("ai-index/metadata/system.md").rendered


def _metadata_user_prompt(album_title: str = "") -> str:
    return load_prompt(
        "ai-index/metadata/user.md",
        variables={"album_title": str(album_title or "").strip()} if album_title else None,
    ).rendered


def _metadata_params() -> dict:
    return dict(load_params("ai-index/metadata/params.toml").values)


def _metadata_prompt_metadata(resolved_params: dict) -> dict:
    return step_prompt_assets_metadata(step="metadata", resolved_params=resolved_params)


def _is_metadata_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("photos"), list)


@dataclass
class MetadataPhotoResult:
    photo_number: int = 0
    location: str = ""
    location_name: str = ""
    est_date: str = ""
    scene_ocr: str = ""
    caption: str = ""
    corrected_caption: str = ""
    people_count: int = 0


@dataclass
class MetadataResult:
    engine: str = ""
    photos: list[MetadataPhotoResult] = field(default_factory=list)
    fallback: bool = False
    error: str = ""

    @property
    def people_count(self) -> int:
        return sum(p.people_count for p in self.photos)


def _parse_metadata_response(value: object, *, finish_reason: str = "") -> MetadataResult:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(f"LM Studio returned empty metadata content.{finish_note}")
    payload = _metadata_payload_from_text(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"LM Studio returned non-dict metadata: {text}")
    return MetadataResult(photos=_metadata_photos_from_payload(payload))


def _metadata_payload_from_text(text: str) -> object:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(text, is_valid=_is_metadata_payload)
        if payload is None:
            raise RuntimeError(f"LM Studio returned invalid metadata JSON: {text}") from exc
    # If the top-level parse succeeded but the document doesn't match the
    # expected schema (e.g. the model emitted a reasoning preamble that
    # itself happened to be valid JSON), search the full text for a payload
    # that does.
    if not _is_metadata_payload(payload):
        alt = _extract_structured_json_payload(text, is_valid=_is_metadata_payload)
        if alt is not None:
            payload = alt
    return payload


def _metadata_photos_from_payload(payload: dict) -> list[MetadataPhotoResult]:
    photos: list[MetadataPhotoResult] = []
    for photo_data in list(payload.get("photos") or []):
        if isinstance(photo_data, dict):
            photos.append(_metadata_photo_from_dict(photo_data))
    return photos


def _metadata_photo_from_dict(photo_data: dict) -> MetadataPhotoResult:
    return MetadataPhotoResult(
        photo_number=int(photo_data.get("photo_number") or 0),
        location=str(photo_data.get("location") or "").strip(),
        location_name=str(photo_data.get("location_name") or "").strip(),
        est_date=str(photo_data.get("est_date") or "").strip(),
        scene_ocr=str(photo_data.get("scene_ocr") or "").strip(),
        caption=str(photo_data.get("caption") or "").strip(),
        corrected_caption=str(photo_data.get("corrected_caption") or "").strip(),
        people_count=int(photo_data.get("people_count") or 0),
    )


class MetadataEngine(LMStudioModelResolverMixin):
    _select_model_name = staticmethod(_select_lmstudio_model)

    def __init__(
        self,
        *,
        engine: str = "lmstudio",
        model_name: str = "",
        lmstudio_base_url: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_image_edge: int = 0,
    ) -> None:
        params = _metadata_params()
        normalized = str(engine or "lmstudio").strip().lower()
        if normalized in {"qwen", "blip", "local"}:
            normalized = "lmstudio"
        if normalized not in {"none", "lmstudio"}:
            raise ValueError(f"Unsupported metadata engine: {engine}")
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
        self.max_tokens = max(256, int(max_tokens if max_tokens is not None else params.get("max_tokens", 2048)))
        self.temperature = max(0.0, float(temperature if temperature is not None else params.get("temperature", 0.1)))
        self.timeout_seconds = float(params.get("timeout_seconds", DEFAULT_LMSTUDIO_TIMEOUT_SECONDS))
        _edge = int(max_image_edge if max_image_edge is not None else params.get("max_image_edge", 0))
        self.max_image_edge = _edge if _edge > 0 else _DEFAULT_MAX_IMAGE_EDGE
        self._resolved_model_name = ""
        self.last_response_text = ""
        self.last_finish_reason = ""

    def analyze(
        self,
        image_path: Path | str,
        *,
        album_title: str = "",
        source_path: Path | str | None = None,
        debug_recorder=None,
        debug_step: str = "metadata",
    ) -> MetadataResult:
        system_prompt = _metadata_system_prompt()
        user_prompt = _metadata_user_prompt(album_title)
        response = ""
        finish_reason = ""
        error_text = ""

        if self.engine == "none":
            emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model="",
                prompt=user_prompt,
                system_prompt=system_prompt,
                source_path=source_path,
                prompt_source="photoalbums/prompts/ai-index/metadata",
                response="",
                finish_reason="",
                metadata={
                    **_metadata_prompt_metadata({"max_tokens": self.max_tokens, "temperature": self.temperature}),
                    "skipped": True,
                },
            )
            return MetadataResult(engine=self.engine, fallback=True, error="")

        try:
            result = run_engine_with_model_fallback(
                self,
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=_metadata_response_format(),
                parse_fn=_parse_metadata_response,
                request_json=_lmstudio_request_json,
                build_data_url=_build_data_url,
            )
            response = self.last_response_text
            finish_reason = self.last_finish_reason
            return result
        except Exception as exc:
            response = str(self.last_response_text or "")
            finish_reason = str(self.last_finish_reason or "")
            error_text = str(exc)
            return MetadataResult(engine=self.engine, fallback=True, error=error_text)
        finally:
            metadata: dict = {}
            metadata.update(
                _metadata_prompt_metadata(
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
                prompt=user_prompt,
                system_prompt=system_prompt,
                source_path=source_path,
                prompt_source="photoalbums/prompts/ai-index/metadata",
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )
