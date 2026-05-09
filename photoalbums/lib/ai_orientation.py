from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from ._caption_lmstudio import (
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _decode_lmstudio_text,
    _extract_structured_json_payload,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from ._lmstudio_helpers import LMStudioModelResolverMixin, emit_prompt_debug
from .ai_lmstudio_structured import run_engine_with_model_fallback, schema_response_format, step_prompt_assets_metadata
from .ai_model_settings import default_caption_model, default_caption_models, default_lmstudio_base_url
from .ai_prompt_assets import load_params, load_prompt

_DEFAULT_MAX_IMAGE_EDGE = 1024


def _orientation_response_format() -> dict:
    return schema_response_format(schema_name="orientation_payload", schema_path="ai-index/orientation/schema.json")


def _orientation_system_prompt() -> str:
    return load_prompt("ai-index/orientation/system.md").rendered


def _orientation_user_prompt() -> str:
    return load_prompt("ai-index/orientation/user.md").rendered


def _orientation_params() -> dict:
    return dict(load_params("ai-index/orientation/params.toml").values)


def _orientation_prompt_metadata(resolved_params: dict) -> dict:
    return step_prompt_assets_metadata(step="orientation", resolved_params=resolved_params)


def _is_orientation_payload(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("right_side_up"), bool)


@dataclass
class OrientationResult:
    engine: str = ""
    right_side_up: bool | None = None
    fallback: bool = False
    error: str = ""


class _OrientationEngineProtocol(Protocol):
    effective_model_name: str

    def analyze(self, image_path: Path | str, *, source_path: Path | str | None = ...) -> OrientationResult: ...


def _parse_orientation_response(value: object, *, finish_reason: str = "") -> OrientationResult:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(f"LM Studio returned empty orientation content.{finish_note}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(text, is_valid=_is_orientation_payload)
        if payload is None:
            raise RuntimeError(f"LM Studio returned invalid orientation JSON: {text}") from exc
    if not _is_orientation_payload(payload):
        payload = _extract_structured_json_payload(text, is_valid=_is_orientation_payload)
    if not _is_orientation_payload(payload):
        raise RuntimeError(f"LM Studio returned invalid orientation payload: {text}")
    return OrientationResult(right_side_up=bool(payload["right_side_up"]))


class OrientationEngine(LMStudioModelResolverMixin):
    _select_model_name = staticmethod(_select_lmstudio_model)

    def __init__(
        self,
        *,
        engine: str = "lmstudio",
        model_name: str = "",
        lmstudio_base_url: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_image_edge: int | None = None,
    ) -> None:
        params = _orientation_params()
        normalized = str(engine or "lmstudio").strip().lower()
        if normalized in {"qwen", "blip", "local"}:
            normalized = "lmstudio"
        if normalized not in {"none", "lmstudio"}:
            raise ValueError(f"Unsupported orientation engine: {engine}")
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
        self.max_tokens = max(8, int(max_tokens if max_tokens is not None else params.get("max_tokens", 32)))
        self.temperature = max(0.0, float(temperature if temperature is not None else params.get("temperature", 0.0)))
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
        source_path: Path | str | None = None,
        debug_recorder=None,
        debug_step: str = "orientation",
    ) -> OrientationResult:
        system_prompt = _orientation_system_prompt()
        user_prompt = _orientation_user_prompt()
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
                prompt_source="photoalbums/prompts/ai-index/orientation",
                response="",
                finish_reason="",
                metadata={
                    **_orientation_prompt_metadata({"max_tokens": self.max_tokens, "temperature": self.temperature}),
                    "skipped": True,
                },
            )
            return OrientationResult(engine=self.engine, fallback=True, error="")

        try:
            result = run_engine_with_model_fallback(
                self,
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=_orientation_response_format(),
                parse_fn=_parse_orientation_response,
            )
            response = self.last_response_text
            finish_reason = self.last_finish_reason
            return result
        except Exception as exc:
            response = str(self.last_response_text or "")
            finish_reason = str(self.last_finish_reason or "")
            error_text = str(exc)
            return OrientationResult(engine=self.engine, fallback=True, error=error_text)
        finally:
            resolved = {
                "max_tokens": int(self.max_tokens),
                "temperature": float(self.temperature),
                "timeout_seconds": float(self.timeout_seconds),
                "max_image_edge": int(self.max_image_edge),
            }
            metadata: dict = {}
            metadata.update(_orientation_prompt_metadata(resolved))
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
                prompt_source="photoalbums/prompts/ai-index/orientation",
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )


def rotate_image_180_in_place(image_path: Path | str) -> None:
    from PIL import Image, ImageSequence  # pylint: disable=import-outside-toplevel

    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    path = Path(image_path)
    fd, temp_name = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(temp_name)
    allow_large_pillow_images(Image)
    frames = []
    try:
        with Image.open(path) as image:
            transpose = getattr(getattr(Image, "Transpose", Image), "ROTATE_180")
            frames = [frame.copy().transpose(transpose) for frame in ImageSequence.Iterator(image)]
            if not frames:
                raise RuntimeError(f"Pillow rotate 180 failed: no image frames in {path}")
            save_kwargs = _pillow_save_kwargs(image, path)
            if len(frames) > 1:
                frames[0].save(tmp_path, save_all=True, append_images=frames[1:], **save_kwargs)
            else:
                frames[0].save(tmp_path, **save_kwargs)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        for frame in frames:
            frame.close()


def _pillow_save_kwargs(image, path: Path) -> dict:
    kwargs: dict = {}
    image_format = str(getattr(image, "format", "") or "").strip()
    if image_format:
        kwargs["format"] = image_format
    if path.suffix.lower() in {".tif", ".tiff"}:
        kwargs["compression"] = "tiff_lzw"
    for key in ("icc_profile", "dpi"):
        if key in image.info:
            kwargs[key] = image.info[key]
    return kwargs


def correct_orientation_after_scan(
    image_path: Path | str,
    *,
    engine: _OrientationEngineProtocol | None = None,
    log_info: Callable[[str], None] | None = None,
) -> dict:
    orientation_engine = engine or OrientationEngine()
    result = orientation_engine.analyze(image_path, source_path=image_path)
    if result.fallback and result.error:
        raise RuntimeError(f"Orientation check failed: {result.error}")
    right_side_up = bool(result.right_side_up)
    rotation_applied_degrees = 0
    if result.right_side_up is False:
        if log_info is not None:
            log_info(f"  [rotate] {Path(image_path).name} 180 degrees")
        rotate_image_180_in_place(image_path)
        right_side_up = True
        rotation_applied_degrees = 180
    return {
        "right_side_up": right_side_up,
        "ai_right_side_up": result.right_side_up,
        "rotation_applied_degrees": rotation_applied_degrees,
        "engine": result.engine,
        "model": str(orientation_engine.effective_model_name),
    }
