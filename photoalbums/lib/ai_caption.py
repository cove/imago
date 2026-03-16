from __future__ import annotations

import urllib.request  # noqa: F401 — kept at module level for test patching via ai_caption.urllib.request
from dataclasses import dataclass
from pathlib import Path

from ._caption_album import (  # noqa: F401
    ALBUM_KIND_FAMILY,
    ALBUM_KIND_PHOTO_ESSAY,
    AlbumContext,
    clean_text,
    dedupe,
    infer_album_context,
    infer_album_title,
    infer_printed_album_title,
    join_human,
    looks_like_album_cover,
)
from ._caption_lmstudio import (  # noqa: F401
    DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    LMSTUDIO_VISION_MODEL_HINTS,
    CaptionDetails,
    LMStudioCaptioner,
    _build_data_url,
    _extract_structured_json_payload,
    _looks_like_reasoning_or_prompt_echo,
    _normalize_gps_value,
    _parse_lmstudio_structured_caption,
    _parse_lmstudio_structured_caption_payload,
    _resize_caption_image,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from ._caption_prompts import (  # noqa: F401
    _build_combined_qwen_prompt,
    _build_describe_prompt,
    _build_qwen_prompt,
    _build_shared_prompt_rules,
    _should_apply_album_prompt_rules,
    build_page_caption,
    build_template_caption,
)
from ._caption_qwen import (  # noqa: F401
    DEFAULT_QWEN_AUTO_MAX_PIXELS,
    DEFAULT_QWEN_CAPTION_MODEL,
    LEGACY_QWEN_CAPTION_MODEL_ALIASES,
    QWEN_ATTN_IMPLEMENTATIONS,
    QwenLocalCaptioner,
    _load_qwen_transformers,
    _parse_qwen_combined_json_output,
    _parse_qwen_json_output,
    _resolve_local_hf_snapshot,
    normalize_qwen_attn_implementation,
)


def resolve_caption_model(engine: str, model_name: str) -> str:
    normalized = str(engine or "").strip().lower()
    if normalized == "blip":
        normalized = "qwen"
    text = str(model_name or "").strip()
    if text and normalized == "qwen":
        alias = LEGACY_QWEN_CAPTION_MODEL_ALIASES.get(text.casefold())
        if alias:
            return alias
    if text:
        return text
    if normalized == "qwen":
        return DEFAULT_QWEN_CAPTION_MODEL
    return ""


@dataclass
class CaptionOutput:
    text: str
    engine: str
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""
    fallback: bool = False
    error: str = ""


class CaptionEngine:
    def __init__(
        self,
        *,
        engine: str = "qwen",
        model_name: str = "",
        caption_prompt: str = "",
        max_tokens: int = 96,
        temperature: float = 0.2,
        qwen_attn_implementation: str = "auto",
        qwen_min_pixels: int = 0,
        qwen_max_pixels: int = 0,
        lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        max_image_edge: int = 0,
        fallback_to_template: bool = True,
        stream: bool = False,
    ):
        normalized = str(engine or "qwen").strip().lower()
        if normalized == "blip":
            normalized = "qwen"
        if normalized not in {"none", "template", "qwen", "lmstudio"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self.fallback_to_template = bool(fallback_to_template)
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._caption_prompt = str(caption_prompt or "").strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._qwen_attn_implementation = normalize_qwen_attn_implementation(
            qwen_attn_implementation
        )
        self._qwen_min_pixels = max(0, int(qwen_min_pixels))
        self._qwen_max_pixels = max(0, int(qwen_max_pixels))
        self._lmstudio_base_url = normalize_lmstudio_base_url(lmstudio_base_url)
        self._max_image_edge = max(0, int(max_image_edge))
        self._stream = bool(stream)

    def _ensure_captioner(self) -> None:
        if self._captioner is not None:
            return
        if self.engine == "lmstudio":
            self._captioner = LMStudioCaptioner(
                model_name=self._model_name,
                prompt_text=self._caption_prompt,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                base_url=self._lmstudio_base_url,
                max_image_edge=self._max_image_edge,
                stream=self._stream,
            )
        else:
            self._captioner = QwenLocalCaptioner(
                model_name=self._model_name,
                prompt_text=self._caption_prompt,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                attn_implementation=self._qwen_attn_implementation,
                min_pixels=self._qwen_min_pixels,
                max_pixels=self._qwen_max_pixels,
                max_image_edge=self._max_image_edge,
                stream=self._stream,
            )

    def generate(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> CaptionOutput:
        context = infer_album_context(
            image_path=source_path or image_path,
            ocr_text=ocr_text,
            allow_ocr=True,
            album_title=album_title,
            printed_album_title=printed_album_title,
        )
        template = build_template_caption(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            album_context=context,
        )
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        if self.engine == "template":
            return CaptionOutput(
                text=template,
                engine="template",
                gps_latitude="",
                gps_longitude="",
                location_name="",
            )
        self._ensure_captioner()
        prompt = _build_describe_prompt(
            self._caption_prompt,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        try:
            caption = self._captioner.describe(
                image_path=image_path,
                prompt=prompt,
            )
            if caption.text:
                return CaptionOutput(
                    text=caption.text,
                    engine=self.engine,
                    gps_latitude=caption.gps_latitude,
                    gps_longitude=caption.gps_longitude,
                    location_name=caption.location_name,
                )
            if not self.fallback_to_template:
                return CaptionOutput(
                    text="",
                    engine=self.engine,
                    fallback=True,
                    error=f"{self.engine.upper()} returned empty output.",
                )
            return CaptionOutput(
                text=template,
                engine="template",
                gps_latitude="",
                gps_longitude="",
                location_name="",
                fallback=True,
                error=f"{self.engine.upper()} returned empty output.",
            )
        except Exception as exc:
            if not self.fallback_to_template:
                raise
            return CaptionOutput(
                text=template, engine="template", fallback=True, error=str(exc)
            )

    def generate_combined(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> tuple[CaptionOutput, str]:
        """Single Qwen inference for both OCR and caption. Returns (CaptionOutput, ocr_text).
        Only valid when engine == 'qwen'. Falls back to empty ocr_text on error."""
        if self.engine != "qwen":
            return (
                CaptionOutput(
                    text="",
                    engine=self.engine,
                    fallback=True,
                    error="generate_combined requires qwen engine",
                ),
                "",
            )
        self._ensure_captioner()
        _kw = {
            "people": people,
            "objects": objects,
            "source_path": source_path or image_path,
            "album_title": album_title,
            "printed_album_title": printed_album_title,
            "photo_count": photo_count,
            "is_cover_page": is_cover_page,
        }
        try:
            ocr_text, caption = self._captioner.describe_combined(
                image_path=image_path, **_kw
            )
            if caption:
                return (
                    CaptionOutput(
                        text=caption,
                        engine=self.engine,
                        gps_latitude="",
                        gps_longitude="",
                        location_name="",
                    ),
                    ocr_text,
                )
            template = build_template_caption(
                people=people,
                objects=[],
                ocr_text=ocr_text,
                album_context=infer_album_context(
                    image_path=source_path or image_path,
                    ocr_text=ocr_text,
                    allow_ocr=True,
                    album_title=album_title,
                    printed_album_title=printed_album_title,
                ),
            )
            return (
                CaptionOutput(
                    text=template,
                    engine="template",
                    fallback=True,
                    error="Qwen combined returned empty description.",
                ),
                ocr_text,
            )
        except Exception as exc:
            return (
                CaptionOutput(
                    text="", engine="template", fallback=True, error=str(exc)
                ),
                "",
            )
