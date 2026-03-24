from __future__ import annotations

import urllib.request  # noqa: F401 — kept at module level for test patching via ai_caption.urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ai_model_settings import default_caption_model
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
    _build_describe_prompt,
    _build_location_prompt,
    _build_people_count_prompt,
    _build_local_prompt,
    _build_shared_prompt_rules,
    _should_apply_album_prompt_rules,
)
from ._caption_local_hf import (  # noqa: F401
    DEFAULT_LOCAL_AUTO_MAX_PIXELS,
    DEFAULT_LOCAL_CAPTION_MODEL,
    LOCAL_ATTN_IMPLEMENTATIONS,
    LocalHFCaptioner,
    _load_hf_transformers,
    _parse_local_json_output,
    _resolve_local_hf_snapshot,
    normalize_local_attn_implementation,
)


def resolve_caption_model(engine: str, model_name: str) -> str:
    normalized = str(engine or "").strip().lower()
    if normalized == "qwen":
        normalized = "local"
    if normalized == "blip":
        normalized = "local"
    text = str(model_name or "").strip()
    if text:
        return text
    configured = default_caption_model()
    if configured:
        return configured
    if normalized == "local":
        return DEFAULT_LOCAL_CAPTION_MODEL
    return ""


@dataclass
class CaptionOutput:
    text: str
    engine: str
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""
    people_present: bool = False
    estimated_people_count: int = 0
    fallback: bool = False
    error: str = ""
    image_regions: list[dict] = None
    album_title: str = ""
    title: str = ""

    def __post_init__(self):
        if self.image_regions is None:
            self.image_regions = []


@dataclass
class PeopleCountOutput:
    engine: str
    people_present: bool = False
    estimated_people_count: int = 0
    fallback: bool = False
    error: str = ""


@dataclass
class LocationOutput:
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
        engine: str = "local",
        model_name: str = "",
        caption_prompt: str = "",
        max_tokens: int = 96,
        temperature: float = 0.2,
        local_attn_implementation: str = "auto",
        local_min_pixels: int = 0,
        local_max_pixels: int = 0,
        lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        max_image_edge: int = 0,
        stream: bool = False,
    ):
        normalized = str(engine or "local").strip().lower()
        if normalized == "qwen":
            normalized = "local"
        if normalized == "blip":
            normalized = "local"
        if normalized not in {"none", "local", "lmstudio"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._caption_prompt = str(caption_prompt or "").strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._local_attn_implementation = normalize_local_attn_implementation(local_attn_implementation)
        self._local_min_pixels = max(0, int(local_min_pixels))
        self._local_max_pixels = max(0, int(local_max_pixels))
        self._lmstudio_base_url = normalize_lmstudio_base_url(lmstudio_base_url)
        self._max_image_edge = max(0, int(max_image_edge))
        self._stream = bool(stream)

    @property
    def effective_model_name(self) -> str:
        """Return the actual model name used, resolved after any lazy API lookup."""
        if self.engine == "none":
            return ""
        if self.engine == "lmstudio" and self._captioner is not None:
            resolved = str(getattr(self._captioner, "_resolved_model_name", "") or "")
            return resolved or self._model_name
        return str(self._model_name)

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
            self._captioner = LocalHFCaptioner(
                model_name=self._model_name,
                prompt_text=self._caption_prompt,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                attn_implementation=self._local_attn_implementation,
                min_pixels=self._local_min_pixels,
                max_pixels=self._local_max_pixels,
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
        people_positions: dict[str, str] | None = None,
        request_photo_regions: bool = False,
    ) -> CaptionOutput:
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        self._ensure_captioner()
        use_page_mode = request_photo_regions and self.engine == "lmstudio"
        prompt = _build_describe_prompt(
            self._caption_prompt,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
            request_photo_regions=use_page_mode,
        )
        try:
            if use_page_mode:
                caption = self._captioner.describe_page(  # type: ignore[attr-defined]
                    image_path=image_path,
                    prompt=prompt,
                )
            else:
                caption = self._captioner.describe(
                    image_path=image_path,
                    prompt=prompt,
                )
            return CaptionOutput(
                text=caption.text,
                engine=self.engine,
                gps_latitude=caption.gps_latitude,
                gps_longitude=caption.gps_longitude,
                location_name=caption.location_name,
                people_present=bool(getattr(caption, "people_present", False)),
                estimated_people_count=max(0, int(getattr(caption, "estimated_people_count", 0) or 0)),
                fallback=not caption.text,
                error=("" if caption.text else f"{self.engine.upper()} returned empty output."),
                image_regions=list(getattr(caption, "image_regions", None) or []),
                album_title=str(getattr(caption, "album_title", "") or ""),
                title=str(getattr(caption, "title", "") or ""),
            )
        except Exception as exc:
            return CaptionOutput(text="", engine=self.engine, fallback=True, error=str(exc))

    def estimate_people(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        people_positions: dict[str, str] | None = None,
    ) -> PeopleCountOutput:
        if self.engine != "lmstudio":
            return PeopleCountOutput(
                engine=self.engine,
                fallback=True,
                error=f"{self.engine.upper()} people counting is not implemented.",
            )
        self._ensure_captioner()
        prompt = _build_people_count_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
        )
        try:
            people_count = self._captioner.estimate_people(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            return PeopleCountOutput(
                engine=self.engine,
                people_present=bool(people_count.people_present),
                estimated_people_count=max(0, int(getattr(people_count, "estimated_people_count", 0) or 0)),
                fallback=False,
            )
        except Exception as exc:
            return PeopleCountOutput(
                engine=self.engine,
                fallback=True,
                error=str(exc),
            )

    def estimate_location(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        is_cover_page: bool = False,
        people_positions: dict[str, str] | None = None,
    ) -> LocationOutput:
        if self.engine != "lmstudio":
            return LocationOutput(
                engine=self.engine,
                fallback=True,
                error=f"{self.engine.upper()} location estimation is not implemented.",
            )
        self._ensure_captioner()
        prompt = _build_location_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
        )
        try:
            location = self._captioner.estimate_location(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            return LocationOutput(
                engine=self.engine,
                gps_latitude=str(getattr(location, "gps_latitude", "") or "").strip(),
                gps_longitude=str(getattr(location, "gps_longitude", "") or "").strip(),
                location_name=str(getattr(location, "location_name", "") or "").strip(),
                fallback=False,
            )
        except Exception as exc:
            return LocationOutput(
                engine=self.engine,
                fallback=True,
                error=str(exc),
            )
