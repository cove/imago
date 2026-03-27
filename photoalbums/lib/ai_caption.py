from __future__ import annotations

import urllib.request  # noqa: F401 — kept at module level for test patching via ai_caption.urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ai_model_settings import default_caption_model, default_lmstudio_base_url
from ._caption_context import (  # noqa: F401
    ALBUM_KIND_FAMILY,
    ALBUM_KIND_PHOTO_ESSAY,
    AlbumContext,
    infer_album_context,
    infer_album_title,
    looks_like_album_cover,
)
from ._caption_text import clean_text, dedupe, join_human  # noqa: F401
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
    location_system_prompt,
    normalize_lmstudio_base_url,
    people_count_system_prompt,
)
from ._caption_prompts import (  # noqa: F401
    _build_location_prompt,
    _build_people_count_prompt,
    _build_local_prompt,
)


def _emit_prompt_debug(
    debug_recorder,
    *,
    step: str,
    engine: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    source_path: str | Path | None = None,
    prompt_source: str = "",
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
        metadata=dict(metadata or {}),
    )


def resolve_caption_model(engine: str, model_name: str) -> str:
    normalized = str(engine or "").strip().lower()
    if normalized in {"qwen", "blip", "local"}:
        normalized = "lmstudio"
    text = str(model_name or "").strip()
    if text:
        return text
    configured = default_caption_model()
    if configured:
        return configured
    return ""


@dataclass
class CaptionOutput:
    text: str
    engine: str
    ocr_text: str = ""
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
    author_text: str = ""
    scene_text: str = ""

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
        engine: str = "lmstudio",
        model_name: str = "",
        caption_prompt: str = "",
        max_tokens: int = 96,
        temperature: float = 0.2,
        lmstudio_base_url: str = "",
        max_image_edge: int = 0,
        stream: bool = False,
    ):
        normalized = str(engine or "lmstudio").strip().lower()
        if normalized in {"qwen", "blip", "local"}:
            normalized = "lmstudio"
        if normalized not in {"none", "lmstudio"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._caption_prompt = str(caption_prompt or "").strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._lmstudio_base_url = normalize_lmstudio_base_url(
            lmstudio_base_url,
            default=default_lmstudio_base_url(),
        )
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
        self._captioner = LMStudioCaptioner(
            model_name=self._model_name,
            prompt_text=self._caption_prompt,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            base_url=self._lmstudio_base_url,
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
        debug_recorder=None,
        debug_step: str = "caption",
    ) -> CaptionOutput:
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        self._ensure_captioner()
        prompt = self._caption_prompt or _build_local_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            is_cover_page=is_cover_page,
            people_positions=people_positions,
        )
        _emit_prompt_debug(
            debug_recorder,
            step=debug_step,
            engine=self.engine,
            model=self.effective_model_name,
            prompt=prompt,
            system_prompt="",
            source_path=source_path or image_path,
            prompt_source=("custom" if self._caption_prompt else "skill"),
            metadata={
                "is_cover_page": bool(is_cover_page),
                "photo_count": int(photo_count),
            },
        )
        try:
            caption = self._captioner.describe_page(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            return CaptionOutput(
                text=caption.text,
                engine=self.engine,
                ocr_text=str(getattr(caption, "ocr_text", "") or ""),
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
                author_text=str(getattr(caption, "author_text", "") or ""),
                scene_text=str(getattr(caption, "scene_text", "") or ""),
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
        debug_recorder=None,
        debug_step: str = "people_count",
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
        _emit_prompt_debug(
            debug_recorder,
            step=debug_step,
            engine=self.engine,
            model=self.effective_model_name,
            prompt=prompt,
            system_prompt=people_count_system_prompt(),
            source_path=source_path or image_path,
            prompt_source=("custom" if self._caption_prompt else "skill"),
            metadata={},
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
        debug_recorder=None,
        debug_step: str = "location",
    ) -> LocationOutput:
        if self.engine != "lmstudio":
            return LocationOutput(
                engine=self.engine,
                fallback=True,
                error=f"{self.engine.upper()} location estimation is not implemented.",
            )
        self._ensure_captioner()
        prompt = _build_location_prompt(
            is_cover_page=is_cover_page,
        )
        _emit_prompt_debug(
            debug_recorder,
            step=debug_step,
            engine=self.engine,
            model=self.effective_model_name,
            prompt=prompt,
            system_prompt=location_system_prompt(),
            source_path=source_path or image_path,
            prompt_source=("custom" if self._caption_prompt else "skill"),
            metadata={"is_cover_page": bool(is_cover_page)},
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
