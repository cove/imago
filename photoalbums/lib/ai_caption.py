from __future__ import annotations

import urllib.request  # noqa: F401 — kept at module level for test patching via ai_caption.urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ai_model_settings import default_caption_model, default_caption_models, default_lmstudio_base_url
from ._caption_text import clean_text, clean_lines, dedupe, join_human  # noqa: F401
from ._lmstudio_helpers import emit_prompt_debug as _emit_prompt_debug
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
    location_shown_system_prompt,
    normalize_lmstudio_base_url,
    people_count_system_prompt,
)
from ._caption_prompts import (  # noqa: F401
    _build_location_prompt,
    _build_people_count_prompt,
    _build_local_prompt,
)


def _caption_has_meaningful_content(caption) -> bool:
    for field_name in (
        "text",
        "ocr_text",
        "author_text",
        "scene_text",
        "album_title",
        "title",
        "location_name",
        "gps_latitude",
        "gps_longitude",
    ):
        if clean_text(str(getattr(caption, field_name, "") or "")):
            return True
    return False


def resolve_caption_model(engine: str, model_name: str) -> str:
    models = resolve_caption_models(engine, model_name)
    return models[0] if models else ""


def resolve_caption_models(engine: str, model_name: str) -> list[str]:
    normalized = str(engine or "").strip().lower()
    if normalized in {"qwen", "blip", "local"}:
        normalized = "lmstudio"
    text = str(model_name or "").strip()
    if text:
        return [text]
    configured = default_caption_models()
    if configured:
        return configured
    fallback = default_caption_model()
    return [fallback] if fallback else []


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
    album_title: str = ""
    title: str = ""
    author_text: str = ""
    scene_text: str = ""
    engine_error: str = ""


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


@dataclass
class LocationsShownOutput:
    engine: str
    locations_shown: list[dict] | None = None
    fallback: bool = False
    error: str = ""


@dataclass
class LocationQueryResult:
    engine: str
    primary_query: str = ""
    named_queries: list[dict[str, str]] = None
    fallback: bool = False
    error: str = ""

    def __post_init__(self):
        if self.named_queries is None:
            self.named_queries = []


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
        self._model_names = resolve_caption_models(normalized, model_name)
        self._model_name = self._model_names[0] if self._model_names else ""
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
            model_name=self._model_names,
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
        people_positions: dict[str, str] | None = None,
        context_ocr_text: str = "",
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
            people_positions=people_positions,
            context_ocr_text=context_ocr_text,
        )
        response = ""
        finish_reason = ""
        error_text = ""
        try:
            caption = self._captioner.describe_page(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            has_content = _caption_has_meaningful_content(caption)
            result = CaptionOutput(
                text=caption.text,
                engine=self.engine,
                ocr_text=str(getattr(caption, "ocr_text", "") or ""),
                gps_latitude=caption.gps_latitude,
                gps_longitude=caption.gps_longitude,
                location_name=caption.location_name,
                people_present=bool(getattr(caption, "people_present", False)),
                estimated_people_count=max(0, int(getattr(caption, "estimated_people_count", 0) or 0)),
                fallback=not has_content,
                error=("" if has_content else f"{self.engine.upper()} returned empty output."),
                album_title=str(getattr(caption, "album_title", "") or ""),
                title=str(getattr(caption, "title", "") or ""),
                author_text=str(getattr(caption, "author_text", "") or ""),
                scene_text=str(getattr(caption, "scene_text", "") or ""),
                engine_error="",
            )
            return result
        except Exception as exc:
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            error_text = str(exc)
            return CaptionOutput(
                text="",
                engine=self.engine,
                fallback=True,
                error=error_text,
                engine_error=error_text,
            )
        finally:
            metadata = {"photo_count": int(photo_count)}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt,
                system_prompt="",
                source_path=source_path or image_path,
                prompt_source=("custom" if self._caption_prompt else "skill"),
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )

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
        response = ""
        finish_reason = ""
        error_text = ""
        try:
            people_count = self._captioner.estimate_people(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            result = PeopleCountOutput(
                engine=self.engine,
                people_present=bool(people_count.people_present),
                estimated_people_count=max(0, int(getattr(people_count, "estimated_people_count", 0) or 0)),
                fallback=False,
            )
            return result
        except Exception as exc:
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            error_text = str(exc)
            return PeopleCountOutput(
                engine=self.engine,
                fallback=True,
                error=error_text,
            )
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt,
                system_prompt=people_count_system_prompt(),
                source_path=source_path or image_path,
                prompt_source=("custom" if self._caption_prompt else "skill"),
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
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
            ocr_text=ocr_text,
            album_title=album_title,
            printed_album_title=printed_album_title,
        )
        response = ""
        finish_reason = ""
        error_text = ""
        try:
            location = self._captioner.estimate_location(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            result = LocationOutput(
                engine=self.engine,
                gps_latitude=str(getattr(location, "gps_latitude", "") or "").strip(),
                gps_longitude=str(getattr(location, "gps_longitude", "") or "").strip(),
                location_name=str(getattr(location, "location_name", "") or "").strip(),
                fallback=False,
            )
            return result
        except Exception as exc:
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            error_text = str(exc)
            return LocationOutput(
                engine=self.engine,
                fallback=True,
                error=error_text,
            )
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt,
                system_prompt=location_system_prompt(),
                source_path=source_path or image_path,
                prompt_source=("custom" if self._caption_prompt else "skill"),
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )

    def estimate_locations_shown(
        self,
        image_path: str | Path,
        *,
        ocr_text: str,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        debug_recorder=None,
        debug_step: str = "locations_shown",
    ) -> LocationsShownOutput:
        if self.engine != "lmstudio":
            return LocationsShownOutput(
                engine=self.engine,
                fallback=True,
                error=f"{self.engine.upper()} locations shown estimation is not implemented.",
            )
        self._ensure_captioner()
        from ._caption_prompts import _build_location_shown_prompt

        prompt = _build_location_shown_prompt(
            ocr_text=ocr_text,
            album_title=album_title,
            printed_album_title=printed_album_title,
        )
        response = ""
        finish_reason = ""
        error_text = ""
        try:
            locations_shown = self._captioner.estimate_locations_shown(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            result = LocationsShownOutput(
                engine=self.engine,
                locations_shown=getattr(locations_shown, "locations_shown", None),
                fallback=False,
            )
            return result
        except Exception as exc:
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            error_text = str(exc)
            return LocationsShownOutput(
                engine=self.engine,
                fallback=True,
                error=error_text,
            )
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt,
                system_prompt=location_system_prompt(),
                source_path=source_path or image_path,
                prompt_source=("custom" if self._caption_prompt else "skill"),
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )

    def generate_location_queries(
        self,
        image_path: str | Path,
        *,
        caption_text: str,
        ocr_text: str = "",
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        debug_recorder=None,
        debug_step: str = "location_queries",
    ) -> LocationQueryResult:
        if self.engine != "lmstudio":
            return LocationQueryResult(
                engine=self.engine,
                fallback=True,
                error=f"{self.engine.upper()} location queries not implemented.",
            )
        self._ensure_captioner()
        from ._caption_prompts import _build_location_queries_prompt

        prompt = _build_location_queries_prompt(
            caption_text=caption_text,
            ocr_text=ocr_text,
            album_title=album_title,
            printed_album_title=printed_album_title,
        )
        response = ""
        finish_reason = ""
        error_text = ""
        try:
            details = self._captioner.generate_location_queries(  # type: ignore[attr-defined]
                image_path=image_path,
                prompt=prompt,
            )
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            named = []
            for entry in list(details.locations_shown or []):
                if not isinstance(entry, dict):
                    continue
                if not str(entry.get("name") or "").strip():
                    continue
                named.append(
                    {
                        "name": str(entry.get("name") or "").strip(),
                        "world_region": str(entry.get("world_region") or "").strip(),
                        "country_name": str(entry.get("country_name") or "").strip(),
                        "country_code": str(entry.get("country_code") or "").strip(),
                        "province_or_state": str(entry.get("province_or_state") or "").strip(),
                        "city": str(entry.get("city") or "").strip(),
                        "sublocation": str(entry.get("sublocation") or "").strip(),
                    }
                )
            return LocationQueryResult(
                engine=self.engine,
                primary_query=str(details.location_name or "").strip(),
                named_queries=[q for q in named if q],
            )
        except Exception as exc:
            response = str(getattr(self._captioner, "last_response_text", "") or "")
            finish_reason = str(getattr(self._captioner, "last_finish_reason", "") or "")
            error_text = str(exc)
            return LocationQueryResult(
                engine=self.engine,
                fallback=True,
                error=error_text,
            )
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt,
                system_prompt=location_system_prompt(),
                source_path=source_path or image_path,
                prompt_source=("custom" if self._caption_prompt else "skill"),
                response=response,
                finish_reason=finish_reason,
                metadata=metadata,
            )
