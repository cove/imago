from __future__ import annotations

import contextlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ai_caption import CaptionEngine
from .ai_geocode import NominatimGeocoder
from .ai_location import run_locations_step, _resolve_location_payload
from .ai_metadata import MetadataEngine
from .ai_ocr import OCREngine, extract_keywords
from .ai_index_steps import StepRunner
from .image_limits import allow_large_pillow_images
from .prompt_debug import PromptDebugSession
from .xmp_sidecar import _dedupe
from .ai_sidecar_state import _is_derived_image_path

from .ai_album_titles import (
    _require_album_title_for_title_page,
    _resolve_title_page_album_title,
)

AI_MODEL_MAX_SOURCE_BYTES = 30 * 1024 * 1024


@dataclass
class ImageAnalysis:
    image_path: Path
    people_names: list[str]
    object_labels: list[str]
    ocr_text: str
    ocr_keywords: list[str]
    subjects: list[str]
    description: str
    payload: dict[str, Any]
    author_text: str = ""
    scene_text: str = ""
    faces_detected: int = 0
    image_regions: list[dict] = None
    album_title: str = ""
    title: str = ""
    ocr_lang: str = ""
    dc_date: str = ""

    def __post_init__(self):
        if self.image_regions is None:
            self.image_regions = []


@dataclass(frozen=True)
class ArchiveScanOCRAuthority:
    page_key: str
    group_paths: tuple[Path, ...]
    signature: str
    ocr_text: str
    ocr_keywords: tuple[str, ...]
    ocr_hash: str
    stitched_image_path: Path | None = None


def _build_caption_metadata(
    *,
    requested_engine: str,
    effective_engine: str,
    fallback: bool,
    error: str,
    engine_error: str,
    model: str,
    people_present: bool = False,
    estimated_people_count: int = 0,
) -> dict[str, Any]:
    return {
        "requested_engine": str(requested_engine),
        "effective_engine": str(effective_engine),
        "fallback": bool(fallback),
        "error": str(error or "")[:500],
        "engine_error": str(engine_error or ""),
        "model": str(model or ""),
        "people_present": bool(people_present),
        "estimated_people_count": max(0, int(estimated_people_count)),
    }


def _refresh_detection_model_metadata(
    detections: dict[str, Any] | None,
    *,
    ocr_model: str,
    caption_model: str,
) -> dict[str, Any]:
    updated = dict(detections or {})
    ocr_payload = dict(updated.get("ocr") or {})
    ocr_payload["model"] = str(ocr_model or "")
    updated["ocr"] = ocr_payload
    caption_payload = dict(updated.get("caption") or {})
    caption_payload["model"] = str(caption_model or "")
    updated["caption"] = caption_payload
    return updated


def _estimate_people_from_detections(
    *,
    people_matches: list | None = None,
    people_names: list[str] | None = None,
    object_labels: list[str] | None = None,
    faces_detected: int = 0,
) -> tuple[bool, int]:
    object_person_count = sum(1 for label in list(object_labels or []) if str(label).strip().casefold() == "person")
    estimated_people_count = max(
        0,
        int(faces_detected or 0),
        len(list(people_matches or [])),
        len(list(people_names or [])),
        int(object_person_count),
    )
    return estimated_people_count > 0, estimated_people_count


def _caption_people_name_score(text: str, people_names: list[str] | None = None) -> int:
    caption_text = str(text or "").casefold()
    score = 0
    for name in _dedupe([str(item or "").strip() for item in list(people_names or [])]):
        if not name:
            continue
        normalized_name = name.casefold()
        if normalized_name in caption_text:
            score += 2
            continue
        first_token = normalized_name.split()[0]
        if len(first_token) >= 4 and re.search(rf"\b{re.escape(first_token)}\b", caption_text):
            score += 1
    return score


def _merge_people_estimates(
    *,
    local_people_present: bool,
    local_estimated_people_count: int,
    model_people_present: bool,
    model_estimated_people_count: int,
) -> tuple[bool, int]:
    estimated_people_count = max(
        0,
        int(local_estimated_people_count),
        int(model_estimated_people_count),
    )
    people_present = bool(local_people_present or model_people_present or estimated_people_count > 0)
    return people_present, estimated_people_count


def _resolve_people_count_metadata(
    *,
    requested_caption_engine: str,
    caption_engine: Any,
    model_image_path: Path,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: Path,
    album_title: str,
    printed_album_title: str,
    people_positions: dict[str, str],
    local_people_present: bool,
    local_estimated_people_count: int,
    prompt_debug: PromptDebugSession | None = None,
    debug_step: str = "people_count",
) -> tuple[bool, int]:
    if str(requested_caption_engine or "").strip().lower() != "lmstudio":
        return local_people_present, local_estimated_people_count
    estimate_people = getattr(caption_engine, "estimate_people", None)
    if not callable(estimate_people):
        return local_people_present, local_estimated_people_count
    try:
        result = estimate_people(
            image_path=model_image_path,
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
            debug_step=debug_step,
        )
    except Exception:
        return local_people_present, local_estimated_people_count
    fallback = getattr(result, "fallback", False)
    if not isinstance(fallback, bool) or fallback:
        return local_people_present, local_estimated_people_count
    model_people_present = getattr(result, "people_present", False)
    model_estimated_people_count = getattr(result, "estimated_people_count", 0)
    if not isinstance(model_people_present, bool):
        model_people_present = False
    if isinstance(model_estimated_people_count, bool):
        model_estimated_people_count = 0
    try:
        model_estimated_people_count = max(0, int(model_estimated_people_count or 0))
    except Exception:
        model_estimated_people_count = 0
    return _merge_people_estimates(
        local_people_present=local_people_present,
        local_estimated_people_count=local_estimated_people_count,
        model_people_present=model_people_present,
        model_estimated_people_count=model_estimated_people_count,
    )


def _serialize_people_matches(people_matches: list) -> list[dict[str, Any]]:
    return [
        {
            "name": row.name,
            "score": round(row.score, 5),
            "certainty": round(float(getattr(row, "certainty", row.score)), 5),
            "reviewed_by_human": bool(getattr(row, "reviewed_by_human", False)),
            "face_id": str(getattr(row, "face_id", "") or ""),
            **({"bbox": [int(v) for v in row.bbox[:4]]} if getattr(row, "bbox", None) else {}),
        }
        for row in people_matches
    ]


def _merge_people_matches(*match_groups: list) -> list:
    merged: dict[str, Any] = {}
    for group in match_groups:
        for row in list(group or []):
            name = str(getattr(row, "name", "") or "").strip()
            if not name:
                continue
            current = merged.get(name)
            if current is None:
                merged[name] = row
                continue
            row_certainty = float(getattr(row, "certainty", getattr(row, "score", 0.0)) or 0.0)
            current_certainty = float(getattr(current, "certainty", getattr(current, "score", 0.0)) or 0.0)
            row_score = float(getattr(row, "score", 0.0) or 0.0)
            current_score = float(getattr(current, "score", 0.0) or 0.0)
            if row_certainty > current_certainty or (row_certainty == current_certainty and row_score > current_score):
                merged[name] = row
    out = list(merged.values())
    out.sort(
        key=lambda row: (
            -float(getattr(row, "certainty", getattr(row, "score", 0.0)) or 0.0),
            -float(getattr(row, "score", 0.0) or 0.0),
            str(getattr(row, "name", "") or "").casefold(),
        )
    )
    return out


@contextlib.contextmanager
def _prepare_ai_model_image(image_path: Path):
    path = Path(image_path)
    try:
        source_size = int(path.stat().st_size)
    except FileNotFoundError:
        yield path
        return
    if source_size <= AI_MODEL_MAX_SOURCE_BYTES:
        yield path
        return

    try:
        from PIL import Image, ImageOps  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(Image)
    except Exception:
        yield path
        return

    temp_dir = tempfile.TemporaryDirectory(prefix="imago-ai-")
    try:
        out_path = Path(temp_dir.name) / f"{path.stem}_ai.jpg"
        with Image.open(str(path)) as image:
            working = ImageOps.exif_transpose(image)
            if working.mode not in {"RGB", "L"}:
                working = working.convert("RGB")
            width, height = working.size
            scale = min(
                0.95,
                max(
                    0.2,
                    ((AI_MODEL_MAX_SOURCE_BYTES / float(max(1, source_size))) ** 0.5) * 0.92,
                ),
            )
            quality = 90
            candidate = working
            created_candidate = False
            while True:
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                if new_size != candidate.size:
                    if created_candidate:
                        candidate.close()
                    resampling = getattr(getattr(working, "Resampling", None), "LANCZOS", None)
                    if resampling is None:
                        resampling = 1
                    candidate = working.resize(new_size, resampling)
                    created_candidate = True
                save_image = candidate.convert("RGB") if candidate.mode != "RGB" else candidate
                save_image.save(out_path, format="JPEG", quality=quality, optimize=True)
                if save_image is not candidate:
                    save_image.close()
                if int(out_path.stat().st_size) <= AI_MODEL_MAX_SOURCE_BYTES or scale <= 0.25:
                    break
                scale = max(0.25, scale * 0.85)
                quality = max(72, quality - 5)
            if created_candidate:
                candidate.close()
        yield out_path
    finally:
        temp_dir.cleanup()


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image as _PIL_Image  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(_PIL_Image)
        with _PIL_Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _run_image_analysis(
    *,
    image_path: Path,
    people_image_path: Path | None = None,
    people_matcher: Any,
    object_detector: Any,
    ocr_engine: OCREngine,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    ocr_engine_name: str,
    ocr_language: str,
    people_hint_text: str = "",
    people_source_path: Path | None = None,
    people_bbox_offset: tuple[int, int] = (0, 0),
    caption_source_path: Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    geocoder: NominatimGeocoder | None = None,
    step_fn=None,
    extra_people_names: list[str] | None = None,
    is_page_scan: bool = False,
    ocr_text_override: str | None = None,
    context_ocr_text: str = "",
    context_location_hint: str = "",
    prompt_debug: PromptDebugSession | None = None,
    title_page_location: dict[str, str] | None = None,
    step_runner: StepRunner | None = None,
    existing_sidecar_state: dict | None = None,
    metadata_engine: MetadataEngine | None = None,
) -> ImageAnalysis:
    # Deferred imports to avoid circular dependency with ai_index
    from .ai_index import (  # pylint: disable=import-outside-toplevel
        _append_geocode_artifact,
        _apply_title_page_location_config,
        _compute_people_positions,
        _contextualize_ocr_text,
        _format_people_step_label,
        _match_people_with_cast_store_retry,
    )

    del ocr_engine_name
    page_photo_count = 0 if is_page_scan else 1
    people_input_path = people_image_path or image_path
    people_coordinate_path = people_source_path or people_input_path
    existing_detections: dict[str, Any] = dict((existing_sidecar_state or {}).get("detections") or {})
    debug_recorder = prompt_debug.record if prompt_debug is not None else None
    geocode_recorder = lambda record: _append_geocode_artifact(image_path=image_path, record=record)
    caption_step_output: dict[str, Any] = {}
    metadata_output: dict[str, Any] = {}
    _metadata_dc_date = ""
    location_payload: dict[str, Any] = {}
    locations_shown: list = []
    locations_shown_ran = False

    with _prepare_ai_model_image(image_path) as model_image_path:
        object_labels: list[str] = []
        ocr_text = str(ocr_text_override or "").strip()
        clean_context_ocr = str(context_ocr_text or "").strip()
        clean_context_location = str(context_location_hint or "").strip()

        # ── OCR step (legacy path only — metadata_engine handles this in step_runner path) ──
        if ocr_text_override is None and ocr_engine.engine != "none" and metadata_engine is None:
            if step_fn:
                step_fn("ocr")
            if step_runner is not None:
                _ocr_ran = False

                def _do_ocr() -> dict[str, Any]:
                    nonlocal ocr_text, _ocr_ran
                    text = ocr_engine.read_text(
                        model_image_path,
                        source_path=(caption_source_path or people_source_path or image_path),
                        debug_recorder=debug_recorder,
                        debug_step="ocr",
                    )
                    ocr_text = text
                    _ocr_ran = True
                    return {
                        "ocr": {
                            "engine": ocr_engine.engine,
                            "model": str(ocr_engine.effective_model_name),
                            "language": ocr_language,
                            "keywords": [],
                            "chars": len(text),
                        }
                    }

                step_runner.run("ocr", _do_ocr, model=str(ocr_engine.effective_model_name))
                if not _ocr_ran:
                    # Step was fresh — load cached text
                    from .ai_sidecar_state import _effective_sidecar_ocr_text  # pylint: disable=import-outside-toplevel
                    ocr_text = str(_effective_sidecar_ocr_text(image_path, existing_sidecar_state) or "").strip()
            else:
                ocr_text = ocr_engine.read_text(
                    model_image_path,
                    source_path=(caption_source_path or people_source_path or image_path),
                    debug_recorder=debug_recorder,
                    debug_step="ocr",
                )

        # ── People step ───────────────────────────────────────────────────────
        combined_hint_text = " ".join(
            part for part in [str(people_hint_text or "").strip(), ocr_text, clean_context_ocr] if part
        ).strip()
        people_matches: list = []
        _faces_detected: int = 0
        if step_runner is not None:
            _people_ran = False

            def _do_people() -> dict[str, Any]:
                nonlocal people_matches, _faces_detected, _people_ran
                matches = (
                    _match_people_with_cast_store_retry(
                        people_matcher=people_matcher,
                        image_path=people_input_path,
                        source_path=people_coordinate_path,
                        bbox_offset=people_bbox_offset,
                        hint_text=combined_hint_text,
                    )
                    if people_matcher
                    else []
                )
                people_matches = matches
                _faces_detected = (
                    (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                    if people_matcher
                    else 0
                )
                _people_ran = True
                return {"people": _serialize_people_matches(matches)}

            people_output = step_runner.run("people", _do_people)
            if not _people_ran:
                # Step was fresh — reconstruct people info from cache
                cached_rows = [r for r in list(people_output.get("people") or []) if isinstance(r, dict)]
                # Build lightweight proxy objects for position computation
                _CachedMatch = type("_CachedMatch", (), {})
                for row in cached_rows:
                    obj = _CachedMatch()
                    obj.name = str(row.get("name") or "")
                    obj.score = float(row.get("score") or 0.0)
                    obj.bbox = row.get("bbox") or []
                    obj.certainty = float(row.get("certainty") or obj.score)
                    obj.reviewed_by_human = bool(row.get("reviewed_by_human", False))
                    obj.face_id = str(row.get("face_id") or "")
                    people_matches.append(obj)
        else:
            people_matches = (
                _match_people_with_cast_store_retry(
                    people_matcher=people_matcher,
                    image_path=people_input_path,
                    source_path=people_coordinate_path,
                    bbox_offset=people_bbox_offset,
                    hint_text=combined_hint_text,
                )
                if people_matcher
                else []
            )
            _faces_detected = (
                (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                if people_matcher
                else 0
            )

        people_match_names = _dedupe([getattr(row, "name", None) or "" for row in people_matches])
        people_match_names = [n for n in people_match_names if n]
        if step_fn:
            step_fn(_format_people_step_label("people", people_match_names))
        people_names = _dedupe(people_match_names + list(extra_people_names or []))
        people_positions = _compute_people_positions(people_matches, people_coordinate_path)

        # ── Objects step ──────────────────────────────────────────────────────
        if step_fn:
            step_fn("objects")
        object_matches: list = []
        if step_runner is not None:
            _objects_ran = False

            def _do_objects() -> dict[str, Any] | None:
                nonlocal object_matches, _objects_ran
                if object_detector is None:
                    return None
                matches = object_detector.detect_image(model_image_path)
                object_matches = matches
                _objects_ran = True
                return {
                    "objects": [{"label": row.label, "score": round(row.score, 5)} for row in matches],
                    "object_model": str(object_detector.model_name),
                }

            objects_output = step_runner.run(
                "objects",
                _do_objects,
                model=str(object_detector.model_name) if object_detector else "",
            )
            if not _objects_ran:
                cached_object_rows = [r for r in list(objects_output.get("objects") or []) if isinstance(r, dict)]
                # Reconstruct proxy objects for label access
                _CachedObjMatch = type("_CachedObjMatch", (), {})
                for row in cached_object_rows:
                    obj = _CachedObjMatch()
                    obj.label = str(row.get("label") or "")
                    obj.score = float(row.get("score") or 0.0)
                    object_matches.append(obj)
        else:
            object_matches = object_detector.detect_image(model_image_path) if object_detector else []

        object_labels = [getattr(row, "label", None) or row.get("label", "") if isinstance(row, dict) else getattr(row, "label", "") for row in object_matches]
        object_labels = [str(l) for l in object_labels if l]

        # ── Caption step (legacy path only — metadata_engine handles this in step_runner path) ──
        if step_fn and metadata_engine is None:
            step_fn("caption")
        caption_output = None
        description = ""
        author_text = ""
        scene_text = ""
        _caption_ocr_text = ""
        _caption_ocr_lang = ""

        if step_runner is not None and metadata_engine is None:
            _caption_ran = False

            def _do_caption() -> dict[str, Any]:
                nonlocal caption_output, _caption_ran
                output = caption_engine.generate(
                    image_path=model_image_path,
                    people=people_names,
                    objects=object_labels,
                    ocr_text=ocr_text,
                    source_path=caption_source_path or people_source_path or image_path,
                    album_title=album_title,
                    printed_album_title=printed_album_title,
                    photo_count=page_photo_count,
                    people_positions=people_positions,
                    context_ocr_text=clean_context_ocr,
                    debug_recorder=debug_recorder,
                    debug_step="caption",
                )
                caption_output = output
                _caption_ran = True
                faces = (
                    (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                    if people_matcher
                    else 0
                )
                (local_pp, local_epc) = _estimate_people_from_detections(
                    people_matches=people_matches,
                    people_names=people_names,
                    object_labels=object_labels,
                    faces_detected=faces,
                )
                pp, epc = _resolve_people_count_metadata(
                    requested_caption_engine=requested_caption_engine,
                    caption_engine=caption_engine,
                    model_image_path=model_image_path,
                    people=people_names,
                    objects=object_labels,
                    ocr_text=ocr_text,
                    source_path=caption_source_path or people_source_path or image_path,
                    album_title=album_title,
                    printed_album_title=printed_album_title,
                    people_positions=people_positions,
                    local_people_present=local_pp,
                    local_estimated_people_count=local_epc,
                    prompt_debug=prompt_debug,
                    debug_step="people_count",
                )
                return {
                    "caption": _build_caption_metadata(
                        requested_engine=requested_caption_engine,
                        effective_engine=str(output.engine),
                        fallback=bool(output.fallback),
                        error=str(output.error or ""),
                        engine_error=str(getattr(output, "engine_error", "") or ""),
                        model=str(caption_engine.effective_model_name),
                        people_present=pp,
                        estimated_people_count=epc,
                    ),
                }

            caption_step_output = step_runner.run("caption", _do_caption, model=str(caption_engine.effective_model_name))
            if _caption_ran and caption_output is not None:
                description = str(caption_output.text or "")
                author_text = str(getattr(caption_output, "author_text", "") or "")
                scene_text = str(getattr(caption_output, "scene_text", "") or "")
                _caption_ocr_text = str(getattr(caption_output, "ocr_text", "") or "")
                _caption_ocr_lang = str(getattr(caption_output, "ocr_lang", "") or "")
                _faces_detected = (
                    (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                    if people_matcher
                    else 0
                )
            else:
                # Caption was fresh — load cached description from existing sidecar
                description = str((existing_sidecar_state or {}).get("description") or "")
                author_text = str((existing_sidecar_state or {}).get("author_text") or "")
                scene_text = str((existing_sidecar_state or {}).get("scene_text") or "")
                _caption_ocr_lang = str((existing_detections.get("ocr") or {}).get("language") or "")
        else:
            caption_output = caption_engine.generate(
                image_path=model_image_path,
                people=people_names,
                objects=object_labels,
                ocr_text=ocr_text,
                source_path=caption_source_path or people_source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                photo_count=page_photo_count,
                people_positions=people_positions,
                context_ocr_text=clean_context_ocr,
                debug_recorder=debug_recorder,
                debug_step="caption",
            )
            _faces_detected = (
                (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                if people_matcher
                else 0
            )
            description = str(caption_output.text or "")
            author_text = str(getattr(caption_output, "author_text", "") or "")
            scene_text = str(getattr(caption_output, "scene_text", "") or "")
            _caption_ocr_text = str(getattr(caption_output, "ocr_text", "") or "")
            _caption_ocr_lang = str(getattr(caption_output, "ocr_lang", "") or "")

    # ── Metadata step (replaces OCR + caption + locations in step_runner path) ──
    if metadata_engine is not None and step_runner is not None:
        if step_fn:
            step_fn("metadata")
        _metadata_ran = False

        def _do_metadata() -> dict[str, Any] | None:
            nonlocal ocr_text, description, author_text, scene_text, _metadata_dc_date, _metadata_ran, location_payload, locations_shown, locations_shown_ran
            if _is_derived_image_path(image_path):
                return None
            try:
                from .xmp_sidecar import read_region_list  # pylint: disable=import-outside-toplevel
                _sidecar = image_path.with_suffix(".xmp")
                if is_page_scan and _sidecar.is_file():
                    from PIL import Image as _PILImage  # pylint: disable=import-outside-toplevel
                    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel
                    allow_large_pillow_images(_PILImage)
                    with _PILImage.open(str(image_path)) as _img:
                        _w, _h = _img.size
                    _regions = read_region_list(_sidecar, _w, _h)
                    n_photos = max(1, len(_regions))
                else:
                    n_photos = 1
            except Exception:
                n_photos = 1
            result = metadata_engine.analyze(
                image_path,
                n_photos=n_photos,
                album_title=album_title,
                source_path=caption_source_path or people_source_path or image_path,
                debug_recorder=debug_recorder,
                debug_step="metadata",
            )
            _metadata_ran = True
            all_captions = [p.caption for p in result.photos.values() if p.caption]
            all_scene_ocr = [p.scene_ocr for p in result.photos.values() if p.scene_ocr]
            description = " ".join(all_captions)
            author_text = description
            scene_text = "\n".join(all_scene_ocr)
            ocr_text = "\n".join(filter(None, [author_text, scene_text]))
            for photo in result.photos.values():
                if photo.est_date:
                    _metadata_dc_date = photo.est_date
                    break
            primary_location = next((p.location for p in result.photos.values() if p.location), "")
            loc_payload: dict[str, Any] = {}
            if primary_location:
                loc_payload = _resolve_location_payload(
                    geocoder=geocoder,
                    gps_latitude="",
                    gps_longitude="",
                    location_name=primary_location,
                    artifact_recorder=geocode_recorder,
                    artifact_step="location",
                )
            location_payload = loc_payload
            locs_shown = [{"name": p.location_name} for p in result.photos.values() if p.location_name]
            locations_shown = locs_shown
            locations_shown_ran = True
            ocr_kw = extract_keywords(ocr_text, max_keywords=15)
            return {
                "ocr": {
                    "engine": metadata_engine.engine,
                    "model": str(metadata_engine.effective_model_name),
                    "language": "",
                    "keywords": ocr_kw,
                    "chars": len(ocr_text),
                },
                "caption": {
                    "effective_engine": metadata_engine.engine,
                    "fallback": bool(result.fallback),
                    "error": str(result.error or ""),
                    "model": str(metadata_engine.effective_model_name),
                    "people_present": bool(result.people_count > 0),
                    "estimated_people_count": result.people_count,
                },
                "location": location_payload,
                "locations_shown": locations_shown,
                "location_shown_ran": locations_shown_ran,
            }

        metadata_output = step_runner.run("metadata", _do_metadata, model=str(metadata_engine.effective_model_name))
        if not _metadata_ran:
            description = str((existing_sidecar_state or {}).get("description") or "")
            author_text = str((existing_sidecar_state or {}).get("author_text") or "")
            scene_text = str((existing_sidecar_state or {}).get("scene_text") or "")
            from .ai_sidecar_state import _effective_sidecar_ocr_text  # pylint: disable=import-outside-toplevel
            ocr_text = str(_effective_sidecar_ocr_text(image_path, existing_sidecar_state) or "").strip()
            location_payload = dict(metadata_output.get("location") or {})
            locations_shown = list(metadata_output.get("locations_shown") or [])
            locations_shown_ran = bool(metadata_output.get("location_shown_ran", False))
        location_payload, _ = _apply_title_page_location_config(
            image_path=image_path,
            location_payload=location_payload,
            title_page_location=title_page_location,
        )

    # ── Post model-image-context ──────────────────────────────────────────────
    if ocr_text_override is None and not ocr_text and metadata_engine is None:
        ocr_text = _caption_ocr_text
    ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
    subjects = _dedupe(object_labels + ocr_keywords)

    # Build payload (step_runner path uses per-step output dicts; legacy path builds inline)
    if step_runner is not None:
        ocr_meta = dict(step_runner.existing_detections.get("ocr") or {})
        if step_runner.reran.get("metadata"):
            ocr_meta = dict(metadata_output.get("ocr") or {})
        elif step_runner.reran.get("ocr"):
            # Fresh OCR just ran (legacy path) — rebuild with actual values
            ocr_meta = {
                "engine": ocr_engine.engine,
                "model": str(ocr_engine.effective_model_name),
                "language": ocr_language,
                "keywords": ocr_keywords,
                "chars": len(ocr_text),
            }
        else:
            # Keep cached ocr metadata but update keywords from potentially-loaded text
            if ocr_meta:
                ocr_meta = dict(ocr_meta)
                ocr_meta["keywords"] = ocr_keywords
                ocr_meta["chars"] = len(ocr_text)

        if step_runner.reran.get("metadata"):
            caption_meta = dict(metadata_output.get("caption") or {})
        else:
            caption_meta = dict(
                (caption_step_output if step_runner.reran.get("caption") else existing_detections).get("caption") or {}
            )
        objects_meta = list(
            (objects_output if step_runner.reran.get("objects") else existing_detections).get("objects") or []
        )
        object_model_val = str(
            (objects_output if step_runner.reran.get("objects") else existing_detections).get("object_model") or ""
        )

        payload: dict[str, Any] = {
            "people": _serialize_people_matches(people_matches) if step_runner.reran.get("people") else list(existing_detections.get("people") or []),
            "objects": objects_meta,
            "ocr": ocr_meta,
            "caption": caption_meta,
        }
        if object_model_val:
            payload["object_model"] = object_model_val
    else:
        (
            local_people_present,
            local_estimated_people_count,
        ) = _estimate_people_from_detections(
            people_matches=people_matches,
            people_names=people_names,
            object_labels=object_labels,
            faces_detected=_faces_detected,
        )
        people_present, estimated_people_count = _resolve_people_count_metadata(
            requested_caption_engine=requested_caption_engine,
            caption_engine=caption_engine,
            model_image_path=model_image_path,
            people=people_names,
            objects=object_labels,
            ocr_text=ocr_text,
            source_path=caption_source_path or people_source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            people_positions=people_positions,
            local_people_present=local_people_present,
            local_estimated_people_count=local_estimated_people_count,
            prompt_debug=prompt_debug,
            debug_step="people_count",
        )
        payload = {
            "people": _serialize_people_matches(people_matches),
            "objects": [{"label": getattr(row, "label", ""), "score": round(getattr(row, "score", 0.0), 5)} for row in object_matches],
            "ocr": {
                "engine": str(caption_output.engine),
                "model": str(caption_engine.effective_model_name),
                "language": str(_caption_ocr_lang or ocr_language),
                "keywords": ocr_keywords,
                "chars": len(ocr_text),
            },
            "caption": _build_caption_metadata(
                requested_engine=requested_caption_engine,
                effective_engine=str(caption_output.engine),
                fallback=bool(caption_output.fallback),
                error=str(caption_output.error or ""),
                engine_error=str(getattr(caption_output, "engine_error", "") or ""),
                model=str(caption_engine.effective_model_name),
                people_present=people_present,
                estimated_people_count=estimated_people_count,
            ),
        }
        if object_detector is not None:
            payload["object_model"] = str(object_detector.model_name)

    # ── Location step (legacy caption_engine path) or via metadata step ───────
    if step_runner is not None and metadata_engine is None:
        if step_fn:
            step_fn("locations")
        caption_text_for_locations = description
        context_ocr_for_locations = _contextualize_ocr_text(
            ocr_text,
            context_ocr_text=clean_context_ocr,
            context_location_hint=clean_context_location,
        )

        def _do_locations() -> dict[str, Any] | None:
            if _is_derived_image_path(image_path):
                return None
            return run_locations_step(
                caption_engine=caption_engine,
                image_path=image_path,
                caption_text=caption_text_for_locations,
                ocr_text=context_ocr_for_locations,
                source_path=caption_source_path or people_source_path or image_path,
                album_title=album_title,
                printed_album_title=printed_album_title,
                geocoder=geocoder,
                prompt_debug=prompt_debug,
                artifact_recorder=geocode_recorder,
            )

        locations_output = step_runner.run("locations", _do_locations, model=str(caption_engine.effective_model_name))
        location_payload = dict(locations_output.get("location") or {})
        locations_shown = list(locations_output.get("locations_shown") or [])
        locations_shown_ran = bool(locations_output.get("location_shown_ran", False))

        location_payload, _ = _apply_title_page_location_config(
            image_path=image_path,
            location_payload=location_payload,
            title_page_location=title_page_location,
        )

    if step_runner is not None:
        if location_payload:
            payload["location"] = location_payload
        if step_fn and metadata_engine is None:
            step_fn("locations_shown")
        payload["locations_shown"] = locations_shown
        payload["location_shown_ran"] = locations_shown_ran
    else:
        # Legacy path (step_runner=None): locations step not run; emit labels only
        if step_fn:
            step_fn("locations_shown")
        payload["locations_shown"] = []
        payload["location_shown_ran"] = False

    resolved_album_title = _resolve_title_page_album_title(
        image_path=image_path,
        album_title=str(getattr(caption_output, "album_title", "") if caption_output is not None else ""),
        ocr_text=ocr_text,
    )
    resolved_album_title = _require_album_title_for_title_page(
        image_path=image_path,
        album_title=(resolved_album_title or album_title),
        context="analysis",
    )
    return ImageAnalysis(
        image_path=image_path,
        people_names=people_names,
        object_labels=object_labels,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        subjects=subjects,
        description=description,
        author_text=author_text,
        scene_text=scene_text,
        payload=payload,
        faces_detected=_faces_detected,
        image_regions=list(getattr(caption_output, "image_regions", None) or []) if caption_output is not None else [],
        album_title=resolved_album_title,
        title=str(getattr(caption_output, "title", "") if caption_output is not None else ""),
        ocr_lang=str(_caption_ocr_lang or ocr_language),
        dc_date=_metadata_dc_date,
    )
