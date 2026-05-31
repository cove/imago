from __future__ import annotations

import contextlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ai_album_titles import (
    _is_album_title_source_candidate,
    _require_album_title_for_title_page,
    _resolve_title_page_album_title,
)
from .ai_caption import CaptionEngine
from .ai_geocode import NominatimGeocoder
from .ai_index_steps import StepRunner
from .ai_location import _resolve_location_payload, run_locations_step
from .ai_metadata import MetadataEngine, MetadataResult
from .ai_ocr import OCREngine, extract_keywords
from .ai_sidecar_state import _is_derived_image_path
from .image_limits import allow_large_pillow_images
from .prompt_debug import PromptDebugSession
from .xmp_sidecar import _dedupe

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
        _write_resized_ai_model_image(Image, ImageOps, path, out_path, source_size)
        yield out_path
    finally:
        temp_dir.cleanup()


def _write_resized_ai_model_image(Image, ImageOps, image_path: Path, out_path: Path, source_size: int) -> None:
    with Image.open(str(image_path)) as image:
        working = ImageOps.exif_transpose(image)
        if working.mode not in {"RGB", "L"}:
            working = working.convert("RGB")
        width, height = working.size
        scale = min(0.95, max(0.2, ((AI_MODEL_MAX_SOURCE_BYTES / float(max(1, source_size))) ** 0.5) * 0.92))
        quality = 90
        candidate = working
        created_candidate = False
        while True:
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            if new_size != candidate.size:
                if created_candidate:
                    candidate.close()
                resampling = getattr(getattr(working, "Resampling", None), "LANCZOS", None)
                candidate = working.resize(new_size, resampling if resampling is not None else 1)
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


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image as _PIL_Image  # pylint: disable=import-outside-toplevel

        allow_large_pillow_images(_PIL_Image)
        with _PIL_Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _build_region_with_caption(
    r: dict,
    photo_captions: dict[int, str],
    photo_numbers: dict[int, int],
    photo_locations: dict[int, str] | None = None,
    photo_location_names: dict[int, str] | None = None,
    photo_est_dates: dict[int, str] | None = None,
) -> tuple[Any, bool]:
    from .ai_view_regions import RegionResult, RegionWithCaption  # pylint: disable=import-outside-toplevel

    i = r["index"]
    existing_hint = str(r.get("caption_hint") or "").strip()
    new_hint = photo_captions.get(i, existing_hint)
    new_pn = photo_numbers.get(i, 0) or int(r.get("photo_number") or 0)
    new_photo_location = photo_locations.get(i) if photo_locations is not None and i in photo_locations else None
    new_photo_location_name = photo_location_names.get(i) if photo_location_names is not None and i in photo_location_names else None
    new_photo_est_date = photo_est_dates.get(i) if photo_est_dates is not None and i in photo_est_dates else None
    changed = (
        new_hint != existing_hint
        or new_pn != int(r.get("photo_number") or 0)
        or (new_photo_location is not None and new_photo_location != r.get("photo_location"))
        or (new_photo_est_date is not None and new_photo_est_date != r.get("photo_est_date"))
    )
    region_obj = RegionResult(
        index=i,
        x=r["x"],
        y=r["y"],
        width=r["width"],
        height=r["height"],
        caption_hint=new_hint,
        location_payload=dict(r.get("location_payload") or {}),
        person_names=list(r.get("person_names") or []),
        photo_number=new_pn,
        photo_location=new_photo_location,
        photo_location_name=new_photo_location_name,
        photo_est_date=new_photo_est_date,
    )
    return RegionWithCaption(region=region_obj, caption=new_hint), changed


def _update_region_captions_from_metadata(image_path: Path, photo_captions_list: list[dict]) -> None:
    """Apply per-photo captions from metadata step output to region caption_hints in the XMP.

    photo_captions_list is the ``photo_captions`` key from the metadata step output:
    [{"photo_number": int, "caption": str}, ...]

    Safe to call whether the step ran fresh or was loaded from cache.
    """
    from .xmp_sidecar import read_region_list, write_region_list  # pylint: disable=import-outside-toplevel

    if not photo_captions_list:
        return
    xmp_path = image_path.with_suffix(".xmp")
    if not xmp_path.is_file():
        return
    img_w, img_h = _get_image_dimensions(image_path)
    if img_w <= 0 or img_h <= 0:
        return
    regions = read_region_list(xmp_path, img_w, img_h)
    if not regions:
        return

    photo_captions, photo_numbers, photo_locations, photo_location_names, photo_est_dates = _metadata_region_caption_maps(photo_captions_list)

    if not photo_captions and not photo_numbers:
        return

    updated = False
    rwcs = []
    for r in regions:
        rwc, changed = _build_region_with_caption(
            r,
            photo_captions,
            photo_numbers,
            photo_locations=photo_locations,
            photo_location_names=photo_location_names,
            photo_est_dates=photo_est_dates,
        )
        if changed:
            updated = True
        rwcs.append(rwc)

    if updated:
        write_region_list(xmp_path, rwcs, img_w, img_h)


def _metadata_region_caption_maps(
    photo_captions_list: list[dict],
) -> tuple[dict[int, str], dict[int, int], dict[int, str], dict[int, str], dict[int, str]]:
    photo_captions: dict[int, str] = {}
    photo_numbers: dict[int, int] = {}
    photo_locations: dict[int, str] = {}
    photo_location_names: dict[int, str] = {}
    photo_est_dates: dict[int, str] = {}
    for entry in photo_captions_list:
        pn = int(entry.get("photo_number") or 0)
        if pn > 0:
            region_idx = pn - 1
            photo_numbers[region_idx] = pn
            photo_captions[region_idx] = _metadata_region_caption(entry)
            photo_locations[region_idx] = str(entry.get("location") or "").strip()
            photo_location_names[region_idx] = str(entry.get("location_name") or "").strip()
            photo_est_dates[region_idx] = str(entry.get("est_date") or "").strip()
    return photo_captions, photo_numbers, photo_locations, photo_location_names, photo_est_dates


def _metadata_region_caption(photo: Any) -> str:
    if isinstance(photo, dict):
        caption = str(photo.get("caption") or "").strip()
    else:
        caption = str(getattr(photo, "caption", "") or "").strip()
    return caption


def _metadata_corrected_caption_subjects(photos: list[Any]) -> list[str]:
    subjects: list[str] = []
    for photo in photos:
        if isinstance(photo, dict):
            caption = str(photo.get("caption") or "").strip()
            corrected = str(photo.get("corrected_caption") or "").strip()
        else:
            caption = str(getattr(photo, "caption", "") or "").strip()
            corrected = str(getattr(photo, "corrected_caption", "") or "").strip()
        if corrected and corrected.casefold() != caption.casefold():
            subjects.append(corrected)
    return subjects


def _metadata_photo_fields_dict(photo: dict) -> tuple[str, str, dict[str, Any]]:
    caption = str(photo.get("caption") or "").strip()
    corrected = str(photo.get("corrected_caption") or "").strip()
    payload: dict[str, Any] = {
        "photo_number": int(photo.get("photo_number") or 0),
        "location": str(photo.get("location") or ""),
        "location_name": str(photo.get("location_name") or ""),
        "est_date": str(photo.get("est_date") or ""),
        "scene_ocr": str(photo.get("scene_ocr") or ""),
        "caption": _metadata_region_caption(photo),
        "corrected_caption": corrected,
        "people_count": int(photo.get("people_count") or 0),
    }
    return caption, corrected, payload


def _metadata_photo_fields_obj(photo: Any) -> tuple[str, str, dict[str, Any]]:
    caption = str(getattr(photo, "caption", "") or "").strip()
    corrected = str(getattr(photo, "corrected_caption", "") or "").strip()
    payload: dict[str, Any] = {
        "photo_number": int(getattr(photo, "photo_number", 0) or 0),
        "location": str(getattr(photo, "location", "") or ""),
        "location_name": str(getattr(photo, "location_name", "") or ""),
        "est_date": str(getattr(photo, "est_date", "") or ""),
        "scene_ocr": str(getattr(photo, "scene_ocr", "") or ""),
        "caption": _metadata_region_caption(photo),
        "corrected_caption": corrected,
        "people_count": int(getattr(photo, "people_count", 0) or 0),
    }
    return caption, corrected, payload


def _metadata_photo_payload(photo: Any) -> dict[str, Any]:
    if isinstance(photo, dict):
        caption, corrected, payload = _metadata_photo_fields_dict(photo)
    else:
        caption, corrected, payload = _metadata_photo_fields_obj(photo)
    if corrected and corrected != caption:
        payload["OriginalCaption"] = caption
    return payload


def _last_faces_detected(people_matcher: Any) -> int:
    return (
        (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0) if people_matcher else 0
    )


def _cached_people_matches(rows: list[dict]) -> list:
    matches: list = []
    _CachedMatch = type("_CachedMatch", (), {})
    for row in rows:
        obj = _CachedMatch()
        obj.name = str(row.get("name") or "")
        obj.score = float(row.get("score") or 0.0)
        obj.bbox = row.get("bbox") or []
        obj.certainty = float(row.get("certainty") or obj.score)
        obj.reviewed_by_human = bool(row.get("reviewed_by_human", False))
        obj.face_id = str(row.get("face_id") or "")
        matches.append(obj)
    return matches


def _run_ocr_analysis_step(
    *,
    image_path: Path,
    model_image_path: Path,
    ocr_text: str,
    ocr_text_override: str | None,
    ocr_engine: OCREngine,
    ocr_language: str,
    caption_source_path: Path | None,
    people_source_path: Path | None,
    debug_recorder,
    step_fn,
    step_runner: StepRunner | None,
    existing_sidecar_state: dict | None,
    metadata_engine: MetadataEngine | None,
) -> str:
    if ocr_text_override is not None or ocr_engine.engine == "none" or metadata_engine is not None:
        return ocr_text
    if step_fn:
        step_fn("ocr")
    if step_runner is None:
        return ocr_engine.read_text(
            model_image_path,
            source_path=(caption_source_path or people_source_path or image_path),
            debug_recorder=debug_recorder,
            debug_step="ocr",
        )

    state = {"text": ocr_text, "ran": False}

    def _do_ocr() -> dict[str, Any]:
        text = ocr_engine.read_text(
            model_image_path,
            source_path=(caption_source_path or people_source_path or image_path),
            debug_recorder=debug_recorder,
            debug_step="ocr",
        )
        state["text"] = text
        state["ran"] = True
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
    if not state["ran"]:
        from .ai_sidecar_state import _effective_sidecar_ocr_text  # pylint: disable=import-outside-toplevel

        state["text"] = str(_effective_sidecar_ocr_text(image_path, existing_sidecar_state) or "").strip()
    return str(state["text"])


def _run_people_analysis_step(
    *,
    people_matcher: Any,
    people_input_path: Path,
    people_coordinate_path: Path,
    people_bbox_offset: tuple[int, int],
    combined_hint_text: str,
    step_runner: StepRunner | None,
    match_people,
) -> tuple[list, int]:
    def _match() -> list:
        return (
            match_people(
                people_matcher=people_matcher,
                image_path=people_input_path,
                source_path=people_coordinate_path,
                bbox_offset=people_bbox_offset,
                hint_text=combined_hint_text,
            )
            if people_matcher
            else []
        )

    if step_runner is None:
        matches = _match()
        return matches, _last_faces_detected(people_matcher)

    state = {"matches": [], "faces": 0, "ran": False}

    def _do_people() -> dict[str, Any]:
        matches = _match()
        state["matches"] = matches
        state["faces"] = _last_faces_detected(people_matcher)
        state["ran"] = True
        return {"people": _serialize_people_matches(matches)}

    people_output = step_runner.run("people", _do_people)
    if state["ran"]:
        return list(state["matches"]), int(state["faces"])
    cached_rows = [r for r in list(people_output.get("people") or []) if isinstance(r, dict)]
    return _cached_people_matches(cached_rows), 0


def _cached_object_matches(rows: list[dict]) -> list:
    matches: list = []
    _CachedObjMatch = type("_CachedObjMatch", (), {})
    for row in rows:
        obj = _CachedObjMatch()
        obj.label = str(row.get("label") or "")
        obj.score = float(row.get("score") or 0.0)
        matches.append(obj)
    return matches


def _run_objects_analysis_step(
    *,
    object_detector: Any,
    model_image_path: Path,
    step_runner: StepRunner | None,
) -> tuple[list, dict[str, Any]]:
    if step_runner is None:
        return object_detector.detect_image(model_image_path) if object_detector else [], {}

    state = {"matches": [], "ran": False}

    def _do_objects() -> dict[str, Any] | None:
        if object_detector is None:
            return None
        matches = object_detector.detect_image(model_image_path)
        state["matches"] = matches
        state["ran"] = True
        return {
            "objects": [{"label": row.label, "score": round(row.score, 5)} for row in matches],
            "object_model": str(object_detector.model_name),
        }

    objects_output = step_runner.run(
        "objects",
        _do_objects,
        model=str(object_detector.model_name) if object_detector else "",
    )
    if state["ran"]:
        return list(state["matches"]), dict(objects_output or {})
    cached_rows = [r for r in list(objects_output.get("objects") or []) if isinstance(r, dict)]
    return _cached_object_matches(cached_rows), dict(objects_output or {})


def _object_labels(object_matches: list) -> list[str]:
    labels = [
        getattr(row, "label", None) or row.get("label", "") if isinstance(row, dict) else getattr(row, "label", "")
        for row in object_matches
    ]
    return [str(label) for label in labels if label]


def _caption_text_fields(caption_output: Any) -> dict[str, str]:
    return {
        "description": str(caption_output.text or ""),
        "author_text": str(getattr(caption_output, "author_text", "") or ""),
        "scene_text": str(getattr(caption_output, "scene_text", "") or ""),
        "caption_ocr_text": str(getattr(caption_output, "ocr_text", "") or ""),
        "caption_ocr_lang": str(getattr(caption_output, "ocr_lang", "") or ""),
    }


def _caption_step_cached_result(
    empty: dict[str, Any],
    caption_step_output: Any,
    existing_sidecar_state: dict | None,
    existing_detections: dict[str, Any],
) -> dict[str, Any]:
    return {
        **empty,
        "caption_step_output": dict(caption_step_output or {}),
        "description": str((existing_sidecar_state or {}).get("description") or ""),
        "author_text": str((existing_sidecar_state or {}).get("author_text") or ""),
        "scene_text": str((existing_sidecar_state or {}).get("scene_text") or ""),
        "caption_ocr_lang": str((existing_detections.get("ocr") or {}).get("language") or ""),
    }


def _run_caption_analysis_step(
    *,
    image_path: Path,
    model_image_path: Path,
    people_source_path: Path | None,
    caption_source_path: Path | None,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    people_matcher: Any,
    people_matches: list,
    people_names: list[str],
    object_labels: list[str],
    ocr_text: str,
    album_title: str,
    printed_album_title: str,
    page_photo_count: int,
    people_positions: list,
    clean_context_ocr: str,
    debug_recorder,
    prompt_debug: PromptDebugSession | None,
    step_runner: StepRunner | None,
    existing_sidecar_state: dict | None,
    existing_detections: dict[str, Any],
    metadata_engine: MetadataEngine | None,
) -> dict[str, Any]:
    empty = {
        "caption_output": None,
        "caption_step_output": {},
        "description": "",
        "author_text": "",
        "scene_text": "",
        "caption_ocr_text": "",
        "caption_ocr_lang": "",
        "faces_detected": _last_faces_detected(people_matcher),
    }
    if metadata_engine is not None:
        return empty

    source_path = caption_source_path or people_source_path or image_path

    def _generate_caption():
        return caption_engine.generate(
            image_path=model_image_path,
            people=people_names,
            objects=object_labels,
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=page_photo_count,
            people_positions=people_positions,
            context_ocr_text=clean_context_ocr,
            debug_recorder=debug_recorder,
            debug_step="caption",
        )

    if step_runner is None:
        caption_output = _generate_caption()
        fields = _caption_text_fields(caption_output)
        return {
            **empty,
            **fields,
            "caption_output": caption_output,
            "faces_detected": _last_faces_detected(people_matcher),
        }

    state = {"caption_output": None, "ran": False}

    def _do_caption() -> dict[str, Any]:
        output = _generate_caption()
        state["caption_output"] = output
        state["ran"] = True
        faces = _last_faces_detected(people_matcher)
        local_pp, local_epc = _estimate_people_from_detections(
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
            source_path=source_path,
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
    if state["ran"] and state["caption_output"] is not None:
        fields = _caption_text_fields(state["caption_output"])
        return {
            **empty,
            **fields,
            "caption_output": state["caption_output"],
            "caption_step_output": dict(caption_step_output or {}),
            "faces_detected": _last_faces_detected(people_matcher),
        }
    return _caption_step_cached_result(empty, caption_step_output, existing_sidecar_state, existing_detections)


def _metadata_image_path_for_step(
    image_path: Path, caption_source_path: Path | None, is_page_scan: bool
) -> Path:
    if not is_page_scan:
        return image_path
    from .ai_view_regions import _region_association_overlay_path  # pylint: disable=import-outside-toplevel

    overlay_lookup_path = Path(caption_source_path) if caption_source_path else image_path
    overlay_path = _region_association_overlay_path(overlay_lookup_path)
    return overlay_path if overlay_path.is_file() else image_path


def _extract_metadata_text_fields(result: Any, state: dict[str, Any]) -> None:
    seen: set[str] = set()
    all_captions: list[str] = []
    for photo in result.photos:
        caption = _metadata_region_caption(photo)
        if caption and caption.casefold() not in seen:
            seen.add(caption.casefold())
            all_captions.append(caption)
    scene_ocr = [photo.scene_ocr for photo in result.photos if photo.scene_ocr]
    state["description"] = " ".join(all_captions)
    state["author_text"] = state["description"]
    state["scene_text"] = "\n".join(scene_ocr)
    state["ocr_text"] = "\n".join(filter(None, [state["author_text"], state["scene_text"]]))
    state["metadata_dc_date"] = next((photo.est_date for photo in result.photos if photo.est_date), "")


def _metadata_cached_state(
    state: dict[str, Any],
    image_path: Path,
    existing_sidecar_state: dict | None,
) -> None:
    from .ai_sidecar_state import _effective_sidecar_ocr_text  # pylint: disable=import-outside-toplevel

    state["description"] = str((existing_sidecar_state or {}).get("description") or "")
    state["author_text"] = str((existing_sidecar_state or {}).get("author_text") or "")
    state["scene_text"] = str((existing_sidecar_state or {}).get("scene_text") or "")
    state["ocr_text"] = str(_effective_sidecar_ocr_text(image_path, existing_sidecar_state) or "").strip()
    state["location_payload"] = _metadata_known_location_payload(state["metadata_output"].get("location"))
    state["locations_shown"] = _metadata_known_locations_shown(state["metadata_output"].get("locations_shown"))
    state["locations_shown_ran"] = bool(state["metadata_output"].get("location_shown_ran", False))


def _metadata_primary_location(result: Any) -> str:
    return next(
        (
            candidate
            for photo in result.photos
            for candidate in (photo.location, photo.location_name)
            if _metadata_location_is_known(candidate)
        ),
        "",
    )


def _metadata_location_is_known(value: object) -> bool:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    return bool(tokens) and any(token not in {"unknown", "country", "location"} for token in tokens)


def _metadata_known_location_payload(value: object) -> dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, dict) else {}
    known_text = next(
        (
            str(payload.get(key) or "").strip()
            for key in ("address", "city", "state", "country", "sublocation", "display_name", "query")
            if _metadata_location_is_known(payload.get(key))
        ),
        "",
    )
    return payload if known_text else {}


def _metadata_known_locations_shown(value: object) -> list[dict[str, Any]]:
    return [
        dict(location)
        for location in list(value or [])
        if isinstance(location, dict) and _metadata_location_is_known(location.get("name"))
    ]


def _metadata_step_location_payload(
    primary_location: str,
    geocoder: NominatimGeocoder | None,
    geocode_recorder,
) -> dict[str, Any]:
    if not primary_location:
        return {}
    return _resolve_location_payload(
        geocoder=geocoder,
        gps_latitude="",
        gps_longitude="",
        location_name=primary_location,
        artifact_recorder=geocode_recorder,
        artifact_step="location",
    )


def _metadata_step_update_state(
    result: Any,
    *,
    state: dict[str, Any],
    image_path: Path,
    caption_source_path: Path | None,
    geocoder: NominatimGeocoder | None,
    geocode_recorder,
) -> None:
    _extract_metadata_text_fields(result, state)
    primary_location = _metadata_primary_location(result)
    state["location_payload"] = _metadata_step_location_payload(primary_location, geocoder, geocode_recorder)
    state["locations_shown"] = _metadata_known_locations_shown(
        [{"name": photo.location_name} for photo in result.photos]
    )
    state["locations_shown_ran"] = True
    _update_region_captions_from_metadata(
        Path(caption_source_path) if caption_source_path else image_path,
        [
            {
                "photo_number": int(photo.photo_number),
                "caption": _metadata_region_caption(photo),
                "location": str(getattr(photo, "location", "") or "").strip(),
                "location_name": str(getattr(photo, "location_name", "") or "").strip(),
                "est_date": str(getattr(photo, "est_date", "") or "").strip(),
            }
            for photo in result.photos
            if int(photo.photo_number) > 0
        ],
    )


def _metadata_step_build_output(
    result: Any,
    state: dict[str, Any],
    metadata_engine: MetadataEngine,
) -> dict[str, Any]:
    ocr_kw = extract_keywords(str(state["ocr_text"]), max_keywords=15)
    metadata_photos = [_metadata_photo_payload(photo) for photo in result.photos]
    subjects = _dedupe(list(result.subjects or []) + _metadata_corrected_caption_subjects(result.photos))
    return {
        "ocr": {
            "engine": metadata_engine.engine,
            "model": str(metadata_engine.effective_model_name),
            "language": "",
            "keywords": ocr_kw,
            "chars": len(str(state["ocr_text"])),
        },
        "caption": {
            "effective_engine": metadata_engine.engine,
            "fallback": bool(result.fallback),
            "error": str(result.error or ""),
            "model": str(metadata_engine.effective_model_name),
            "people_present": bool(result.people_count > 0),
            "estimated_people_count": result.people_count,
            "photos": metadata_photos,
        },
        "subjects": subjects,
        "location": state["location_payload"],
        "locations_shown": state["locations_shown"],
        "location_shown_ran": state["locations_shown_ran"],
    }


def _run_metadata_analysis_step(
    *,
    image_path: Path,
    caption_source_path: Path | None,
    people_source_path: Path | None,
    album_title: str,
    geocoder: NominatimGeocoder | None,
    geocode_recorder,
    debug_recorder,
    metadata_engine: MetadataEngine | None,
    step_runner: StepRunner | None,
    existing_sidecar_state: dict | None,
    is_page_scan: bool,
    step_fn,
    title_page_location: dict[str, str] | None,
    apply_title_location,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "metadata_output": {},
        "metadata_dc_date": "",
        "description": "",
        "author_text": "",
        "scene_text": "",
        "ocr_text": "",
        "location_payload": {},
        "locations_shown": [],
        "locations_shown_ran": False,
    }
    if metadata_engine is None or step_runner is None:
        return state
    if step_fn:
        step_fn("metadata")
    ran = {"value": False}

    def _do_metadata() -> dict[str, Any] | None:
        if _is_derived_image_path(image_path):
            return None
        metadata_image_path = _metadata_image_path_for_step(image_path, caption_source_path, is_page_scan)
        if _is_album_title_source_candidate(caption_source_path or image_path):
            # Title pages have no individual photos — the metadata LLM prompt asks about photos
            # and returns photos:[] for covers. Run OCR directly to capture the cover text.
            ocr_engine = OCREngine(engine=metadata_engine.engine, base_url=metadata_engine.base_url)
            state["ocr_text"] = ocr_engine.read_text(
                metadata_image_path, debug_recorder=debug_recorder, debug_step="cover_ocr"
            )
            ran["value"] = True
            return _metadata_step_build_output(MetadataResult(engine=metadata_engine.engine), state, metadata_engine)
        source = caption_source_path or people_source_path or image_path
        result = metadata_engine.analyze(
            metadata_image_path,
            album_title=album_title,
            source_path=source,
            debug_recorder=debug_recorder,
            debug_step="metadata",
        )
        ran["value"] = True
        _metadata_step_update_state(
            result,
            state=state,
            image_path=image_path,
            caption_source_path=caption_source_path,
            geocoder=geocoder,
            geocode_recorder=geocode_recorder,
        )
        return _metadata_step_build_output(result, state, metadata_engine)

    metadata_output = step_runner.run("metadata", _do_metadata, model=str(metadata_engine.effective_model_name))
    state["metadata_output"] = dict(metadata_output or {})
    if not ran["value"]:
        _metadata_cached_state(state, image_path, existing_sidecar_state)
    state["location_payload"], _ = apply_title_location(
        image_path=image_path,
        location_payload=state["location_payload"],
        title_page_location=title_page_location,
    )
    return state


def _build_analysis_payload(
    *,
    step_runner: StepRunner | None,
    existing_detections: dict[str, Any],
    metadata_output: dict[str, Any],
    caption_step_output: dict[str, Any],
    objects_output: dict[str, Any],
    people_matches: list,
    people_names: list[str],
    object_matches: list,
    object_labels: list[str],
    faces_detected: int,
    ocr_text: str,
    ocr_keywords: list[str],
    ocr_engine: OCREngine,
    ocr_language: str,
    requested_caption_engine: str,
    caption_engine: CaptionEngine,
    caption_output: Any,
    caption_ocr_lang: str,
    model_image_path: Path,
    source_path: Path,
    album_title: str,
    printed_album_title: str,
    people_positions: list,
    prompt_debug: PromptDebugSession | None,
    object_detector: Any,
) -> dict[str, Any]:
    if step_runner is not None:
        return _build_step_runner_analysis_payload(
            step_runner=step_runner,
            existing_detections=existing_detections,
            metadata_output=metadata_output,
            caption_step_output=caption_step_output,
            objects_output=objects_output,
            people_matches=people_matches,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            ocr_keywords=ocr_keywords,
            ocr_text=ocr_text,
        )

    local_people_present, local_estimated_people_count = _estimate_people_from_detections(
        people_matches=people_matches,
        people_names=people_names,
        object_labels=object_labels,
        faces_detected=faces_detected,
    )
    people_present, estimated_people_count = _resolve_people_count_metadata(
        requested_caption_engine=requested_caption_engine,
        caption_engine=caption_engine,
        model_image_path=model_image_path,
        people=people_names,
        objects=object_labels,
        ocr_text=ocr_text,
        source_path=source_path,
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
        "objects": [
            {"label": getattr(row, "label", ""), "score": round(getattr(row, "score", 0.0), 5)}
            for row in object_matches
        ],
        "ocr": {
            "engine": str(caption_output.engine),
            "model": str(caption_engine.effective_model_name),
            "language": str(caption_ocr_lang or ocr_language),
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
    return payload


def _build_step_runner_analysis_payload(
    *,
    step_runner: StepRunner,
    existing_detections: dict[str, Any],
    metadata_output: dict[str, Any],
    caption_step_output: dict[str, Any],
    objects_output: dict[str, Any],
    people_matches: list,
    ocr_engine: OCREngine,
    ocr_language: str,
    ocr_keywords: list[str],
    ocr_text: str,
) -> dict[str, Any]:
    selected_object_output = objects_output if step_runner.reran.get("objects") else existing_detections
    payload: dict[str, Any] = {
        "people": _step_runner_people_payload(
            step_runner=step_runner,
            existing_detections=existing_detections,
            people_matches=people_matches,
        ),
        "objects": list(selected_object_output.get("objects") or []),
        "ocr": _step_runner_ocr_meta(
            step_runner=step_runner,
            metadata_output=metadata_output,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            ocr_keywords=ocr_keywords,
            ocr_text=ocr_text,
        ),
        "caption": _step_runner_caption_meta(
            step_runner=step_runner,
            metadata_output=metadata_output,
            caption_step_output=caption_step_output,
            existing_detections=existing_detections,
        ),
    }
    object_model_val = str(selected_object_output.get("object_model") or "")
    if object_model_val:
        payload["object_model"] = object_model_val
    return payload


def _step_runner_people_payload(
    *, step_runner: StepRunner, existing_detections: dict[str, Any], people_matches: list
) -> list:
    if step_runner.reran.get("people"):
        return _serialize_people_matches(people_matches)
    return list(existing_detections.get("people") or [])


def _step_runner_ocr_meta(
    *,
    step_runner: StepRunner,
    metadata_output: dict[str, Any],
    ocr_engine: OCREngine,
    ocr_language: str,
    ocr_keywords: list[str],
    ocr_text: str,
) -> dict[str, Any]:
    ocr_meta = dict(step_runner.existing_detections.get("ocr") or {})
    if step_runner.reran.get("metadata"):
        return dict(metadata_output.get("ocr") or {})
    if step_runner.reran.get("ocr"):
        return {
            "engine": ocr_engine.engine,
            "model": str(ocr_engine.effective_model_name),
            "language": ocr_language,
            "keywords": ocr_keywords,
            "chars": len(ocr_text),
        }
    if ocr_meta:
        ocr_meta["keywords"] = ocr_keywords
        ocr_meta["chars"] = len(ocr_text)
    return ocr_meta


def _step_runner_caption_meta(
    *,
    step_runner: StepRunner,
    metadata_output: dict[str, Any],
    caption_step_output: dict[str, Any],
    existing_detections: dict[str, Any],
) -> dict[str, Any]:
    if step_runner.reran.get("metadata"):
        return dict(metadata_output.get("caption") or {})
    source = caption_step_output if step_runner.reran.get("caption") else existing_detections
    return dict(source.get("caption") or {})


def _run_step_runner_locations(
    *,
    image_path: Path,
    caption_source_path: Path | None,
    people_source_path: Path | None,
    caption_engine: CaptionEngine,
    description: str,
    ocr_text: str,
    clean_context_ocr: str,
    clean_context_location: str,
    album_title: str,
    printed_album_title: str,
    geocoder: NominatimGeocoder | None,
    prompt_debug: PromptDebugSession | None,
    geocode_recorder,
    step_runner: StepRunner,
    title_page_location: dict[str, str] | None,
    apply_title_location,
    context_ocr_text_fn,
    step_fn,
) -> tuple[dict[str, Any], list, bool]:
    if step_fn:
        step_fn("locations")
    context_ocr_for_locations = context_ocr_text_fn(
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
            caption_text=description,
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
    location_payload, _ = apply_title_location(
        image_path=image_path,
        location_payload=location_payload,
        title_page_location=title_page_location,
    )
    return location_payload, locations_shown, locations_shown_ran


def _run_model_image_analysis_steps(
    *,
    image_path: Path,
    people_input_path: Path,
    people_coordinate_path: Path,
    people_bbox_offset: tuple[int, int],
    caption_source_path: Path | None,
    people_source_path: Path | None,
    people_hint_text: str,
    context_ocr_text: str,
    context_location_hint: str,
    extra_people_names: list[str] | None,
    ocr_text_override: str | None,
    ocr_engine: OCREngine,
    ocr_language: str,
    caption_engine: CaptionEngine,
    requested_caption_engine: str,
    people_matcher: Any,
    object_detector: Any,
    album_title: str,
    printed_album_title: str,
    page_photo_count: int,
    debug_recorder,
    prompt_debug: PromptDebugSession | None,
    step_runner: StepRunner | None,
    existing_sidecar_state: dict | None,
    existing_detections: dict[str, Any],
    metadata_engine: MetadataEngine | None,
    step_fn,
    match_people,
    format_people_step_label,
    compute_people_positions,
) -> dict[str, Any]:
    with _prepare_ai_model_image(image_path) as model_image_path:
        clean_context_ocr = str(context_ocr_text or "").strip()
        clean_context_location = str(context_location_hint or "").strip()
        ocr_text = _run_ocr_analysis_step(
            image_path=image_path,
            model_image_path=model_image_path,
            ocr_text=str(ocr_text_override or "").strip(),
            ocr_text_override=ocr_text_override,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            caption_source_path=caption_source_path,
            people_source_path=people_source_path,
            debug_recorder=debug_recorder,
            step_fn=step_fn,
            step_runner=step_runner,
            existing_sidecar_state=existing_sidecar_state,
            metadata_engine=metadata_engine,
        )
        people_matches, faces_detected = _model_people_state(
            people_matcher=people_matcher,
            people_input_path=people_input_path,
            people_coordinate_path=people_coordinate_path,
            people_bbox_offset=people_bbox_offset,
            people_hint_text=people_hint_text,
            ocr_text=ocr_text,
            clean_context_ocr=clean_context_ocr,
            step_runner=step_runner,
            match_people=match_people,
        )
        people_names = _model_people_names(
            people_matches=people_matches,
            extra_people_names=extra_people_names,
            step_fn=step_fn,
            format_people_step_label=format_people_step_label,
        )
        people_positions = compute_people_positions(people_matches, people_coordinate_path)
        object_matches, objects_output = _model_objects_state(
            object_detector=object_detector,
            model_image_path=model_image_path,
            step_runner=step_runner,
            step_fn=step_fn,
        )
        object_labels = _object_labels(object_matches)
        caption_state = _model_caption_state(
            image_path=image_path,
            model_image_path=model_image_path,
            people_source_path=people_source_path,
            caption_source_path=caption_source_path,
            caption_engine=caption_engine,
            requested_caption_engine=requested_caption_engine,
            people_matcher=people_matcher,
            people_matches=people_matches,
            people_names=people_names,
            object_labels=object_labels,
            ocr_text=ocr_text,
            album_title=album_title,
            printed_album_title=printed_album_title,
            page_photo_count=page_photo_count,
            people_positions=people_positions,
            clean_context_ocr=clean_context_ocr,
            debug_recorder=debug_recorder,
            prompt_debug=prompt_debug,
            step_runner=step_runner,
            existing_sidecar_state=existing_sidecar_state,
            existing_detections=existing_detections,
            metadata_engine=metadata_engine,
            step_fn=step_fn,
        )
    return {
        "model_image_path": model_image_path,
        "clean_context_ocr": clean_context_ocr,
        "clean_context_location": clean_context_location,
        "ocr_text": ocr_text,
        "people_matches": people_matches,
        "people_names": people_names,
        "people_positions": people_positions,
        "object_matches": object_matches,
        "objects_output": objects_output,
        "object_labels": object_labels,
        **caption_state,
        "faces_detected": int(caption_state["faces_detected"] or faces_detected),
    }


def _model_people_state(
    *,
    people_matcher: Any,
    people_input_path: Path,
    people_coordinate_path: Path,
    people_bbox_offset: tuple[int, int],
    people_hint_text: str,
    ocr_text: str,
    clean_context_ocr: str,
    step_runner: StepRunner | None,
    match_people,
) -> tuple[list, int]:
    combined_hint_text = " ".join(
        part for part in [str(people_hint_text or "").strip(), ocr_text, clean_context_ocr] if part
    ).strip()
    return _run_people_analysis_step(
        people_matcher=people_matcher,
        people_input_path=people_input_path,
        people_coordinate_path=people_coordinate_path,
        people_bbox_offset=people_bbox_offset,
        combined_hint_text=combined_hint_text,
        step_runner=step_runner,
        match_people=match_people,
    )


def _model_people_names(
    *,
    people_matches: list,
    extra_people_names: list[str] | None,
    step_fn,
    format_people_step_label,
) -> list[str]:
    people_match_names = _dedupe([getattr(row, "name", None) or "" for row in people_matches])
    people_match_names = [n for n in people_match_names if n]
    if step_fn:
        step_fn(format_people_step_label("people", people_match_names))
    return _dedupe(people_match_names + list(extra_people_names or []))


def _model_objects_state(
    *,
    object_detector: Any,
    model_image_path: Path,
    step_runner: StepRunner | None,
    step_fn,
) -> tuple[list, dict[str, Any]]:
    if step_fn:
        step_fn("objects")
    return _run_objects_analysis_step(
        object_detector=object_detector,
        model_image_path=model_image_path,
        step_runner=step_runner,
    )


def _model_caption_state(*, step_fn, metadata_engine: MetadataEngine | None, **kwargs) -> dict[str, Any]:
    if step_fn and metadata_engine is None:
        step_fn("caption")
    return _run_caption_analysis_step(metadata_engine=metadata_engine, **kwargs)


def _attach_location_payload(
    *,
    payload: dict[str, Any],
    step_runner: StepRunner | None,
    metadata_engine: MetadataEngine | None,
    location_payload: dict[str, Any],
    locations_shown: list,
    locations_shown_ran: bool,
    step_fn,
) -> None:
    if step_runner is not None:
        if location_payload:
            payload["location"] = location_payload
        if step_fn and metadata_engine is None:
            step_fn("locations_shown")
        payload["locations_shown"] = locations_shown
        payload["location_shown_ran"] = locations_shown_ran
        return
    if step_fn:
        step_fn("locations_shown")
    payload["locations_shown"] = []
    payload["location_shown_ran"] = False


def _unpack_model_state(model_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_image_path": model_state["model_image_path"],
        "clean_context_ocr": str(model_state["clean_context_ocr"]),
        "clean_context_location": str(model_state["clean_context_location"]),
        "ocr_text": str(model_state["ocr_text"]),
        "people_matches": list(model_state["people_matches"]),
        "people_names": list(model_state["people_names"]),
        "people_positions": list(model_state["people_positions"]),
        "object_matches": list(model_state["object_matches"]),
        "objects_output": dict(model_state["objects_output"]),
        "object_labels": list(model_state["object_labels"]),
        "caption_output": model_state["caption_output"],
        "caption_step_output": dict(model_state["caption_step_output"]),
        "description": str(model_state["description"]),
        "author_text": str(model_state["author_text"]),
        "scene_text": str(model_state["scene_text"]),
        "caption_ocr_text": str(model_state["caption_ocr_text"]),
        "caption_ocr_lang": str(model_state["caption_ocr_lang"]),
        "faces_detected": int(model_state["faces_detected"]),
    }


def _ria_override_ocr_from_caption(
    ocr_text: str,
    caption_ocr_text: str,
    ocr_text_override: str | None,
    metadata_engine: MetadataEngine | None,
) -> str:
    if ocr_text_override is None and not ocr_text and metadata_engine is None:
        return caption_ocr_text
    return ocr_text


def _ria_apply_metadata_values(
    ms: dict[str, Any],
    metadata_state: dict[str, Any],
    metadata_engine: MetadataEngine | None,
    step_runner: StepRunner | None,
) -> dict[str, Any]:
    out = {
        "metadata_output": {},
        "metadata_dc_date": "",
        "description": ms["description"],
        "author_text": ms["author_text"],
        "scene_text": ms["scene_text"],
        "ocr_text": ms["ocr_text"],
        "location_payload": {},
        "locations_shown": [],
        "locations_shown_ran": False,
    }
    if metadata_engine is not None and step_runner is not None:
        mv = _metadata_state_values(metadata_state)
        out.update(mv)
    return out


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
    from .ai_index_runner import (  # pylint: disable=import-outside-toplevel
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

    def geocode_recorder(record) -> None:
        _append_geocode_artifact(image_path=image_path, record=record)

    ms = _unpack_model_state(_run_model_image_analysis_steps(
        image_path=image_path,
        people_input_path=people_input_path,
        people_coordinate_path=people_coordinate_path,
        people_bbox_offset=people_bbox_offset,
        caption_source_path=caption_source_path,
        people_source_path=people_source_path,
        people_hint_text=people_hint_text,
        context_ocr_text=context_ocr_text,
        context_location_hint=context_location_hint,
        extra_people_names=extra_people_names,
        ocr_text_override=ocr_text_override,
        ocr_engine=ocr_engine,
        ocr_language=ocr_language,
        caption_engine=caption_engine,
        requested_caption_engine=requested_caption_engine,
        people_matcher=people_matcher,
        object_detector=object_detector,
        album_title=album_title,
        printed_album_title=printed_album_title,
        page_photo_count=page_photo_count,
        debug_recorder=debug_recorder,
        prompt_debug=prompt_debug,
        step_runner=step_runner,
        existing_sidecar_state=existing_sidecar_state,
        existing_detections=existing_detections,
        metadata_engine=metadata_engine,
        step_fn=step_fn,
        match_people=_match_people_with_cast_store_retry,
        format_people_step_label=_format_people_step_label,
        compute_people_positions=_compute_people_positions,
    ))

    metadata_state = _run_metadata_analysis_step(
        image_path=image_path,
        caption_source_path=caption_source_path,
        people_source_path=people_source_path,
        album_title=album_title,
        geocoder=geocoder,
        geocode_recorder=geocode_recorder,
        debug_recorder=debug_recorder,
        metadata_engine=metadata_engine,
        step_runner=step_runner,
        existing_sidecar_state=existing_sidecar_state,
        is_page_scan=is_page_scan,
        step_fn=step_fn,
        title_page_location=title_page_location,
        apply_title_location=_apply_title_page_location_config,
    )
    av = _ria_apply_metadata_values(ms, metadata_state, metadata_engine, step_runner)
    ocr_text = _ria_override_ocr_from_caption(
        av["ocr_text"], ms["caption_ocr_text"], ocr_text_override, metadata_engine
    )
    ocr_keywords = extract_keywords(ocr_text, max_keywords=15)
    llm_subjects = list(av["metadata_output"].get("subjects") or [])
    subjects = _dedupe(ms["object_labels"] + (llm_subjects if llm_subjects else ocr_keywords))
    payload = _build_analysis_payload(
        step_runner=step_runner,
        existing_detections=existing_detections,
        metadata_output=av["metadata_output"],
        caption_step_output=ms["caption_step_output"],
        objects_output=ms["objects_output"],
        people_matches=ms["people_matches"],
        people_names=ms["people_names"],
        object_matches=ms["object_matches"],
        object_labels=ms["object_labels"],
        faces_detected=ms["faces_detected"],
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        ocr_engine=ocr_engine,
        ocr_language=ocr_language,
        requested_caption_engine=requested_caption_engine,
        caption_engine=caption_engine,
        caption_output=ms["caption_output"],
        caption_ocr_lang=ms["caption_ocr_lang"],
        model_image_path=ms["model_image_path"],
        source_path=caption_source_path or people_source_path or image_path,
        album_title=album_title,
        printed_album_title=printed_album_title,
        people_positions=ms["people_positions"],
        prompt_debug=prompt_debug,
        object_detector=object_detector,
    )
    location_payload = av["location_payload"]
    locations_shown = av["locations_shown"]
    locations_shown_ran = av["locations_shown_ran"]

    # ── Location step (legacy caption_engine path) or via metadata step ───────
    if step_runner is not None and metadata_engine is None:
        location_payload, locations_shown, locations_shown_ran = _run_step_runner_locations(
            image_path=image_path,
            caption_source_path=caption_source_path,
            people_source_path=people_source_path,
            caption_engine=caption_engine,
            description=av["description"],
            ocr_text=ocr_text,
            clean_context_ocr=ms["clean_context_ocr"],
            clean_context_location=ms["clean_context_location"],
            album_title=album_title,
            printed_album_title=printed_album_title,
            geocoder=geocoder,
            prompt_debug=prompt_debug,
            geocode_recorder=geocode_recorder,
            step_runner=step_runner,
            title_page_location=title_page_location,
            apply_title_location=_apply_title_page_location_config,
            context_ocr_text_fn=_contextualize_ocr_text,
            step_fn=step_fn,
        )

    return _build_image_analysis(
        image_path=image_path,
        caption_output=ms["caption_output"],
        ocr_text=ocr_text,
        ocr_language=ocr_language,
        ocr_keywords=ocr_keywords,
        subjects=subjects,
        people_names=ms["people_names"],
        object_labels=ms["object_labels"],
        description=av["description"],
        author_text=av["author_text"],
        scene_text=av["scene_text"],
        payload=payload,
        faces_detected=ms["faces_detected"],
        album_title=album_title,
        caption_ocr_lang=ms["caption_ocr_lang"],
        metadata_dc_date=av["metadata_dc_date"],
        location_payload=location_payload,
        locations_shown=locations_shown,
        locations_shown_ran=locations_shown_ran,
        step_runner=step_runner,
        metadata_engine=metadata_engine,
        step_fn=step_fn,
    )


def _build_image_analysis(
    *,
    image_path,
    caption_output,
    ocr_text,
    ocr_language,
    ocr_keywords,
    subjects,
    people_names,
    object_labels,
    description,
    author_text,
    scene_text,
    payload,
    faces_detected,
    album_title,
    caption_ocr_lang,
    metadata_dc_date,
    location_payload,
    locations_shown,
    locations_shown_ran,
    step_runner,
    metadata_engine,
    step_fn,
) -> ImageAnalysis:
    _attach_location_payload(
        payload=payload,
        step_runner=step_runner,
        metadata_engine=metadata_engine,
        location_payload=location_payload,
        locations_shown=locations_shown,
        locations_shown_ran=locations_shown_ran,
        step_fn=step_fn,
    )
    resolved_album_title = _analysis_album_title(
        image_path=image_path,
        caption_output=caption_output,
        ocr_text=ocr_text,
        album_title=album_title,
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
        faces_detected=faces_detected,
        image_regions=_caption_image_regions(caption_output),
        album_title=resolved_album_title,
        title=_caption_title(caption_output),
        ocr_lang=str(caption_ocr_lang or ocr_language),
        dc_date=metadata_dc_date,
    )


def _metadata_state_values(metadata_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_output": dict(metadata_state["metadata_output"]),
        "metadata_dc_date": str(metadata_state["metadata_dc_date"]),
        "description": str(metadata_state["description"]),
        "author_text": str(metadata_state["author_text"]),
        "scene_text": str(metadata_state["scene_text"]),
        "ocr_text": str(metadata_state["ocr_text"]),
        "location_payload": dict(metadata_state["location_payload"]),
        "locations_shown": list(metadata_state["locations_shown"]),
        "locations_shown_ran": bool(metadata_state["locations_shown_ran"]),
    }


def _analysis_album_title(*, image_path: Path, caption_output: Any, ocr_text: str, album_title: str) -> str:
    resolved_album_title = _resolve_title_page_album_title(
        image_path=image_path,
        album_title=str(getattr(caption_output, "album_title", "") if caption_output is not None else ""),
        ocr_text=ocr_text,
    )
    return _require_album_title_for_title_page(
        image_path=image_path,
        album_title=(resolved_album_title or album_title),
        context="analysis",
    )


def _caption_image_regions(caption_output: Any) -> list:
    return list(getattr(caption_output, "image_regions", None) or []) if caption_output is not None else []


def _caption_title(caption_output: Any) -> str:
    return str(getattr(caption_output, "title", "") if caption_output is not None else "")
