from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ai_album_titles import (
    _expand_album_title_dependencies,
    _is_album_title_source_candidate,
    _require_album_title_for_title_page,
    _resolve_album_printed_title_hint,
    _resolve_album_title_from_sidecars,
    _resolve_album_title_hint,
    _resolve_title_page_album_title,
    _store_album_printed_title_hint,
)
from .album_sets import find_archive_set_by_photos_root
from .ai_caption import (
    CaptionEngine,
    DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
    normalize_lmstudio_base_url,
    resolve_caption_model,
)
from .ai_date import DateEstimateEngine
from .ai_metadata import MetadataEngine
from .ai_geocode import NominatimGeocoder
from .ai_location import (
    _has_legacy_ai_locations_shown_gps,
    _resolve_location_metadata,
    _resolve_location_payload,
    _resolve_locations_shown,
    _xmp_gps_to_decimal,
)
from .ai_ocr import OCREngine
from .ai_page_layout import prepare_image_layout
from .ai_processing_locks import (
    _acquire_batch_processing_lock,
    _acquire_image_processing_lock,
    _release_batch_processing_lock,
    _release_image_processing_lock,
)
from .ai_render_settings import (
    find_archive_dir_for_image,
    load_render_settings,
    resolve_effective_settings,
)
from .ai_sidecar_state import (
    _compute_xmp_title,
    _effective_sidecar_album_title,
    _effective_sidecar_location_payload,
    _effective_sidecar_ocr_text,
    _resolve_xmp_text_layers,
    _sidecar_current_for_paths,
    _xmp_timestamp_from_path,
    has_current_sidecar,
    has_valid_sidecar,
    read_embedded_create_date,
)
from .metadata_resolver import resolve_person_in_image
from .prompt_debug import PromptDebugSession
from .ai_prompt_assets import load_params
from .xmp_sidecar import (
    _dedupe,
    _resolve_date_time_original,
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    read_pipeline_state,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
)
from .xmp_review import load_ai_xmp_review
from .ai_index_steps import StepRunner

from .ai_index_args import (
    IMAGE_EXTENSIONS,
    _absolute_cli_path,
    _explicit_cli_flags,
    _resolve_caption_prompt,
    parse_args,
)
from .ai_index_engine_cache import (
    PROCESSOR_SIGNATURE,
    _init_caption_engine,
    _init_date_engine,
    _init_object_detector,
    _init_people_matcher,
    _settings_signature,
)
from .ai_index_analysis import (
    ArchiveScanOCRAuthority,
    _build_caption_metadata,
    _estimate_people_from_detections,
    _get_image_dimensions,
    _prepare_ai_model_image,
    _refresh_detection_model_metadata,
    _resolve_people_count_metadata,
    _run_image_analysis,
    _serialize_people_matches,
)
from .ai_index_scan import (
    _bounds_offset,
    _build_dc_source,
    _build_flat_page_description,
    _build_flat_payload,
    _dc_source_needs_refresh,
    _hash_text,
    _page_scan_filenames,
    _resolve_archive_scan_authoritative_ocr,
    _scan_group_paths,
    _scan_group_signature,
)
from .ai_index import (
    _apply_title_page_location_config,
    _match_people_with_cast_store_retry,
    _compute_people_positions,
    _format_people_step_label,
    _mirror_page_sidecars,
    _append_xmp_job_artifact,
    _emit_prompt_debug_artifact,
    _append_geocode_artifact,
    discover_images,
    _coalesce_archive_processing_files,
    _filter_files_by_tree,
    _apply_shard,
    _display_work_label,
    _format_eta,
    _progress_ticker,
    _format_reprocess_reasons,
    _format_location_hint_from_state,
    _resolve_upstream_page_sidecar_state,
    needs_processing,
    _sidecar_has_lmstudio_caption_error,
    _sidecar_has_people_to_refresh,
    _date_estimate_input_hash,
    _dc_date_value,
    _has_dc_date,
    _dc_date_needs_refresh,
    _resolve_dc_date,
    _configured_title_page_location_payload,
    _is_archive_file,
    _page_sort_key,
)


def _write_sidecar_and_record(
    sidecar_path: Path,
    image_path: Path,
    *,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str = "",
    location_payload: dict[str, Any],
    source_text: str = "",
    ocr_text: str,
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    detections_payload: dict[str, Any] | None = None,
    subphotos: list[dict[str, Any]] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    create_date: str = "",
    dc_date: str | list[str] = "",
    date_time_original: str = "",
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
    title_page_location: dict[str, str] | None = None,
) -> None:
    """Write XMP sidecar and record the artifact.  Derives history_when and image
    dimensions from image_path; unpacks GPS fields from location_payload."""
    img_w, img_h = _get_image_dimensions(image_path)
    loc, detections_payload = _apply_title_page_location_config(
        image_path=image_path,
        location_payload=location_payload,
        detections_payload=detections_payload,
        title_page_location=title_page_location,
    )
    resolved_person_names = resolve_person_in_image(
        person_names,
        locations_shown=(
            list(detections_payload.get("locations_shown") or [])
            if isinstance(detections_payload, dict)
            else []
        ),
        location_payload=loc,
    )
    write_xmp_sidecar(
        sidecar_path,
        person_names=resolved_person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=description,
        album_title=album_title,
        gps_latitude=str(loc.get("gps_latitude") or ""),
        gps_longitude=str(loc.get("gps_longitude") or ""),
        location_city=str(loc.get("city") or ""),
        location_state=str(loc.get("state") or ""),
        location_country=str(loc.get("country") or ""),
        location_sublocation=str(loc.get("sublocation") or ""),
        source_text=source_text,
        ocr_text=ocr_text,
        ocr_lang=ocr_lang,
        author_text=author_text,
        scene_text=scene_text,
        detections_payload=detections_payload,
        subphotos=subphotos,
        stitch_key=stitch_key,
        ocr_authority_source=ocr_authority_source,
        create_date=create_date,
        dc_date=dc_date,
        date_time_original=date_time_original,
        history_when=_xmp_timestamp_from_path(image_path),
        image_width=img_w,
        image_height=img_h,
        ocr_ran=ocr_ran,
        people_detected=bool(people_detected or resolved_person_names),
        people_identified=bool(resolved_person_names),
        locations_shown=detections_payload.get("locations_shown") if detections_payload else None,
    )
    _append_xmp_job_artifact(image_path, sidecar_path)


_CAPTION_PROMPT_OVERRIDE_FLAGS = frozenset(
    {
        "--caption-prompt",
        "--local-prompt",
        "--qwen-prompt",
        "--caption-prompt-file",
        "--local-prompt-file",
        "--qwen-prompt-file",
    }
)

_SIMPLE_CLI_OVERRIDES: tuple[tuple[str, str, str, Any], ...] = (
    ("--ocr-engine", "ocr_engine", "ocr_engine", str),
    ("--ocr-model", "ocr_model", "ocr_model", str),
    ("--caption-engine", "caption_engine", "caption_engine", str),
    ("--caption-model", "caption_model", "caption_model", str),
    ("--caption-max-tokens", "caption_max_tokens", "caption_max_tokens", int),
    ("--caption-temperature", "caption_temperature", "caption_temperature", float),
    ("--caption-max-edge", "caption_max_edge", "caption_max_edge", int),
)

_OVERRIDE_SOURCE_FLAGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("caption_prompt", _CAPTION_PROMPT_OVERRIDE_FLAGS),
    ("caption_max_tokens", frozenset({"--caption-max-tokens"})),
    ("caption_temperature", frozenset({"--caption-temperature"})),
    ("caption_max_edge", frozenset({"--caption-max-edge"})),
)


def _caption_engine_lower(effective: dict[str, Any], defaults: dict[str, Any]) -> str:
    return str(effective.get("caption_engine", defaults["caption_engine"])).strip().lower()


def _is_gps_repair_requested(state: "_ProcessOneState") -> bool:
    return (
        state.existing_sidecar_current
        and state.existing_sidecar_complete
        and state.existing_sidecar_state is not None
        and (state.location_shown_missing or state.location_shown_gps_dirty)
        and not state.reprocess_required
        and not state.source_refresh_required
        and not state.date_refresh_required
    )


@dataclass
class _FullEngines:
    caption_engine: CaptionEngine
    caption_key: tuple[str, str, str, int, float, str, int, bool]
    ocr_engine: OCREngine
    ocr_key: tuple[str, str, str, str]
    object_detector: Any


@dataclass
class _FullHints:
    album_title_hint: str
    upstream_context_ocr: str
    upstream_location_hint: str


@dataclass
class _FullAnalysisTargets:
    analysis_target: Path
    people_analysis_source: Path


@dataclass
class _FullAnalysisOutcome:
    analysis: Any
    payload: dict[str, Any]
    person_names: list[str]
    subjects: list[str]
    description: str
    ocr_text: str
    resolved_album_title: str
    analysis_mode: str
    ocr_authority_hash: str
    scan_filenames: list[str]
    step_runner: StepRunner
    existing_detections: dict[str, Any]


def _resolve_full_hints(image_path: Path, printed_album_title_cache: dict[str, str]) -> _FullHints:
    album_title_hint = _resolve_album_title_hint(image_path)
    printed_hint = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
    upstream_page_state = _resolve_upstream_page_sidecar_state(image_path)
    upstream_context_ocr = str((upstream_page_state or {}).get("ocr_text") or "").strip()
    upstream_location_hint = _format_location_hint_from_state(upstream_page_state)
    upstream_album = str((upstream_page_state or {}).get("album_title") or "").strip()
    if not album_title_hint:
        album_title_hint = upstream_album
    if not printed_hint:
        printed_hint = upstream_album
    return _FullHints(
        album_title_hint=album_title_hint,
        upstream_context_ocr=upstream_context_ocr,
        upstream_location_hint=upstream_location_hint,
    )


def _full_analysis_targets(
    image_path: Path,
    layout: Any,
    scan_ocr_authority: ArchiveScanOCRAuthority | None,
) -> _FullAnalysisTargets:
    stitched_path = scan_ocr_authority.stitched_image_path if scan_ocr_authority is not None else None
    layout_target = layout.content_path if layout.page_like else image_path
    analysis_target = stitched_path or layout_target
    people_analysis_source = analysis_target if scan_ocr_authority is not None else layout_target
    return _FullAnalysisTargets(analysis_target=analysis_target, people_analysis_source=people_analysis_source)


def _full_engine_model_name(engine: Any, requested: str) -> str:
    if str(requested).strip().lower() in {"local", "lmstudio"}:
        return str(engine.effective_model_name)
    return ""


def _merge_pipeline_records(
    payload: dict[str, Any], existing_detections: dict[str, Any], step_runner: StepRunner
) -> None:
    pending = step_runner.get_pending_records()
    merged = dict(existing_detections.get("pipeline") or {})
    merged.update(dict(payload.get("pipeline") or {}))
    merged.update(pending)
    if merged:
        payload["pipeline"] = merged


def _full_resolve_location_payload(
    payload: dict[str, Any],
    step_runner: StepRunner,
    image_path: Path,
    existing_sidecar_state: dict | None,
) -> dict[str, Any]:
    location_payload = dict(payload.get("location") or {}) if isinstance(payload, dict) else {}
    if step_runner.reran.get("metadata") or step_runner.reran.get("locations"):
        return location_payload
    effective_location_payload = _effective_sidecar_location_payload(image_path, existing_sidecar_state)
    return effective_location_payload or location_payload


def _full_final_dc_date(analysis: Any, existing_sidecar_state: dict | None) -> Any:
    metadata_dc_date = str(analysis.dc_date or "").strip()
    if metadata_dc_date and not _has_dc_date(_dc_date_value(existing_sidecar_state)):
        return metadata_dc_date
    return _dc_date_value(existing_sidecar_state)


def _full_processing_payload(
    *,
    image_path: Path,
    settings_sig: str,
    current_cast_signature: str,
    effective: dict[str, Any],
    ocr_text: str,
    final_album_title: str,
    final_dc_date: Any,
    existing_sidecar_state: dict | None,
    scan_ocr_authority: ArchiveScanOCRAuthority | None,
    ocr_authority_hash: str,
    analysis_mode: str,
    date_estimation_enabled: bool,
) -> dict[str, Any]:
    stat = image_path.stat()
    if date_estimation_enabled or final_dc_date:
        date_hash: str = _date_estimate_input_hash(ocr_text, final_album_title)
    else:
        date_hash = str((existing_sidecar_state or {}).get("date_estimate_input_hash") or "")
    return {
        "processor_signature": PROCESSOR_SIGNATURE,
        "settings_signature": settings_sig,
        "cast_store_signature": (
            current_cast_signature if bool(effective.get("enable_people", True)) else ""
        ),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "date_estimate_input_hash": date_hash,
        "ocr_authority_signature": (
            str(scan_ocr_authority.signature) if scan_ocr_authority is not None else ""
        ),
        "ocr_authority_hash": ocr_authority_hash,
        "analysis_mode": str(analysis_mode),
    }


@dataclass
class _PeopleUpdateInputs:
    detections: dict[str, Any]
    existing_people_rows: list[dict[str, Any]]
    existing_caption_payload: dict[str, Any]
    existing_ocr_text: str
    existing_ocr_keywords: list[str]
    existing_object_labels: list[str]
    existing_location: dict[str, Any]


def _pu_inputs_from_state(image_path: Path, state: dict[str, Any]) -> _PeopleUpdateInputs:
    det = state.get("detections") or {}
    existing_people_rows = [r for r in list(det.get("people") or []) if isinstance(r, dict)]
    existing_caption_payload = dict(det.get("caption") or {})
    existing_ocr_text = _effective_sidecar_ocr_text(image_path, state)
    existing_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
    existing_object_rows = [r for r in list(det.get("objects") or []) if isinstance(r, dict)]
    existing_object_labels = [str(r.get("label") or "") for r in existing_object_rows if r.get("label")]
    existing_location = _effective_sidecar_location_payload(image_path, state)
    return _PeopleUpdateInputs(
        detections=det,
        existing_people_rows=existing_people_rows,
        existing_caption_payload=existing_caption_payload,
        existing_ocr_text=existing_ocr_text,
        existing_ocr_keywords=existing_ocr_keywords,
        existing_object_labels=existing_object_labels,
        existing_location=existing_location,
    )


def _people_matcher_faces(people_matcher: Any) -> int:
    if not people_matcher:
        return 0
    last = getattr(people_matcher, "last_faces_detected", 0)
    return last if isinstance(last, int) else 0


def _pu_match_people(
    people_matcher: Any, image_path: Path, ocr_text: str
) -> tuple[list[Any], int]:
    if not people_matcher:
        return [], 0
    matches = _match_people_with_cast_store_retry(
        people_matcher=people_matcher,
        image_path=image_path,
        source_path=image_path,
        bbox_offset=(0, 0),
        hint_text=ocr_text,
    )
    return matches, _people_matcher_faces(people_matcher)


def _pu_finalize_detections(
    pu_updated_det: dict[str, Any],
    *,
    existing_location: dict[str, Any],
    cast_store_signature: str,
    ocr_text: str,
    album_title: str,
    stamp_date_hash: bool,
) -> dict[str, Any]:
    pu_proc = dict(pu_updated_det.get("processing") or {})
    pu_proc["cast_store_signature"] = cast_store_signature
    if stamp_date_hash:
        pu_proc["date_estimate_input_hash"] = _date_estimate_input_hash(ocr_text, album_title)
    if existing_location:
        pu_updated_det["location"] = existing_location
    return {**pu_updated_det, "processing": pu_proc}


@dataclass
class _GpsInputs:
    detections: dict[str, Any]
    ocr_text: str
    ocr_keywords: list[str]
    people_names: list[str]
    object_labels: list[str]
    album_title: str
    printed_title: str
    existing_location_name: str


def _gps_inputs_from_state(
    image_path: Path, state: dict[str, Any], printed_album_title_cache: dict[str, str]
) -> _GpsInputs:
    det = state.get("detections") or {}
    ocr_text = _effective_sidecar_ocr_text(image_path, state)
    ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
    people_names = _dedupe(
        [
            str(r.get("name") or "")
            for r in list(det.get("people") or [])
            if isinstance(r, dict) and r.get("name")
        ]
    )
    object_labels = [
        str(r.get("label") or "")
        for r in list(det.get("objects") or [])
        if isinstance(r, dict) and r.get("label")
    ]
    album_title = _effective_sidecar_album_title(image_path, state)
    printed_title = _resolve_album_printed_title_hint(image_path, printed_album_title_cache)
    existing_location_name = str((dict(det.get("location") or {})).get("query") or "").strip()
    return _GpsInputs(
        detections=det,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        people_names=people_names,
        object_labels=object_labels,
        album_title=album_title,
        printed_title=printed_title,
        existing_location_name=existing_location_name,
    )


def _gps_updated_detections(
    det: dict[str, Any],
    gps_location_payload: dict[str, Any],
    gps_locations_shown: list[Any],
    gps_locations_shown_ran: bool,
) -> dict[str, Any]:
    updated = {**det}
    if gps_location_payload:
        updated["location"] = gps_location_payload
    elif "location" in updated:
        del updated["location"]
    updated["locations_shown"] = gps_locations_shown
    updated["location_shown_ran"] = gps_locations_shown_ran
    return updated


def _state_writer_kwargs(
    *,
    state: dict[str, Any],
    image_path: Path,
    person_names: list[str],
    subjects: list[str],
    title: str,
    title_source: str,
    album_title: str,
    location_payload: dict[str, Any],
    source_text: str,
    ocr_text: str,
    detections_payload: dict[str, Any],
    dc_date: Any,
    date_time_original: str,
) -> dict[str, Any]:
    return dict(
        person_names=person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=str(state.get("description") or ""),
        album_title=album_title,
        location_payload=location_payload,
        source_text=source_text,
        ocr_text=ocr_text,
        ocr_lang=str(state.get("ocr_lang") or ""),
        author_text=str(state.get("author_text") or ""),
        scene_text=str(state.get("scene_text") or ""),
        detections_payload=detections_payload,
        stitch_key=str(state.get("stitch_key") or ""),
        ocr_authority_source=str(state.get("ocr_authority_source") or ""),
        create_date=(str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
        dc_date=dc_date,
        date_time_original=date_time_original,
        ocr_ran=bool(state.get("ocr_ran")),
        people_detected=bool(state.get("people_detected")),
        people_identified=bool(state.get("people_identified")),
    )


def _refresh_gps_coords(refresh_location: dict[str, Any], review: dict[str, Any]) -> tuple[str, str]:
    refresh_gps_lat = str(refresh_location.get("gps_latitude") or "").strip()
    refresh_gps_lon = str(refresh_location.get("gps_longitude") or "").strip()
    if not refresh_gps_lat:
        refresh_gps_lat = _xmp_gps_to_decimal(review.get("gps_latitude"), axis="lat")
    if not refresh_gps_lon:
        refresh_gps_lon = _xmp_gps_to_decimal(review.get("gps_longitude"), axis="lon")
    return refresh_gps_lat, refresh_gps_lon


def _refresh_page_like(review: dict[str, Any], refresh_detections: dict[str, Any]) -> bool:
    return bool(review.get("subphotos")) or (
        str((refresh_detections.get("caption") or {}).get("effective_engine") or "").strip() == "page-summary"
    )


def _refresh_analysis_mode(existing_sidecar_state: dict | None, review: dict[str, Any]) -> str:
    refresh_subphotos = review.get("subphotos")
    return str(
        (existing_sidecar_state or {}).get("analysis_mode")
        or ("page_subphotos" if isinstance(refresh_subphotos, list) and refresh_subphotos else "single_image")
    )


def _refresh_processing_payload(
    *,
    image_path: Path,
    review: dict[str, Any],
    existing_sidecar_state: dict,
    settings_sig: str,
    current_cast_signature: str,
    effective: dict[str, Any],
    refresh_ocr_text: str,
    refresh_album_title: str,
) -> dict[str, Any]:
    stat = image_path.stat()
    return {
        "processor_signature": PROCESSOR_SIGNATURE,
        "settings_signature": settings_sig,
        "cast_store_signature": (
            current_cast_signature if bool(effective.get("enable_people", True)) else ""
        ),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "date_estimate_input_hash": _date_estimate_input_hash(refresh_ocr_text, refresh_album_title),
        "ocr_authority_signature": str(existing_sidecar_state.get("ocr_authority_signature") or ""),
        "ocr_authority_hash": str(existing_sidecar_state.get("ocr_authority_hash") or ""),
        "analysis_mode": _refresh_analysis_mode(existing_sidecar_state, review),
    }


def _refresh_text_layers(
    image_path: Path,
    review: dict[str, Any],
    refresh_ocr_text: str,
    refresh_detections: dict[str, Any],
) -> dict[str, str]:
    return _resolve_xmp_text_layers(
        image_path=image_path,
        ocr_text=refresh_ocr_text,
        page_like=_refresh_page_like(review, refresh_detections),
        ocr_authority_source=str(review.get("ocr_authority_source") or ""),
        author_text=str(review.get("author_text") or ""),
        scene_text=str(review.get("scene_text") or ""),
    )


def _refresh_xmp_title(
    image_path: Path, review: dict[str, Any], text_layers: dict[str, str]
) -> tuple[str, str]:
    return _compute_xmp_title(
        image_path=image_path,
        explicit_title=str(review.get("title") or ""),
        title_source=str(review.get("title_source") or ""),
        author_text=str(text_layers.get("author_text") or ""),
    )


def _refresh_album_title(image_path: Path, review: dict[str, Any], refresh_ocr_text: str) -> str:
    base_title = str(review.get("album_title") or "").strip() or _resolve_album_title_hint(image_path)
    return _require_album_title_for_title_page(
        image_path=image_path,
        album_title=_resolve_title_page_album_title(
            image_path=image_path,
            album_title=base_title,
            ocr_text=refresh_ocr_text,
        ),
        context="refresh",
    )


def _refresh_writer_kwargs(
    *,
    review: dict[str, Any],
    text_layers: dict[str, str],
    refresh_album_title: str,
    refresh_write_location: dict[str, Any],
    refresh_ocr_text: str,
    refresh_detections: dict[str, Any],
    refresh_dc_date: str,
    refresh_date_time_original: str,
    xmp_title: str,
    xmp_title_source: str,
    image_path: Path,
) -> dict[str, Any]:
    return dict(
        person_names=list(review.get("person_names") or []),
        subjects=list(review.get("subjects") or []),
        title=xmp_title,
        title_source=xmp_title_source,
        description=str(review.get("description") or ""),
        album_title=refresh_album_title,
        location_payload=refresh_write_location,
        source_text=_build_dc_source(refresh_album_title, image_path, _page_scan_filenames(image_path)),
        ocr_text=refresh_ocr_text,
        ocr_lang=str(review.get("ocr_lang") or ""),
        author_text=str(text_layers.get("author_text") or ""),
        scene_text=str(text_layers.get("scene_text") or ""),
        detections_payload=refresh_detections,
        stitch_key=str(review.get("stitch_key") or ""),
        ocr_authority_source=str(review.get("ocr_authority_source") or ""),
        create_date=(str(review.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
        dc_date=refresh_dc_date,
        date_time_original=refresh_date_time_original,
        ocr_ran=bool(review.get("ocr_ran")),
        people_detected=bool(review.get("people_detected")),
        people_identified=bool(review.get("people_identified")),
    )


def _refresh_write_location(
    refresh_detections: dict[str, Any],
    refresh_location: dict[str, Any],
    refresh_gps_lat: str,
    refresh_gps_lon: str,
) -> dict[str, Any]:
    refresh_write_location = dict((refresh_detections or {}).get("location") or {})
    if not refresh_write_location:
        refresh_write_location = dict(refresh_location or {})
    if refresh_gps_lat and not refresh_write_location.get("gps_latitude"):
        refresh_write_location["gps_latitude"] = refresh_gps_lat
    if refresh_gps_lon and not refresh_write_location.get("gps_longitude"):
        refresh_write_location["gps_longitude"] = refresh_gps_lon
    return refresh_write_location


def _sidecar_matches_stitched_authority(state: "_ProcessOneState", existing_ocr_hash: str) -> bool:
    sidecar_state = state.existing_sidecar_state or {}
    sidecar_source = str(sidecar_state.get("ocr_authority_source") or "").strip()
    sidecar_signature = str(sidecar_state.get("ocr_authority_signature") or "").strip()
    sidecar_hash = str(sidecar_state.get("ocr_authority_hash") or "").strip()
    has_current_authority = (
        sidecar_source == "archive_stitched"
        and bool(existing_ocr_hash)
        and _sidecar_current_for_paths(state.sidecar_path, state.multi_scan_group_paths)
    )
    matches_authority = (
        sidecar_source == "archive_stitched"
        and sidecar_signature == state.multi_scan_group_signature
        and bool(sidecar_hash)
        and sidecar_hash == existing_ocr_hash
    )
    return matches_authority or has_current_authority


@dataclass
class _ProcessOneState:
    image_path: Path
    sidecar_path: Path
    effective: dict[str, Any] = field(default_factory=dict)
    settings_sig: str = ""
    date_estimation_enabled: bool = False
    existing_xmp_people: list[str] = field(default_factory=list)

    existing_sidecar_valid: bool = False
    existing_sidecar_current: bool = False
    existing_sidecar_state: dict | None = None
    existing_sidecar_complete: bool = False
    source_refresh_required: bool = False
    date_refresh_required: bool = False
    reprocess_required: bool = False
    reprocess_reasons: list[str] = field(default_factory=list)

    location_shown_missing: bool = False
    location_shown_backfill_needed: bool = False
    location_shown_gps_dirty: bool = False

    people_update_only: bool = False
    people_matcher: Any = None
    current_cast_signature: str = ""

    gps_repair_requested: bool = False

    archive_stitched_ocr_required: bool = False
    multi_scan_group_paths: list[Path] = field(default_factory=list)
    multi_scan_group_signature: str = ""

    needs_full: bool = False
    gps_update_only: bool = False
    extra_forced: set[str] = field(default_factory=set)


class IndexRunner:
    def __init__(self, argv: list[str] | None = None) -> None:
        self.args = parse_args(argv)
        self.explicit_flags = _explicit_cli_flags(argv)
        self.requested_caption_prompt = _resolve_caption_prompt(
            str(getattr(self.args, "caption_prompt", "")),
            str(getattr(self.args, "caption_prompt_file", "")),
        )
        self.photos_root = _absolute_cli_path(self.args.photos_root)
        self.archive_set = find_archive_set_by_photos_root(self.photos_root)
        self.title_page_location = self.archive_set.title_page_location if self.archive_set is not None else None
        self.stdout_only = bool(self.args.stdout)
        self.reprocess_mode = str(self.args.reprocess_mode)
        self.force_processing = bool(self.args.force or self.stdout_only or self.reprocess_mode == "all")
        self.dry_run = bool(self.args.dry_run or self.stdout_only)
        self.shard_count = int(self.args.shard_count or 1)
        self.shard_index = int(self.args.shard_index or 0)
        self._validate_init_args()

        self.include_archive, self.include_view = self._resolve_inclusion_flags()
        self.ext_set = self._resolve_extension_set()
        self.single_photo = str(self.args.photo or "").strip()
        self.defaults = self._build_defaults()
        self._init_caches()

        self.processed = 0
        self.skipped = 0
        self.failures = 0
        self.completed_times: list[float] = []

        self.files: list[Path] = []
        self.batch_lock_path: Path | None = None
        self.allow_concurrent_shards = False

    def _validate_init_args(self) -> None:
        if not self.photos_root.is_dir():
            raise SystemExit(f"Photo root is not a directory: {self.photos_root}")
        if self.shard_count < 1:
            raise SystemExit("--shard-count must be at least 1")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise SystemExit("--shard-index must be between 0 and --shard-count - 1")

    def _resolve_inclusion_flags(self) -> tuple[bool, bool]:
        include_archive = bool(self.args.include_archive)
        include_view = bool(self.args.include_view)
        if not include_archive and not include_view:
            return True, False
        return include_archive, include_view

    def _resolve_extension_set(self) -> set[str]:
        ext_set = {
            (item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}")
            for item in str(self.args.extensions or "").split(",")
            if item.strip()
        }
        return ext_set or set(IMAGE_EXTENSIONS)

    def _build_defaults(self) -> dict[str, Any]:
        caption_params = load_params("ai-index/metadata/params.toml").values
        is_lmstudio = str(self.args.caption_engine) == "lmstudio"
        max_tokens = int(self.args.caption_max_tokens)
        if "--caption-max-tokens" not in self.explicit_flags and is_lmstudio:
            max_tokens = int(caption_params.get("max_tokens", DEFAULT_LMSTUDIO_MAX_NEW_TOKENS))
        temperature = float(self.args.caption_temperature)
        if "--caption-temperature" not in self.explicit_flags and is_lmstudio:
            temperature = float(caption_params.get("temperature", self.args.caption_temperature))
        max_edge = int(self.args.caption_max_edge)
        if "--caption-max-edge" not in self.explicit_flags and is_lmstudio:
            max_edge = int(caption_params.get("max_image_edge", self.args.caption_max_edge))
        return {
            "skip": False,
            "enable_people": not bool(self.args.disable_people),
            "enable_objects": not bool(self.args.disable_objects),
            "ocr_engine": str(self.args.ocr_engine),
            "ocr_lang": str(self.args.ocr_lang),
            "ocr_model": str(self.args.ocr_model),
            "caption_engine": str(self.args.caption_engine),
            "caption_model": resolve_caption_model(str(self.args.caption_engine), str(self.args.caption_model)),
            "caption_prompt": str(self.requested_caption_prompt),
            "caption_max_tokens": max_tokens,
            "caption_temperature": temperature,
            "caption_max_edge": max_edge,
            "caption_thinking": bool(caption_params.get("thinking", False)),
            "lmstudio_base_url": normalize_lmstudio_base_url(str(self.args.lmstudio_base_url)),
            "people_threshold": float(self.args.people_threshold),
            "object_threshold": float(self.args.object_threshold),
            "min_face_size": int(self.args.min_face_size),
            "model": str(self.args.model),
        }

    def _init_caches(self) -> None:
        self.archive_settings_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.people_matcher_cache: dict[tuple[str, float, int], Any] = {}
        self.object_detector_cache: dict[tuple[str, float], Any] = {}
        self.ocr_engine_cache: dict[tuple[str, str, str, str], OCREngine] = {}
        self.caption_engine_cache: dict[tuple[str, str, str, int, float, str, int, bool], CaptionEngine] = {}
        self.date_engine_cache: dict[tuple[str, str, int, float, str], DateEstimateEngine] = {}
        self.metadata_engine_cache: dict[tuple[str, str, str], MetadataEngine] = {}
        self.archive_scan_ocr_cache: dict[str, ArchiveScanOCRAuthority] = {}
        self.printed_album_title_cache: dict[str, str] = {}
        self.geocoder = NominatimGeocoder()
        self.stitch_cap_td = tempfile.TemporaryDirectory(prefix="imago-stitch-cap-")
        self.stitch_cap_dir = Path(self.stitch_cap_td.name)

    def emit_info(self, message: str) -> None:
        if not self.stdout_only:
            print(message)

    def emit_error(self, message: str) -> None:
        print(message, file=sys.stderr if self.stdout_only else sys.stdout, flush=True)

    def _caption_key_from_effective(
        self, effective: dict[str, Any]
    ) -> tuple[str, str, str, int, float, str, int, bool]:
        return (
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
            str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
            int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
            float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
            str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
            int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
            bool(effective.get("caption_thinking", self.defaults["caption_thinking"])),
        )

    def _get_caption_engine_for_key(
        self,
        caption_key: tuple[str, str, str, int, float, str, int, bool],
        effective: dict[str, Any],
        *,
        stream: bool = True,
    ) -> CaptionEngine:
        caption_engine = self.caption_engine_cache.get(caption_key)
        if caption_engine is None:
            caption_engine = _init_caption_engine(
                engine=caption_key[0],
                model_name=caption_key[1],
                caption_prompt=caption_key[2],
                max_tokens=int(caption_key[3]),
                temperature=float(caption_key[4]),
                lmstudio_base_url=caption_key[5],
                max_image_edge=int(caption_key[6]),
                stream=stream,
                thinking=bool(caption_key[7]),
                override_sources=dict(effective.get("_override_sources") or {}),
            )
            self.caption_engine_cache[caption_key] = caption_engine
        caption_engine.override_sources = dict(effective.get("_override_sources") or {})
        return caption_engine

    def _get_caption_engine(self, effective: dict[str, Any]) -> CaptionEngine:
        return self._get_caption_engine_for_key(
            self._caption_key_from_effective(effective), effective
        )

    def _get_date_engine(self, effective: dict[str, Any]) -> DateEstimateEngine:
        date_key = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
            int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
            0.0,
            str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
        )
        date_engine = self.date_engine_cache.get(date_key)
        if date_engine is None:
            date_engine = _init_date_engine(
                engine=date_key[0],
                model_name=date_key[1],
                max_tokens=int(date_key[2]),
                temperature=0.0,
                lmstudio_base_url=date_key[4],
            )
            self.date_engine_cache[date_key] = date_engine
        return date_engine

    def _get_metadata_engine(self, effective: dict[str, Any]) -> MetadataEngine:
        engine = str(effective.get("caption_engine", self.defaults["caption_engine"]))
        model = str(effective.get("caption_model", self.defaults["caption_model"]))
        base_url = str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"]))
        metadata_key = (engine, model, base_url)
        cached = self.metadata_engine_cache.get(metadata_key)
        if cached is None:
            cached = MetadataEngine(
                engine=engine,
                model_name=model,
                lmstudio_base_url=base_url,
            )
            self.metadata_engine_cache[metadata_key] = cached
        return cached

    def _record_success(self, idx: int, file_start: float) -> None:
        self.processed += 1
        self.completed_times.append(time.monotonic() - file_start)

    def _record_failure(self, idx: int, image_path: Path, exc: Exception) -> None:
        self.failures += 1
        self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def run(self) -> int:
        setup_result = self._setup()
        if setup_result is not None:
            return setup_result
        for idx, image_path in enumerate(self.files, 1):
            self._process_one(idx, image_path)
        return self._summarize()

    def _setup(self) -> int | None:
        if self.single_photo:
            if self.shard_count > 1:
                raise SystemExit("--shard-count > 1 is only supported for multi-photo discovery runs")
            photo_path = _absolute_cli_path(self.single_photo)
            if not photo_path.is_file():
                raise SystemExit(f"Photo not found: {photo_path}")
            self.files = [photo_path]
            self.force_processing = True
        else:
            self.files = discover_images(
                self.photos_root,
                include_archive=self.include_archive,
                include_view=self.include_view,
                extensions=self.ext_set,
            )
            album_filter = str(self.args.album or "").strip()
            if album_filter:
                album_lower = album_filter.casefold()
                self.files = [f for f in self.files if album_lower in f.parent.name.casefold()]
            self.files = _coalesce_archive_processing_files(self.files)
            photo_offset = int(self.args.photo_offset or 0)
            if photo_offset > 0:
                self.files = self.files[photo_offset:]
            if self.args.max_images and self.args.max_images > 0:
                self.files = self.files[: int(self.args.max_images)]
            self.files = _apply_shard(self.files, self.shard_count, self.shard_index)

        original_file_count = len(self.files)
        self.files = _expand_album_title_dependencies(self.files, self.ext_set)
        if not self.single_photo:
            self.files = _filter_files_by_tree(
                self.files,
                include_archive=self.include_archive,
                include_view=self.include_view,
            )

        self.emit_info(f"Discovered {len(self.files)} image files")
        if len(self.files) > original_file_count:
            self.emit_info(f"Added {len(self.files) - original_file_count} title-page dependency files")
        if not self.files:
            return 0

        self.allow_concurrent_shards = not self.single_photo and self.shard_count > 1
        if not self.single_photo and not self.allow_concurrent_shards:
            try:
                self.batch_lock_path = _acquire_batch_processing_lock(self.photos_root)
            except RuntimeError as exc:
                self.emit_error(str(exc))
                return 1

        return None

    def _summarize(self) -> int:
        stitch_failures = 0
        if bool(getattr(self.args, "stitch_scans", False)):
            self.emit_info("Scan stitch pass skipped: archive scan OCR stitching now happens during normal processing.")

        if not self.stdout_only:
            print("\nSummary")
            print(f"- Processed: {self.processed}")
            print(f"- Skipped:   {self.skipped}")
            print(f"- Failed:    {self.failures + stitch_failures}")
        _release_batch_processing_lock(self.batch_lock_path)
        self.stitch_cap_td.cleanup()
        return 1 if (self.failures or stitch_failures) else 0

    def _resolve_effective_settings(self, image_path: Path) -> tuple[dict[str, Any], str, bool]:
        settings_file, loaded_settings = self._load_archive_settings(image_path)
        effective = resolve_effective_settings(
            image_path,
            defaults=self.defaults,
            loaded=loaded_settings,
        )
        self._apply_cli_overrides(effective)
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
        )
        effective["_override_sources"] = self._build_override_sources(effective, loaded_settings, settings_file)
        settings_sig = _settings_signature(effective)
        date_estimation_enabled = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])).strip().lower() == "lmstudio"
        )
        return effective, settings_sig, date_estimation_enabled

    def _load_archive_settings(self, image_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
        archive_dir = find_archive_dir_for_image(image_path)
        if archive_dir is None or self.args.ignore_render_settings:
            return None, None
        key = str(archive_dir.resolve())
        cached = self.archive_settings_cache.get(key)
        if cached is None:
            path, payload = load_render_settings(
                archive_dir,
                defaults=self.defaults,
                create=False,
            )
            cached = (path, payload)
            self.archive_settings_cache[key] = cached
        return cached

    def _apply_cli_overrides(self, effective: dict[str, Any]) -> None:
        if self.args.disable_people:
            effective["enable_people"] = False
        if self.args.disable_objects:
            effective["enable_objects"] = False
        for flag, key, attr, converter in _SIMPLE_CLI_OVERRIDES:
            if flag in self.explicit_flags:
                effective[key] = converter(getattr(self.args, attr))
        if _CAPTION_PROMPT_OVERRIDE_FLAGS & self.explicit_flags:
            effective["caption_prompt"] = str(self.requested_caption_prompt)
        if "--lmstudio-base-url" in self.explicit_flags:
            effective["lmstudio_base_url"] = normalize_lmstudio_base_url(str(self.args.lmstudio_base_url))

    def _build_override_sources(
        self,
        effective: dict[str, Any],
        loaded_settings: dict[str, Any] | None,
        settings_file: Path | None,
    ) -> dict[str, str]:
        override_sources: dict[str, str] = {}
        for key, flags in _OVERRIDE_SOURCE_FLAGS:
            if flags & self.explicit_flags:
                override_sources[key] = "cli"
            elif loaded_settings is not None and effective.get(key) != self.defaults.get(key):
                override_sources[key] = f"render_settings:{settings_file}"
        return override_sources

    def _get_people_matcher_and_signature(self, effective: dict[str, Any]) -> tuple[Any, str]:
        if not bool(effective.get("enable_people", True)):
            return None, ""
        people_key = (
            str(Path(self.args.cast_store).resolve()),
            float(effective.get("people_threshold", self.defaults["people_threshold"])),
            int(effective.get("min_face_size", self.defaults["min_face_size"])),
        )
        people_matcher = self.people_matcher_cache.get(people_key)
        if people_matcher is None:
            people_matcher = _init_people_matcher(
                cast_store=Path(self.args.cast_store),
                min_similarity=float(people_key[1]),
                min_face_size=int(people_key[2]),
            )
            self.people_matcher_cache[people_key] = people_matcher
        return people_matcher, self._people_invalidation_signature(people_matcher)

    def _people_invalidation_signature(self, people_matcher: Any) -> str:
        reviewed_signature_fn = getattr(people_matcher, "reviewed_identity_signature", None)
        if callable(reviewed_signature_fn):
            reviewed_signature = reviewed_signature_fn()
            if isinstance(reviewed_signature, str) and reviewed_signature.strip():
                return reviewed_signature
        return str(people_matcher.store_signature())

    # ── Per-image dispatch ──────────────────────────────────────────────────

    def _process_one(self, idx: int, image_path: Path) -> None:
        sidecar_path = image_path.with_suffix(".xmp")
        state = _ProcessOneState(
            image_path=image_path,
            sidecar_path=sidecar_path,
            existing_xmp_people=read_person_in_image(sidecar_path),
        )
        state.effective, state.settings_sig, state.date_estimation_enabled = self._resolve_effective_settings(
            image_path
        )

        self._evaluate_existing_sidecar(state)
        self._evaluate_locations_shown(state)
        self._evaluate_people_update(state)
        state.gps_repair_requested = _is_gps_repair_requested(state)

        if self._can_skip_current(state):
            self._emit_skip(idx, image_path, "current xmp")
            return

        if state.people_matcher is None:
            state.people_matcher, state.current_cast_signature = self._get_people_matcher_and_signature(
                state.effective
            )

        self._evaluate_multi_scan(state)
        self._evaluate_extra_reprocess_reasons(state)
        self._decide_processing_mode(state)

        if self._should_skip_after_decision(state):
            self._emit_skip(idx, image_path, "")
            return
        if bool(state.effective.get("skip", False)):
            self._emit_skip(idx, image_path, "render_settings skip=true")
            return
        if not self._matches_reprocess_mode(state):
            self._emit_skip(idx, image_path, f"reprocess_mode={self.reprocess_mode}")
            return

        self._emit_reprocess_status(idx, state)
        self._dispatch_with_lock(idx, state)

    def _emit_skip(self, idx: int, image_path: Path, reason: str) -> None:
        self.skipped += 1
        if self.args.verbose and not self.stdout_only:
            suffix = f" ({reason})" if reason else ""
            print(f"[{idx}/{len(self.files)}] skip  {image_path.name}{suffix}")

    def _evaluate_existing_sidecar(self, state: _ProcessOneState) -> None:
        image_path = state.image_path
        sidecar_path = state.sidecar_path
        effective = state.effective
        state.existing_sidecar_valid = has_valid_sidecar(image_path)
        state.existing_sidecar_current = (
            has_current_sidecar(image_path) if state.existing_sidecar_valid else False
        )
        if state.existing_sidecar_valid:
            state.existing_sidecar_state = read_ai_sidecar_state(sidecar_path)
        if state.existing_sidecar_valid and not state.existing_sidecar_current:
            state.reprocess_reasons.append("sidecar_older_than_image")
        if _sidecar_has_lmstudio_caption_error(state.existing_sidecar_state):
            state.reprocess_required = True
            state.reprocess_reasons.append("lmstudio_caption_error")
        if state.existing_sidecar_valid:
            state.existing_sidecar_complete = sidecar_has_expected_ai_fields(
                sidecar_path,
                enable_people=bool(effective.get("enable_people", True)),
                enable_objects=bool(effective.get("enable_objects", True)),
                ocr_engine=str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
                caption_engine=str(effective.get("caption_engine", self.defaults["caption_engine"])),
            )
            state.source_refresh_required = _dc_source_needs_refresh(image_path, state.existing_sidecar_state)
            if state.source_refresh_required:
                state.reprocess_reasons.append("dc_source_stale")
            state.date_refresh_required = _dc_date_needs_refresh(
                image_path,
                state.existing_sidecar_state,
                enabled=state.date_estimation_enabled,
            )
            if state.date_refresh_required:
                state.reprocess_reasons.append("timeline_date_missing")

    def _evaluate_locations_shown(self, state: _ProcessOneState) -> None:
        if not (
            state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
            and _caption_engine_lower(state.effective, self.defaults) == "lmstudio"
        ):
            return
        det = state.existing_sidecar_state.get("detections") or {}
        detected_locations = list(det.get("locations_shown") or []) if isinstance(det, dict) else []
        written_locations = read_locations_shown(state.sidecar_path)
        location_shown_ran = isinstance(det, dict) and det.get("location_shown_ran") is True
        state.location_shown_missing = bool(written_locations) is False and (
            location_shown_ran or bool(detected_locations)
        )
        state.location_shown_backfill_needed = (
            not location_shown_ran
            and not detected_locations
            and not written_locations
            and isinstance(det, dict)
            and bool(det.get("location"))
        )
        state.location_shown_gps_dirty = _has_legacy_ai_locations_shown_gps(state.existing_sidecar_state)

    def _evaluate_people_update(self, state: _ProcessOneState) -> None:
        if state.existing_sidecar_state is None or not bool(state.effective.get("enable_people", True)):
            return
        old_cast_signature = str(state.existing_sidecar_state.get("cast_store_signature") or "")
        if not (old_cast_signature and _sidecar_has_people_to_refresh(state.existing_sidecar_state)):
            return
        state.people_matcher, state.current_cast_signature = self._get_people_matcher_and_signature(
            state.effective
        )
        if old_cast_signature != state.current_cast_signature:
            state.people_update_only = True
            state.reprocess_reasons.append("cast_store_signature_changed")

    def _can_skip_current(self, state: _ProcessOneState) -> bool:
        return (
            state.existing_sidecar_current
            and state.existing_sidecar_complete
            and not state.reprocess_required
            and not state.source_refresh_required
            and not state.date_refresh_required
            and not self.force_processing
            and self.reprocess_mode != "gps"
            and not state.people_update_only
            and not state.gps_repair_requested
        )

    def _evaluate_multi_scan(self, state: _ProcessOneState) -> None:
        state.multi_scan_group_paths = _scan_group_paths(state.image_path)
        state.archive_stitched_ocr_required = (
            str(state.effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower() != "none"
            and len(state.multi_scan_group_paths) > 1
        )
        state.multi_scan_group_signature = (
            _scan_group_signature(state.multi_scan_group_paths) if state.archive_stitched_ocr_required else ""
        )

    def _evaluate_extra_reprocess_reasons(self, state: _ProcessOneState) -> None:
        if state.existing_sidecar_valid and not state.existing_sidecar_complete:
            state.reprocess_required = True
            state.reprocess_reasons.append("sidecar_incomplete")
        existing_album_title = str((state.existing_sidecar_state or {}).get("album_title") or "").strip()
        if not existing_album_title and (
            _is_album_title_source_candidate(state.image_path)
            or _resolve_album_title_from_sidecars(state.image_path)
        ):
            state.reprocess_required = True
            state.reprocess_reasons.append("missing_album_title")
        if state.archive_stitched_ocr_required and not _sidecar_matches_stitched_authority(
            state, _hash_text(str((state.existing_sidecar_state or {}).get("ocr_text") or ""))
        ):
            state.reprocess_required = True
            state.reprocess_reasons.append("missing_stitched_authority")
        if state.existing_sidecar_state is not None:
            old_sig = str(state.existing_sidecar_state.get("settings_signature") or "")
            if old_sig != state.settings_sig and not (
                state.existing_sidecar_current and state.existing_sidecar_complete
            ):
                state.reprocess_required = True
                state.reprocess_reasons.append("settings_signature_mismatch")

    def _decide_processing_mode(self, state: _ProcessOneState) -> None:
        state.needs_full = needs_processing(
            state.image_path,
            state.existing_sidecar_state,
            self.force_processing and not state.gps_repair_requested,
            reprocess_required=state.reprocess_required,
        )
        if state.gps_repair_requested:
            state.gps_update_only = True
            self._record_location_shown_reasons(state)
        if not state.gps_update_only and self._gps_mode_eligible(state):
            state.gps_update_only = True
        if not state.gps_update_only and self._lmstudio_location_repair_eligible(state):
            if state.location_shown_missing:
                state.gps_update_only = True
            if state.location_shown_gps_dirty:
                state.gps_update_only = True
            self._record_location_shown_reasons(state)
        if not state.gps_update_only and state.people_update_only and state.location_shown_backfill_needed:
            state.gps_update_only = True
            state.reprocess_reasons.append("missing_location_shown")

    def _gps_mode_eligible(self, state: _ProcessOneState) -> bool:
        return (
            self.reprocess_mode == "gps"
            and not state.needs_full
            and state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
        )

    def _lmstudio_location_repair_eligible(self, state: _ProcessOneState) -> bool:
        return (
            not state.needs_full
            and not state.source_refresh_required
            and not state.date_refresh_required
            and state.existing_sidecar_complete
            and state.existing_sidecar_state is not None
            and _caption_engine_lower(state.effective, self.defaults) == "lmstudio"
        )

    @staticmethod
    def _record_location_shown_reasons(state: _ProcessOneState) -> None:
        if state.location_shown_missing:
            state.reprocess_reasons.append("missing_location_shown")
        if state.location_shown_gps_dirty:
            state.reprocess_reasons.append("location_shown_ai_gps_stale")

    def _should_skip_after_decision(self, state: _ProcessOneState) -> bool:
        return (
            not state.needs_full
            and not state.people_update_only
            and not state.gps_update_only
            and not isinstance(state.existing_sidecar_state, dict)
        )

    def _matches_reprocess_mode(self, state: _ProcessOneState) -> bool:
        mode = self.reprocess_mode
        if mode in ("unprocessed", "all"):
            return True
        reasons_set = set(state.reprocess_reasons)
        if mode == "new_only":
            return state.existing_sidecar_state is None
        if mode == "errors_only":
            return bool(reasons_set & {"lmstudio_caption_error", "sidecar_incomplete"})
        if mode == "outdated":
            return "sidecar_older_than_image" in reasons_set
        if mode == "cast_changed":
            return "cast_store_signature_changed" in reasons_set
        if mode == "gps":
            return state.gps_update_only
        return True

    def _emit_reprocess_status(self, idx: int, state: _ProcessOneState) -> None:
        if not state.existing_sidecar_valid or self.stdout_only:
            return
        reason_text = _format_reprocess_reasons(state.reprocess_reasons)
        if not reason_text:
            return
        prefix = f"  [{idx}/{len(self.files)}]  {state.image_path.name}"
        if state.needs_full:
            print(f"{prefix}  [reprocess: {reason_text}]", flush=True)
        elif state.people_update_only:
            print(f"{prefix}  [update: {reason_text}]", flush=True)
        elif state.source_refresh_required or state.date_refresh_required:
            print(f"{prefix}  [refresh: {reason_text}]", flush=True)

    def _dispatch_with_lock(self, idx: int, state: _ProcessOneState) -> None:
        try:
            lock_path = _acquire_image_processing_lock(state.image_path)
        except RuntimeError as exc:
            if self.allow_concurrent_shards and "already processing" in str(exc):
                self._emit_skip(idx, state.image_path, str(exc))
            else:
                self._record_failure(idx, state.image_path, exc)
            return
        try:
            self._dispatch_processing(idx, state)
        finally:
            _release_image_processing_lock(lock_path)

    def _dispatch_processing(self, idx: int, state: _ProcessOneState) -> None:
        if not state.needs_full and not state.people_update_only and not state.gps_update_only:
            if isinstance(state.existing_sidecar_state, dict):
                self._process_refresh(
                    idx,
                    state.image_path,
                    state.sidecar_path,
                    state.effective,
                    state.settings_sig,
                    state.date_estimation_enabled,
                    state.existing_sidecar_state,
                    state.current_cast_signature,
                )
            return

        if not state.needs_full and state.people_update_only:
            if isinstance(state.existing_sidecar_state, dict):
                state.extra_forced.add("people")
            state.needs_full = True
        if not state.needs_full and state.gps_update_only:
            if not isinstance(state.existing_sidecar_state, dict):
                return
            state.extra_forced.add("metadata")
            state.needs_full = True

        self._process_full(
            idx,
            state.image_path,
            state.sidecar_path,
            state.effective,
            state.settings_sig,
            state.date_estimation_enabled,
            state.existing_sidecar_state,
            state.existing_xmp_people,
            state.people_matcher,
            state.current_cast_signature,
            state.archive_stitched_ocr_required,
            state.multi_scan_group_paths,
            state.multi_scan_group_signature,
            extra_forced_steps=state.extra_forced or None,
        )

    # ── Refresh fast-path ───────────────────────────────────────────────────

    def _process_refresh(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        current_cast_signature: str,
    ) -> None:
        if not isinstance(existing_sidecar_state, dict):
            return
        file_start = time.monotonic()
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
        try:
            review = load_ai_xmp_review(sidecar_path)
            review_dict = review if isinstance(review, dict) else None
            refresh_ocr_text = _effective_sidecar_ocr_text(image_path, review_dict)
            refresh_location = _effective_sidecar_location_payload(image_path, review_dict)
            refresh_detections = (
                dict(review.get("detections") or {}) if isinstance(review.get("detections"), dict) else {}
            )
            if refresh_location:
                refresh_detections["location"] = refresh_location
            refresh_location, refresh_detections = _apply_title_page_location_config(
                image_path=image_path,
                location_payload=refresh_location,
                detections_payload=refresh_detections,
                title_page_location=self.title_page_location,
            )
            if not self.dry_run:
                self._write_refresh_payload(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    review=review,
                    effective=effective,
                    settings_sig=settings_sig,
                    date_estimation_enabled=date_estimation_enabled,
                    existing_sidecar_state=existing_sidecar_state,
                    current_cast_signature=current_cast_signature,
                    refresh_ocr_text=refresh_ocr_text,
                    refresh_location=refresh_location,
                    refresh_detections=refresh_detections,
                    prompt_debug=prompt_debug,
                )
                _append_xmp_job_artifact(image_path, sidecar_path)
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            if not self.stdout_only:
                eta_str = _format_eta(self.completed_times, len(self.files) - idx)
                eta_part = f"  {eta_str}" if eta_str else ""
                print(
                    f"[{idx}/{len(self.files)}]{eta_part}  ok    {image_path.name}  [refresh]",
                    flush=True,
                )
        except Exception as exc:
            self.failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _write_refresh_payload(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        review: dict,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict,
        current_cast_signature: str,
        refresh_ocr_text: str,
        refresh_location: dict[str, Any],
        refresh_detections: dict[str, Any],
        prompt_debug: PromptDebugSession,
    ) -> None:
        refresh_gps_lat, refresh_gps_lon = _refresh_gps_coords(refresh_location, review)
        text_layers = _refresh_text_layers(image_path, review, refresh_ocr_text, refresh_detections)
        xmp_title, xmp_title_source = _refresh_xmp_title(image_path, review, text_layers)
        refresh_album_title = _refresh_album_title(image_path, review, refresh_ocr_text)
        date_engine = self._refresh_date_engine(effective, date_estimation_enabled, review)
        refresh_dc_date = _resolve_dc_date(
            existing_dc_date=_dc_date_value(review),
            ocr_text=refresh_ocr_text,
            album_title=refresh_album_title,
            image_path=image_path,
            date_engine=date_engine,
            prompt_debug=prompt_debug,
        )
        refresh_date_time_original = _resolve_date_time_original(
            dc_date=refresh_dc_date,
            date_time_original=str(review.get("date_time_original") or ""),
        )
        refresh_detections["processing"] = _refresh_processing_payload(
            image_path=image_path,
            review=review,
            existing_sidecar_state=existing_sidecar_state,
            settings_sig=settings_sig,
            current_cast_signature=current_cast_signature,
            effective=effective,
            refresh_ocr_text=refresh_ocr_text,
            refresh_album_title=refresh_album_title,
        )
        refresh_write_location = _refresh_write_location(
            refresh_detections, refresh_location, refresh_gps_lat, refresh_gps_lon
        )
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            **_refresh_writer_kwargs(
                review=review,
                text_layers=text_layers,
                refresh_album_title=refresh_album_title,
                refresh_write_location=refresh_write_location,
                refresh_ocr_text=refresh_ocr_text,
                refresh_detections=refresh_detections,
                refresh_dc_date=refresh_dc_date,
                refresh_date_time_original=refresh_date_time_original,
                xmp_title=xmp_title,
                xmp_title_source=xmp_title_source,
                image_path=image_path,
            ),
            title_page_location=self.title_page_location,
        )

    def _refresh_date_engine(
        self,
        effective: dict[str, Any],
        date_estimation_enabled: bool,
        existing: dict[str, Any],
    ) -> DateEstimateEngine | None:
        if date_estimation_enabled and not _has_dc_date(_dc_date_value(existing)):
            return self._get_date_engine(effective)
        return None

    # ── People-update fast-path (deleted — routed via _process_full + StepRunner) ──

    def _process_people_update(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        chain_gps: bool,
        *,
        preserve_existing_xmp_people: bool = True,
        raise_on_error: bool = False,
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return

        file_start = time.monotonic()
        pu_inputs = _pu_inputs_from_state(image_path, state)
        prefix = self._format_progress_prefix(idx, image_path)
        print(prefix, flush=True)
        _pu_stop, _pu_step = _progress_ticker(prefix)

        try:
            _pu_step("people")
            pu_people_matches, pu_faces_detected = _pu_match_people(
                people_matcher, image_path, pu_inputs.existing_ocr_text
            )
            pu_people_match_names = _dedupe([r.name for r in pu_people_matches])
            _pu_step(_format_people_step_label("people", pu_people_match_names))
            pu_person_names = (
                _dedupe(pu_people_match_names + existing_xmp_people)
                if preserve_existing_xmp_people
                else pu_people_match_names
            )
            pu_album_title = (
                _resolve_album_title_hint(image_path)
                or _effective_sidecar_album_title(image_path, state)
            )
            pu_printed_title = _resolve_album_printed_title_hint(image_path, self.printed_album_title_cache)
            pu_people_payload = _serialize_people_matches(pu_people_matches)
            people_names_changed = pu_person_names != existing_xmp_people

            if people_names_changed:
                pu_updated_det, pu_faces_detected, pu_prompt_debug = self._pu_recompute_caption(
                    image_path=image_path,
                    effective=effective,
                    people_matcher=people_matcher,
                    pu_inputs=pu_inputs,
                    pu_people_matches=pu_people_matches,
                    pu_person_names=pu_person_names,
                    pu_album_title=pu_album_title,
                    pu_printed_title=pu_printed_title,
                    pu_people_payload=pu_people_payload,
                    step_fn=_pu_step,
                )
            else:
                pu_updated_det = {
                    **pu_inputs.detections,
                    "people": pu_people_payload or pu_inputs.existing_people_rows,
                    "caption": pu_inputs.existing_caption_payload,
                }
                pu_prompt_debug = None

            pu_subjects = _dedupe(
                pu_inputs.existing_object_labels
                + pu_inputs.existing_ocr_keywords
                + ([pu_album_title] if pu_album_title else [])
            )
            pu_people_detected = pu_faces_detected > 0 or len(pu_person_names) > 0
            pu_people_identified = len(pu_person_names) > 0

            if not self.dry_run:
                self._write_pu_payload(
                    sidecar_path=sidecar_path,
                    image_path=image_path,
                    state=state,
                    effective=effective,
                    date_estimation_enabled=date_estimation_enabled,
                    people_matcher=people_matcher,
                    pu_inputs=pu_inputs,
                    pu_album_title=pu_album_title,
                    pu_person_names=pu_person_names,
                    pu_subjects=pu_subjects,
                    pu_updated_det=pu_updated_det,
                    pu_people_detected=pu_people_detected,
                    pu_people_identified=pu_people_identified,
                    pu_prompt_debug=pu_prompt_debug,
                )

            _pu_stop()
            if not chain_gps:
                self.processed += 1
                self.completed_times.append(time.monotonic() - file_start)
                self._emit_ok(idx, image_path)
        except Exception as exc:
            self.failures += 1
            _pu_stop()
            if raise_on_error:
                raise
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _pu_recompute_caption(
        self,
        *,
        image_path: Path,
        effective: dict[str, Any],
        people_matcher: Any,
        pu_inputs: "_PeopleUpdateInputs",
        pu_people_matches: list[Any],
        pu_person_names: list[str],
        pu_album_title: str,
        pu_printed_title: str,
        pu_people_payload: list[Any],
        step_fn: Any,
    ) -> tuple[dict[str, Any], int, PromptDebugSession]:
        caption_key = self._caption_key_from_effective(effective)
        pu_caption_engine = self._get_caption_engine_for_key(caption_key, effective)
        pu_people_positions = _compute_people_positions(pu_people_matches, image_path)
        step_fn("caption")
        pu_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
        with _prepare_ai_model_image(image_path) as pu_model_path:
            pu_caption_out = pu_caption_engine.generate(
                image_path=pu_model_path,
                people=pu_person_names,
                objects=pu_inputs.existing_object_labels,
                ocr_text=pu_inputs.existing_ocr_text,
                source_path=image_path,
                album_title=pu_album_title,
                printed_album_title=pu_printed_title,
                people_positions=pu_people_positions,
                debug_recorder=pu_prompt_debug.record,
                debug_step="caption_refresh",
            )
            pu_faces_detected = _people_matcher_faces(people_matcher)
            pu_local_people_present, pu_local_estimated_people_count = _estimate_people_from_detections(
                people_matches=pu_people_matches,
                people_names=pu_person_names,
                object_labels=pu_inputs.existing_object_labels,
                faces_detected=pu_faces_detected,
            )
            pu_people_present, pu_estimated_people_count = _resolve_people_count_metadata(
                requested_caption_engine=str(caption_key[0]),
                caption_engine=pu_caption_engine,
                model_image_path=pu_model_path,
                people=pu_person_names,
                objects=pu_inputs.existing_object_labels,
                ocr_text=pu_inputs.existing_ocr_text,
                source_path=image_path,
                album_title=pu_album_title,
                printed_album_title=pu_printed_title,
                people_positions=pu_people_positions,
                local_people_present=pu_local_people_present,
                local_estimated_people_count=pu_local_estimated_people_count,
                prompt_debug=pu_prompt_debug,
                debug_step="people_count_refresh",
            )
        _emit_prompt_debug_artifact(pu_prompt_debug, dry_run=self.dry_run)
        pu_caption_payload = _build_caption_metadata(
            requested_engine=str(caption_key[0]),
            effective_engine=str(pu_caption_out.engine),
            fallback=bool(pu_caption_out.fallback),
            error=str(pu_caption_out.error or ""),
            engine_error=str(getattr(pu_caption_out, "engine_error", "") or ""),
            model=str(caption_key[1] if caption_key[0] in {"local", "lmstudio"} else ""),
            people_present=pu_people_present,
            estimated_people_count=pu_estimated_people_count,
        )
        pu_ocr_model = self._pu_resolved_ocr_model(pu_inputs.detections, effective)
        pu_caption_model = (
            str(pu_caption_engine.effective_model_name)
            if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
            else ""
        )
        pu_updated_det = _refresh_detection_model_metadata(
            {
                **pu_inputs.detections,
                "people": pu_people_payload,
                "caption": pu_caption_payload,
            },
            ocr_model=pu_ocr_model,
            caption_model=pu_caption_model,
        )
        return pu_updated_det, pu_faces_detected, pu_prompt_debug

    def _pu_resolved_ocr_model(self, det: dict[str, Any], effective: dict[str, Any]) -> str:
        existing = dict(det.get("ocr") or {}).get("model")
        if existing:
            return str(existing)
        ocr_engine = str(effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower()
        if ocr_engine in {"local", "lmstudio"}:
            return str(effective.get("ocr_model", self.defaults["ocr_model"]))
        return ""

    def _write_pu_payload(
        self,
        *,
        sidecar_path: Path,
        image_path: Path,
        state: dict,
        effective: dict[str, Any],
        date_estimation_enabled: bool,
        people_matcher: Any,
        pu_inputs: "_PeopleUpdateInputs",
        pu_album_title: str,
        pu_person_names: list[str],
        pu_subjects: list[str],
        pu_updated_det: dict[str, Any],
        pu_people_detected: bool,
        pu_people_identified: bool,
        pu_prompt_debug: PromptDebugSession | None,
    ) -> None:
        pu_album_title = _require_album_title_for_title_page(
            image_path=image_path,
            album_title=_resolve_title_page_album_title(
                image_path=image_path,
                album_title=pu_album_title,
                ocr_text=pu_inputs.existing_ocr_text,
            ),
            context="people update",
        )
        date_engine = self._refresh_date_engine(effective, date_estimation_enabled, state)
        pu_dc_date = _resolve_dc_date(
            existing_dc_date=_dc_date_value(state),
            ocr_text=pu_inputs.existing_ocr_text,
            album_title=pu_album_title,
            image_path=image_path,
            date_engine=date_engine,
            prompt_debug=pu_prompt_debug,
        )
        pu_date_time_original = _resolve_date_time_original(
            dc_date=pu_dc_date,
            date_time_original=str(state.get("date_time_original") or ""),
        )
        pu_source_text = _build_dc_source(pu_album_title, image_path, _page_scan_filenames(image_path))
        pu_page_like = (
            str((pu_updated_det.get("caption") or {}).get("effective_engine") or "").strip() == "page-summary"
        )
        text_layers = _resolve_xmp_text_layers(
            image_path=image_path,
            ocr_text=pu_inputs.existing_ocr_text,
            page_like=pu_page_like,
            ocr_authority_source=str(state.get("ocr_authority_source") or ""),
            author_text=str(state.get("author_text") or ""),
            scene_text=str(state.get("scene_text") or ""),
        )
        xmp_title, xmp_title_source = _compute_xmp_title(
            image_path=image_path,
            explicit_title=str(state.get("title") or ""),
            title_source=str(state.get("title_source") or ""),
            author_text=str(text_layers.get("author_text") or ""),
        )
        current_cast_signature = self._people_invalidation_signature(people_matcher)
        pu_updated_det = _pu_finalize_detections(
            pu_updated_det,
            existing_location=pu_inputs.existing_location,
            cast_store_signature=current_cast_signature,
            ocr_text=pu_inputs.existing_ocr_text,
            album_title=pu_album_title,
            stamp_date_hash=date_estimation_enabled or bool(pu_dc_date),
        )
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            person_names=pu_person_names,
            subjects=pu_subjects,
            title=xmp_title,
            title_source=xmp_title_source,
            description=str(state.get("description") or ""),
            album_title=pu_album_title,
            location_payload=pu_inputs.existing_location,
            source_text=pu_source_text,
            ocr_text=pu_inputs.existing_ocr_text,
            author_text=str(text_layers.get("author_text") or ""),
            scene_text=str(text_layers.get("scene_text") or ""),
            detections_payload=pu_updated_det,
            stitch_key=str(state.get("stitch_key") or ""),
            ocr_authority_source=str(state.get("ocr_authority_source") or ""),
            create_date=(str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
            dc_date=pu_dc_date,
            date_time_original=pu_date_time_original,
            ocr_ran=bool(state.get("ocr_ran") or True),
            people_detected=pu_people_detected,
            people_identified=pu_people_identified,
            title_page_location=self.title_page_location,
        )

    # ── GPS-update path ─────────────────────────────────────────────────────

    def _process_gps_update(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return
        file_start = time.monotonic()
        gps_inputs = _gps_inputs_from_state(image_path, state, self.printed_album_title_cache)

        prefix = self._format_progress_prefix(idx, image_path)
        print(prefix, flush=True)
        _gps_stop, _gps_step = _progress_ticker(prefix)

        try:
            caption_key = self._caption_key_from_effective(effective)
            gps_caption_engine = self._get_caption_engine_for_key(caption_key, effective)
            gps_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
            gps_location_payload, gps_locations_shown, gps_locations_shown_ran = self._run_gps_resolution(
                image_path=image_path,
                caption_key=caption_key,
                caption_engine=gps_caption_engine,
                gps_inputs=gps_inputs,
                prompt_debug=gps_prompt_debug,
                step_fn=_gps_step,
            )
            _emit_prompt_debug_artifact(gps_prompt_debug, dry_run=self.dry_run)

            if not self.dry_run:
                self._write_gps_payload(
                    sidecar_path=sidecar_path,
                    image_path=image_path,
                    state=state,
                    existing_xmp_people=existing_xmp_people,
                    gps_inputs=gps_inputs,
                    gps_location_payload=gps_location_payload,
                    gps_locations_shown=gps_locations_shown,
                    gps_locations_shown_ran=gps_locations_shown_ran,
                )

            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            _gps_stop()
            self._emit_ok(idx, image_path, "[gps]")
        except Exception as exc:
            self.failures += 1
            _gps_stop()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _format_progress_prefix(self, idx: int, image_path: Path) -> str:
        eta_str = _format_eta(self.completed_times, len(self.files) - idx + 1)
        eta_part = f"  {eta_str}" if eta_str else ""
        return f"[{idx}/{len(self.files)}]{eta_part}  {_display_work_label(image_path)}"

    def _emit_ok(self, idx: int, image_path: Path, suffix: str = "") -> None:
        if self.stdout_only:
            return
        eta_str = _format_eta(self.completed_times, len(self.files) - idx)
        eta_part = f"  {eta_str}" if eta_str else ""
        suffix_text = f"  {suffix}" if suffix else ""
        print(
            f"[{idx}/{len(self.files)}]{eta_part}  ok    {image_path.name}{suffix_text}",
            flush=True,
        )

    def _run_gps_resolution(
        self,
        *,
        image_path: Path,
        caption_key: tuple[str, str, str, int, float, str, int, bool],
        caption_engine: CaptionEngine,
        gps_inputs: "_GpsInputs",
        prompt_debug: PromptDebugSession,
        step_fn: Any,
    ) -> tuple[dict[str, Any], list[Any], bool]:
        recorder = lambda record: _append_geocode_artifact(image_path=image_path, record=record)  # noqa: E731
        step_fn("location")
        with _prepare_ai_model_image(image_path) as gps_model_path:
            gps_latitude, gps_longitude, location_name = _resolve_location_metadata(
                requested_caption_engine=str(caption_key[0]),
                caption_engine=caption_engine,
                model_image_path=gps_model_path,
                people=gps_inputs.people_names,
                objects=gps_inputs.object_labels,
                ocr_text=gps_inputs.ocr_text,
                source_path=image_path,
                album_title=gps_inputs.album_title,
                printed_album_title=gps_inputs.printed_title,
                people_positions={},
                fallback_location_name=gps_inputs.existing_location_name,
                prompt_debug=prompt_debug,
                debug_step="location_gps_step",
            )
            step_fn("locations_shown")
            locations_shown, locations_shown_ran = _resolve_locations_shown(
                requested_caption_engine=str(caption_key[0]),
                caption_engine=caption_engine,
                model_image_path=gps_model_path,
                ocr_text=gps_inputs.ocr_text,
                source_path=image_path,
                album_title=gps_inputs.album_title,
                printed_album_title=gps_inputs.printed_title,
                geocoder=self.geocoder,
                prompt_debug=prompt_debug,
                debug_step="locations_shown_gps_step",
                artifact_recorder=recorder,
            )
        location_payload = _resolve_location_payload(
            geocoder=self.geocoder,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_name=location_name,
            artifact_recorder=recorder,
            artifact_step="location_gps_step",
        )
        location_payload, _ = _apply_title_page_location_config(
            image_path=image_path,
            location_payload=location_payload,
            title_page_location=self.title_page_location,
        )
        return location_payload, locations_shown, locations_shown_ran

    def _write_gps_payload(
        self,
        *,
        sidecar_path: Path,
        image_path: Path,
        state: dict,
        existing_xmp_people: list[str],
        gps_inputs: "_GpsInputs",
        gps_location_payload: dict[str, Any],
        gps_locations_shown: list[Any],
        gps_locations_shown_ran: bool,
    ) -> None:
        gps_updated_det = _gps_updated_detections(
            gps_inputs.detections, gps_location_payload, gps_locations_shown, gps_locations_shown_ran
        )
        gps_subjects = _dedupe(
            gps_inputs.object_labels
            + gps_inputs.ocr_keywords
            + ([gps_inputs.album_title] if gps_inputs.album_title else [])
        )
        gps_scan_filenames = _page_scan_filenames(image_path)
        gps_source_text = (
            _build_dc_source(gps_inputs.album_title, image_path, gps_scan_filenames)
            if gps_scan_filenames
            else str(state.get("source_text") or "")
        )
        xmp_title, xmp_title_source = _compute_xmp_title(
            image_path=image_path,
            explicit_title=str(state.get("title") or ""),
            title_source=str(state.get("title_source") or ""),
            author_text=str(state.get("author_text") or ""),
        )
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            **_state_writer_kwargs(
                state=state,
                image_path=image_path,
                person_names=list(existing_xmp_people),
                subjects=gps_subjects,
                title=xmp_title,
                title_source=xmp_title_source,
                album_title=gps_inputs.album_title,
                location_payload=gps_location_payload,
                source_text=gps_source_text,
                ocr_text=gps_inputs.ocr_text,
                detections_payload=gps_updated_det,
                dc_date=_dc_date_value(state),
                date_time_original=str(state.get("date_time_original") or ""),
            ),
            title_page_location=self.title_page_location,
        )

    # ── Full processing path ────────────────────────────────────────────────

    def _process_full(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        archive_stitched_ocr_required: bool,
        multi_scan_group_paths: list[Path],
        multi_scan_group_signature: str,
        extra_forced_steps: set[str] | None = None,
    ) -> None:
        file_start = time.monotonic()
        stop_ticker, set_step = self._begin_full_progress(idx, image_path)
        hints = _resolve_full_hints(image_path, self.printed_album_title_cache)
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))

        try:
            engines = self._init_full_engines(effective)
            scan_ocr_authority = self._resolve_full_scan_ocr_authority(
                archive_stitched_ocr_required=archive_stitched_ocr_required,
                image_path=image_path,
                multi_scan_group_paths=multi_scan_group_paths,
                multi_scan_group_signature=multi_scan_group_signature,
                ocr_engine=engines.ocr_engine,
                set_step=set_step,
                prompt_debug=prompt_debug,
            )
            with prepare_image_layout(image_path, split_mode="off") as layout:
                outcome = self._run_full_analysis(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    effective=effective,
                    hints=hints,
                    engines=engines,
                    layout=layout,
                    scan_ocr_authority=scan_ocr_authority,
                    people_matcher=people_matcher,
                    existing_xmp_people=existing_xmp_people,
                    existing_sidecar_state=existing_sidecar_state,
                    current_cast_signature=current_cast_signature,
                    multi_scan_group_signature=multi_scan_group_signature,
                    set_step=set_step,
                    prompt_debug=prompt_debug,
                    extra_forced_steps=extra_forced_steps,
                )
                if not self.dry_run:
                    self._write_full_payload(
                        sidecar_path=sidecar_path,
                        image_path=image_path,
                        effective=effective,
                        settings_sig=settings_sig,
                        date_estimation_enabled=date_estimation_enabled,
                        existing_sidecar_state=existing_sidecar_state,
                        people_matcher=people_matcher,
                        current_cast_signature=current_cast_signature,
                        layout=layout,
                        scan_ocr_authority=scan_ocr_authority,
                        outcome=outcome,
                    )
                    self._run_propagate_to_crops(image_path, outcome)

            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            if stop_ticker is not None:
                stop_ticker()
            self._emit_full_completion(idx, image_path, outcome)
            _mirror_page_sidecars(image_path)
        except Exception as exc:
            self.failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            if stop_ticker is not None:
                stop_ticker()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    def _begin_full_progress(self, idx: int, image_path: Path) -> tuple[Any, Any]:
        if self.stdout_only:
            return None, None
        prefix = self._format_progress_prefix(idx, image_path)
        print(prefix, flush=True)
        return _progress_ticker(prefix)

    def _init_full_engines(self, effective: dict[str, Any]) -> "_FullEngines":
        object_detector = self._get_object_detector(effective) if bool(
            effective.get("enable_objects", True)
        ) else None
        caption_key = self._caption_key_from_effective(effective)
        caption_engine = self._get_caption_engine_for_key(
            caption_key, effective, stream=not self.stdout_only
        )
        ocr_engine, ocr_key = self._get_ocr_engine(effective)
        return _FullEngines(
            caption_engine=caption_engine,
            caption_key=caption_key,
            ocr_engine=ocr_engine,
            ocr_key=ocr_key,
            object_detector=object_detector,
        )

    def _get_object_detector(self, effective: dict[str, Any]) -> Any:
        object_key = (
            str(effective.get("model", self.defaults["model"])),
            float(effective.get("object_threshold", self.defaults["object_threshold"])),
        )
        detector = self.object_detector_cache.get(object_key)
        if detector is None:
            detector = _init_object_detector(
                model_name=str(object_key[0]),
                confidence=float(object_key[1]),
            )
            self.object_detector_cache[object_key] = detector
        return detector

    def _get_ocr_engine(
        self, effective: dict[str, Any]
    ) -> tuple[OCREngine, tuple[str, str, str, str]]:
        ocr_key = (
            str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
            str(effective.get("ocr_lang", self.defaults["ocr_lang"])),
            str(effective.get("ocr_model", self.defaults["ocr_model"])),
            normalize_lmstudio_base_url(
                str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"]))
            ),
        )
        engine = self.ocr_engine_cache.get(ocr_key)
        if engine is None:
            engine = OCREngine(
                engine=ocr_key[0],
                language=ocr_key[1],
                model_name=ocr_key[2],
                base_url=ocr_key[3],
            )
            self.ocr_engine_cache[ocr_key] = engine
        return engine, ocr_key

    def _resolve_full_scan_ocr_authority(
        self,
        *,
        archive_stitched_ocr_required: bool,
        image_path: Path,
        multi_scan_group_paths: list[Path],
        multi_scan_group_signature: str,
        ocr_engine: OCREngine,
        set_step: Any,
        prompt_debug: PromptDebugSession | None,
    ) -> ArchiveScanOCRAuthority | None:
        if not archive_stitched_ocr_required:
            return None
        return _resolve_archive_scan_authoritative_ocr(
            image_path=image_path,
            group_paths=multi_scan_group_paths,
            group_signature=multi_scan_group_signature,
            cache=self.archive_scan_ocr_cache,
            ocr_engine=ocr_engine,
            step_fn=set_step,
            stitched_image_dir=self.stitch_cap_dir,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
        )

    def _build_full_step_runner(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        engines: "_FullEngines",
        current_cast_signature: str,
        multi_scan_group_signature: str,
        existing_sidecar_state: dict | None,
        extra_forced_steps: set[str] | None,
    ) -> tuple[StepRunner, dict[str, Any]]:
        from .ai_index_propagate import (  # pylint: disable=import-outside-toplevel
            _crop_paths_signature as _cps_fn,
            _find_crop_paths_for_page,
        )

        crop_paths = _find_crop_paths_for_page(image_path)
        step_settings = {
            "ocr_engine": str(engines.ocr_key[0]),
            "ocr_model": str(engines.ocr_key[2]),
            "ocr_lang": str(engines.ocr_key[1]),
            "scan_group_signature": multi_scan_group_signature,
            "cast_store_signature": (
                current_cast_signature if bool(effective.get("enable_people", True)) else ""
            ),
            "caption_engine": str(engines.caption_key[0]),
            "caption_model": str(engines.caption_key[1]),
            "nominatim_base_url": str(getattr(self.geocoder, "base_url", "") or "") if self.geocoder else "",
            "model": str(effective.get("model", self.defaults.get("model", ""))),
            "enable_objects": bool(effective.get("enable_objects", True)),
            "crop_paths_signature": _cps_fn(crop_paths),
        }
        existing_pipeline_state = read_pipeline_state(sidecar_path)
        existing_detections = dict((existing_sidecar_state or {}).get("detections") or {})
        steps_arg = str(getattr(self.args, "steps", "") or "").strip()
        forced_steps = {s.strip() for s in steps_arg.split(",") if s.strip()} if steps_arg else set()
        if extra_forced_steps:
            forced_steps = forced_steps | extra_forced_steps
        runner = StepRunner(
            settings=step_settings,
            existing_pipeline_state=existing_pipeline_state,
            existing_detections=existing_detections,
            forced_steps=forced_steps,
        )
        return runner, existing_detections

    def _run_full_analysis(
        self,
        *,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        hints: "_FullHints",
        engines: "_FullEngines",
        layout: Any,
        scan_ocr_authority: ArchiveScanOCRAuthority | None,
        people_matcher: Any,
        existing_xmp_people: list[str],
        existing_sidecar_state: dict | None,
        current_cast_signature: str,
        multi_scan_group_signature: str,
        set_step: Any,
        prompt_debug: PromptDebugSession,
        extra_forced_steps: set[str] | None,
    ) -> "_FullAnalysisOutcome":
        scan_filenames = _page_scan_filenames(image_path)
        if not scan_filenames and scan_ocr_authority is not None:
            scan_filenames = [path.name for path in scan_ocr_authority.group_paths]
        targets = _full_analysis_targets(image_path, layout, scan_ocr_authority)
        derived_ocr_override = _effective_sidecar_ocr_text(image_path, existing_sidecar_state)
        step_runner, existing_detections = self._build_full_step_runner(
            image_path=image_path,
            sidecar_path=sidecar_path,
            effective=effective,
            engines=engines,
            current_cast_signature=current_cast_signature,
            multi_scan_group_signature=multi_scan_group_signature,
            existing_sidecar_state=existing_sidecar_state,
            extra_forced_steps=extra_forced_steps,
        )
        analysis = _run_image_analysis(
            image_path=targets.analysis_target,
            people_image_path=targets.people_analysis_source,
            people_matcher=people_matcher,
            object_detector=engines.object_detector,
            ocr_engine=engines.ocr_engine,
            caption_engine=engines.caption_engine,
            requested_caption_engine=str(engines.caption_key[0]),
            ocr_engine_name=engines.ocr_key[0],
            ocr_language=engines.ocr_key[1],
            people_source_path=targets.people_analysis_source,
            people_bbox_offset=(_bounds_offset(layout.content_bounds) if layout.page_like else (0, 0)),
            caption_source_path=(image_path if layout.page_like else targets.analysis_target),
            album_title=hints.album_title_hint,
            printed_album_title=hints.album_title_hint,
            geocoder=self.geocoder,
            step_fn=set_step,
            extra_people_names=existing_xmp_people,
            is_page_scan=layout.page_like,
            ocr_text_override=(
                scan_ocr_authority.ocr_text
                if scan_ocr_authority is not None
                else (derived_ocr_override or None)
            ),
            context_ocr_text=hints.upstream_context_ocr,
            context_location_hint=hints.upstream_location_hint,
            prompt_debug=prompt_debug,
            title_page_location=self.title_page_location,
            step_runner=step_runner,
            existing_sidecar_state=existing_sidecar_state,
            metadata_engine=self._get_metadata_engine(effective),
        )
        resolved_album_title = analysis.album_title or hints.album_title_hint
        _store_album_printed_title_hint(
            image_path,
            self.printed_album_title_cache,
            resolved_album_title,
        )
        person_names = _dedupe(analysis.people_names + existing_xmp_people)
        subjects = _dedupe(analysis.subjects + ([resolved_album_title] if resolved_album_title else []))
        description = (
            _build_flat_page_description(analysis=analysis) if layout.page_like else analysis.description
        )
        payload = _build_flat_payload(layout, analysis)
        analysis_mode = "page_flat" if layout.page_like else "single_image"
        ocr_authority_hash = str(scan_ocr_authority.ocr_hash) if scan_ocr_authority is not None else ""
        payload = _refresh_detection_model_metadata(
            payload,
            ocr_model=_full_engine_model_name(engines.ocr_engine, engines.ocr_key[0]),
            caption_model=_full_engine_model_name(engines.caption_engine, engines.caption_key[0]),
        )
        return _FullAnalysisOutcome(
            analysis=analysis,
            payload=payload,
            person_names=person_names,
            subjects=subjects,
            description=description,
            ocr_text=analysis.ocr_text,
            resolved_album_title=resolved_album_title,
            analysis_mode=analysis_mode,
            ocr_authority_hash=ocr_authority_hash,
            scan_filenames=scan_filenames,
            step_runner=step_runner,
            existing_detections=existing_detections,
        )

    def _write_full_payload(
        self,
        *,
        sidecar_path: Path,
        image_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        people_matcher: Any,
        current_cast_signature: str,
        layout: Any,
        scan_ocr_authority: ArchiveScanOCRAuthority | None,
        outcome: "_FullAnalysisOutcome",
    ) -> None:
        payload = outcome.payload
        analysis = outcome.analysis
        _merge_pipeline_records(payload, outcome.existing_detections, outcome.step_runner)
        location_payload = _full_resolve_location_payload(
            payload, outcome.step_runner, image_path, existing_sidecar_state
        )
        if location_payload:
            payload["location"] = location_payload
        final_album_title = _require_album_title_for_title_page(
            image_path=image_path,
            album_title=_resolve_title_page_album_title(
                image_path=image_path,
                album_title=(outcome.resolved_album_title or _resolve_album_title_hint(image_path)),
                ocr_text=outcome.ocr_text,
            ),
            context="write",
        )
        final_dc_date = _full_final_dc_date(analysis, existing_sidecar_state)
        final_date_time_original = _resolve_date_time_original(
            dc_date=final_dc_date,
            date_time_original=str((existing_sidecar_state or {}).get("date_time_original") or ""),
        )
        text_layers = _resolve_xmp_text_layers(
            image_path=image_path,
            ocr_text=outcome.ocr_text,
            page_like=bool(layout.page_like),
            ocr_authority_source=("archive_stitched" if scan_ocr_authority is not None else ""),
            author_text=str(analysis.author_text or ""),
            scene_text=str(analysis.scene_text or ""),
        )
        xmp_title, xmp_title_source = _compute_xmp_title(
            image_path=image_path,
            explicit_title=str(analysis.title or ""),
            author_text=str(text_layers.get("author_text") or ""),
        )
        if people_matcher is not None:
            current_cast_signature = self._people_invalidation_signature(people_matcher)
        payload["processing"] = _full_processing_payload(
            image_path=image_path,
            settings_sig=settings_sig,
            current_cast_signature=current_cast_signature,
            effective=effective,
            ocr_text=outcome.ocr_text,
            final_album_title=final_album_title,
            final_dc_date=final_dc_date,
            existing_sidecar_state=existing_sidecar_state,
            scan_ocr_authority=scan_ocr_authority,
            ocr_authority_hash=outcome.ocr_authority_hash,
            analysis_mode=outcome.analysis_mode,
            date_estimation_enabled=date_estimation_enabled,
        )
        _write_sidecar_and_record(
            sidecar_path,
            image_path,
            person_names=outcome.person_names,
            subjects=outcome.subjects,
            title=xmp_title,
            title_source=xmp_title_source,
            description=outcome.description,
            album_title=final_album_title,
            location_payload=location_payload,
            source_text=_build_dc_source(final_album_title, image_path, outcome.scan_filenames),
            ocr_text=outcome.ocr_text,
            ocr_lang=str(analysis.ocr_lang or ""),
            author_text=str(text_layers.get("author_text") or ""),
            scene_text=str(text_layers.get("scene_text") or ""),
            detections_payload=payload,
            subphotos=None,
            ocr_authority_source=("archive_stitched" if scan_ocr_authority is not None else ""),
            create_date=read_embedded_create_date(image_path),
            dc_date=final_dc_date,
            date_time_original=final_date_time_original,
            ocr_ran=str(effective.get("ocr_engine", self.defaults["ocr_engine"])).lower() != "none",
            people_detected=analysis.faces_detected > 0 or len(outcome.person_names) > 0,
            people_identified=len(outcome.person_names) > 0,
            title_page_location=self.title_page_location,
        )

    def _run_propagate_to_crops(self, image_path: Path, outcome: "_FullAnalysisOutcome") -> None:
        from .ai_index_propagate import run_propagate_to_crops  # pylint: disable=import-outside-toplevel

        locations_out = dict(outcome.payload.get("location") or {})
        people_out = list(outcome.payload.get("people") or [])

        def _do_propagate() -> dict:
            return run_propagate_to_crops(
                image_path,
                location_payload=locations_out,
                people_payload=people_out,
            )

        outcome.step_runner.run("propagate-to-crops", _do_propagate)

    def _emit_full_completion(
        self, idx: int, image_path: Path, outcome: "_FullAnalysisOutcome"
    ) -> None:
        if self.stdout_only:
            payload = outcome.payload
            caption_meta = dict(payload.get("caption") or {}) if isinstance(payload, dict) else {}
            fallback_error = str(caption_meta.get("error") or "").strip()
            if bool(caption_meta.get("fallback")) and fallback_error:
                self.emit_error(
                    f"[{idx}/{len(self.files)}] warn  {image_path.name}: caption fallback: {fallback_error}"
                )
            print(f"{image_path.name}: {outcome.description}" if outcome.description else image_path.name)
            return
        self._emit_ok(idx, image_path)


def refresh_rendered_view_people_metadata(
    image_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
) -> None:
    rendered_image_path = Path(image_path)
    rendered_sidecar_path = Path(sidecar_path) if sidecar_path is not None else rendered_image_path.with_suffix(".xmp")
    if not has_valid_sidecar(rendered_image_path):
        raise RuntimeError(f"Rendered sidecar missing or invalid for people refresh: {rendered_sidecar_path}")
    existing_sidecar_state = read_ai_sidecar_state(rendered_sidecar_path)
    if not isinstance(existing_sidecar_state, dict):
        raise RuntimeError(f"Rendered sidecar could not be parsed for people refresh: {rendered_sidecar_path}")

    runner = IndexRunner(["--photo", str(rendered_image_path), "--include-view"])
    runner.files = [rendered_image_path]
    effective, settings_sig, date_estimation_enabled = runner._resolve_effective_settings(
        rendered_image_path
    )
    people_matcher, current_cast_signature = runner._get_people_matcher_and_signature(effective)
    if people_matcher is None:
        return

    runner._process_people_update(
        1,
        rendered_image_path,
        rendered_sidecar_path,
        effective,
        settings_sig,
        date_estimation_enabled,
        existing_sidecar_state,
        read_person_in_image(rendered_sidecar_path),
        people_matcher,
        current_cast_signature,
        False,
        preserve_existing_xmp_people=False,
        raise_on_error=True,
    )
