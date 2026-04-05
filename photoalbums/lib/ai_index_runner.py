from __future__ import annotations

import sys
import tempfile
import time
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
    _effective_sidecar_location_payload,
    _effective_sidecar_ocr_text,
    _resolve_xmp_text_layers,
    _sidecar_current_for_paths,
    _xmp_timestamp_from_path,
    has_current_sidecar,
    has_valid_sidecar,
    read_embedded_create_date,
)
from .prompt_debug import PromptDebugSession
from .xmp_sidecar import (
    _dedupe,
    _resolve_date_time_original,
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    sidecar_has_expected_ai_fields,
    write_xmp_sidecar,
)
from .xmp_review import load_ai_xmp_review

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
    creator_tool: str,
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
    write_xmp_sidecar(
        sidecar_path,
        creator_tool=creator_tool,
        person_names=person_names,
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
        people_detected=people_detected,
        people_identified=people_identified,
        locations_shown=detections_payload.get("locations_shown") if detections_payload else None,
    )
    _append_xmp_job_artifact(image_path, sidecar_path)


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

        if not self.photos_root.is_dir():
            raise SystemExit(f"Photo root is not a directory: {self.photos_root}")
        if self.shard_count < 1:
            raise SystemExit("--shard-count must be at least 1")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise SystemExit("--shard-index must be between 0 and --shard-count - 1")

        self.include_archive = bool(self.args.include_archive)
        self.include_view = bool(self.args.include_view)
        if not self.include_archive and not self.include_view:
            self.include_archive = True
            self.include_view = False

        self.ext_set = {
            (item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}")
            for item in str(self.args.extensions or "").split(",")
            if item.strip()
        }
        if not self.ext_set:
            self.ext_set = set(IMAGE_EXTENSIONS)

        self.single_photo = str(self.args.photo or "").strip()

        default_caption_max_tokens = int(self.args.caption_max_tokens)
        if "--caption-max-tokens" not in self.explicit_flags and str(self.args.caption_engine) == "lmstudio":
            default_caption_max_tokens = max(default_caption_max_tokens, int(DEFAULT_LMSTUDIO_MAX_NEW_TOKENS))

        self.defaults = {
            "skip": False,
            "enable_people": not bool(self.args.disable_people),
            "enable_objects": not bool(self.args.disable_objects),
            "ocr_engine": str(self.args.ocr_engine),
            "ocr_lang": str(self.args.ocr_lang),
            "ocr_model": str(self.args.ocr_model),
            "caption_engine": str(self.args.caption_engine),
            "caption_model": resolve_caption_model(str(self.args.caption_engine), str(self.args.caption_model)),
            "caption_prompt": str(self.requested_caption_prompt),
            "caption_max_tokens": int(default_caption_max_tokens),
            "caption_temperature": float(self.args.caption_temperature),
            "caption_max_edge": int(self.args.caption_max_edge),
            "lmstudio_base_url": normalize_lmstudio_base_url(str(self.args.lmstudio_base_url)),
            "people_threshold": float(self.args.people_threshold),
            "object_threshold": float(self.args.object_threshold),
            "min_face_size": int(self.args.min_face_size),
            "model": str(self.args.model),
            "creator_tool": str(self.args.creator_tool),
        }

        self.archive_settings_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.people_matcher_cache: dict[tuple[str, float, int], Any] = {}
        self.object_detector_cache: dict[tuple[str, float], Any] = {}
        self.ocr_engine_cache: dict[tuple[str, str, str, str], OCREngine] = {}
        self.caption_engine_cache: dict[tuple[str, str, str, int, float, str, int], CaptionEngine] = {}
        self.date_engine_cache: dict[tuple[str, str, int, float, str], DateEstimateEngine] = {}
        self.archive_scan_ocr_cache: dict[str, ArchiveScanOCRAuthority] = {}
        self.printed_album_title_cache: dict[str, str] = {}
        self.geocoder = NominatimGeocoder()
        self.stitch_cap_td = tempfile.TemporaryDirectory(prefix="imago-stitch-cap-")
        self.stitch_cap_dir = Path(self.stitch_cap_td.name)

        self.processed = 0
        self.skipped = 0
        self.failures = 0
        self.completed_times: list[float] = []

        self.files: list[Path] = []
        self.batch_lock_path: Path | None = None
        self.allow_concurrent_shards = False

    def emit_info(self, message: str) -> None:
        if not self.stdout_only:
            print(message)

    def emit_error(self, message: str) -> None:
        print(message, file=sys.stderr if self.stdout_only else sys.stdout, flush=True)

    def _get_caption_engine(self, effective: dict[str, Any]) -> CaptionEngine:
        caption_key = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
            str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
            int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
            float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
            str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
            int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
        )
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
                stream=True,
            )
            self.caption_engine_cache[caption_key] = caption_engine
        return caption_engine

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

    def _resolve_effective_settings(self, image_path: Path) -> tuple[dict[str, Any], str, str, bool]:
        archive_dir = find_archive_dir_for_image(image_path)
        settings_file: Path | None = None
        loaded_settings: dict[str, Any] | None = None
        if archive_dir is not None and not self.args.ignore_render_settings:
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
            settings_file, loaded_settings = cached

        effective = resolve_effective_settings(
            image_path,
            defaults=self.defaults,
            loaded=loaded_settings,
        )
        if self.args.disable_people:
            effective["enable_people"] = False
        if self.args.disable_objects:
            effective["enable_objects"] = False
        if "--ocr-engine" in self.explicit_flags:
            effective["ocr_engine"] = str(self.args.ocr_engine)
        if "--ocr-model" in self.explicit_flags:
            effective["ocr_model"] = str(self.args.ocr_model)
        if "--caption-engine" in self.explicit_flags:
            effective["caption_engine"] = str(self.args.caption_engine)
        if "--caption-model" in self.explicit_flags:
            effective["caption_model"] = str(self.args.caption_model)
        if (
            "--caption-prompt" in self.explicit_flags
            or "--local-prompt" in self.explicit_flags
            or "--local-prompt" in self.explicit_flags
            or "--qwen-prompt" in self.explicit_flags
            or "--caption-prompt-file" in self.explicit_flags
            or "--local-prompt-file" in self.explicit_flags
            or "--local-prompt-file" in self.explicit_flags
            or "--qwen-prompt-file" in self.explicit_flags
        ):
            effective["caption_prompt"] = str(self.requested_caption_prompt)
        if "--caption-max-tokens" in self.explicit_flags:
            effective["caption_max_tokens"] = int(self.args.caption_max_tokens)
        if "--caption-temperature" in self.explicit_flags:
            effective["caption_temperature"] = float(self.args.caption_temperature)
        if "--caption-max-edge" in self.explicit_flags:
            effective["caption_max_edge"] = int(self.args.caption_max_edge)
        if "--lmstudio-base-url" in self.explicit_flags:
            effective["lmstudio_base_url"] = normalize_lmstudio_base_url(str(self.args.lmstudio_base_url))
        effective["caption_model"] = resolve_caption_model(
            str(effective.get("caption_engine", self.defaults["caption_engine"])),
            str(effective.get("caption_model", self.defaults["caption_model"])),
        )
        settings_sig = _settings_signature(effective)
        creator_tool = str(effective.get("creator_tool", self.args.creator_tool))
        date_estimation_enabled = (
            str(effective.get("caption_engine", self.defaults["caption_engine"])).strip().lower() == "lmstudio"
        )
        return effective, settings_sig, creator_tool, date_estimation_enabled

    # ── Per-image dispatch ──────────────────────────────────────────────────

    def _process_one(self, idx: int, image_path: Path) -> None:  # noqa: C901
        sidecar_path = image_path.with_suffix(".xmp")
        existing_xmp_people = read_person_in_image(sidecar_path)

        effective, settings_sig, creator_tool, date_estimation_enabled = self._resolve_effective_settings(image_path)

        existing_sidecar_valid = has_valid_sidecar(image_path)
        existing_sidecar_current = has_current_sidecar(image_path) if existing_sidecar_valid else False
        existing_sidecar_state: dict | None = None
        source_refresh_required = False
        if existing_sidecar_valid:
            existing_sidecar_state = read_ai_sidecar_state(sidecar_path)

        existing_sidecar_complete = False
        reprocess_required = False
        reprocess_reasons: list[str] = []
        date_refresh_required = False
        if existing_sidecar_valid and not existing_sidecar_current:
            reprocess_reasons.append("sidecar_older_than_image")
        if _sidecar_has_lmstudio_caption_error(existing_sidecar_state):
            reprocess_required = True
            reprocess_reasons.append("lmstudio_caption_error")
        if existing_sidecar_valid:
            existing_sidecar_complete = sidecar_has_expected_ai_fields(
                sidecar_path,
                creator_tool=creator_tool,
                enable_people=bool(effective.get("enable_people", True)),
                enable_objects=bool(effective.get("enable_objects", True)),
                ocr_engine=str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
                caption_engine=str(effective.get("caption_engine", self.defaults["caption_engine"])),
            )
            source_refresh_required = _dc_source_needs_refresh(image_path, existing_sidecar_state)
            if source_refresh_required:
                reprocess_reasons.append("dc_source_stale")
            date_refresh_required = _dc_date_needs_refresh(
                image_path,
                existing_sidecar_state,
                enabled=date_estimation_enabled,
            )
            if date_refresh_required:
                reprocess_reasons.append("timeline_date_missing")
        caption_engine_name = str(effective.get("caption_engine", self.defaults["caption_engine"])).strip().lower()
        location_shown_missing = False
        location_shown_gps_dirty = False
        if existing_sidecar_complete and existing_sidecar_state is not None and caption_engine_name == "lmstudio":
            det = existing_sidecar_state.get("detections") or {}
            detected_locations = list(det.get("locations_shown") or []) if isinstance(det, dict) else []
            written_locations = read_locations_shown(sidecar_path)
            location_shown_ran = isinstance(det, dict) and det.get("location_shown_ran") is True
            location_shown_missing = (isinstance(det, dict) and det.get("location_shown_ran") is not True) or (
                (location_shown_ran or bool(detected_locations)) and not written_locations
            )
            location_shown_gps_dirty = _has_legacy_ai_locations_shown_gps(existing_sidecar_state)
        gps_repair_requested = (
            existing_sidecar_current
            and existing_sidecar_complete
            and existing_sidecar_state is not None
            and (location_shown_missing or location_shown_gps_dirty)
            and not reprocess_required
            and not source_refresh_required
            and not date_refresh_required
        )

        if (
            existing_sidecar_current
            and existing_sidecar_complete
            and not reprocess_required
            and not source_refresh_required
            and not date_refresh_required
            and not self.force_processing
            and not gps_repair_requested
        ):
            self.skipped += 1
            if self.args.verbose and not self.stdout_only:
                print(f"[{idx}/{len(self.files)}] skip  {image_path.name} (current xmp)")
            return

        people_matcher = None
        current_cast_signature = ""
        if bool(effective.get("enable_people", True)):
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
            current_cast_signature = str(people_matcher.store_signature())

        existing_sidecar_ocr_hash = _hash_text(str((existing_sidecar_state or {}).get("ocr_text") or ""))
        multi_scan_group_paths = _scan_group_paths(image_path)
        archive_stitched_ocr_required = (
            str(effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower() != "none"
            and len(multi_scan_group_paths) > 1
        )
        multi_scan_group_signature = (
            _scan_group_signature(multi_scan_group_paths) if archive_stitched_ocr_required else ""
        )

        people_update_only = False
        if existing_sidecar_valid and not existing_sidecar_complete:
            reprocess_required = True
            reprocess_reasons.append("sidecar_incomplete")
        existing_album_title = str((existing_sidecar_state or {}).get("album_title") or "").strip()
        if not existing_album_title and (
            _is_album_title_source_candidate(image_path) or _resolve_album_title_from_sidecars(image_path)
        ):
            reprocess_required = True
            reprocess_reasons.append("missing_album_title")
        if archive_stitched_ocr_required:
            sidecar_source = str((existing_sidecar_state or {}).get("ocr_authority_source") or "").strip()
            sidecar_signature = str((existing_sidecar_state or {}).get("ocr_authority_signature") or "").strip()
            sidecar_hash = str((existing_sidecar_state or {}).get("ocr_authority_hash") or "").strip()
            sidecar_has_current_stitched_authority = (
                sidecar_source == "archive_stitched"
                and bool(existing_sidecar_ocr_hash)
                and _sidecar_current_for_paths(sidecar_path, multi_scan_group_paths)
            )
            sidecar_matches_stitched_authority = (
                sidecar_source == "archive_stitched"
                and sidecar_signature == multi_scan_group_signature
                and bool(sidecar_hash)
                and sidecar_hash == existing_sidecar_ocr_hash
            )
            if not sidecar_matches_stitched_authority and not sidecar_has_current_stitched_authority:
                reprocess_required = True
                reprocess_reasons.append("missing_stitched_authority")
        if existing_sidecar_state is not None:
            old_sig = str(existing_sidecar_state.get("settings_signature") or "")
            if old_sig != settings_sig and not (existing_sidecar_current and existing_sidecar_complete):
                reprocess_required = True
                reprocess_reasons.append("settings_signature_mismatch")
            elif bool(effective.get("enable_people", True)):
                if str(existing_sidecar_state.get("cast_store_signature") or "") != current_cast_signature:
                    if _sidecar_has_people_to_refresh(existing_sidecar_state):
                        people_update_only = True
                        reprocess_reasons.append("cast_store_signature_changed")

        needs_full = needs_processing(
            image_path,
            existing_sidecar_state,
            self.force_processing and not gps_repair_requested,
            reprocess_required=reprocess_required,
        )

        gps_update_only = False
        if gps_repair_requested:
            gps_update_only = True
            if location_shown_missing:
                reprocess_reasons.append("missing_location_shown")
            if location_shown_gps_dirty:
                reprocess_reasons.append("location_shown_ai_gps_stale")
        if (
            not gps_update_only
            and self.reprocess_mode == "gps"
            and not needs_full
            and existing_sidecar_complete
            and existing_sidecar_state is not None
        ):
            gps_update_only = True
        if (
            not gps_update_only
            and not needs_full
            and not source_refresh_required
            and not date_refresh_required
            and existing_sidecar_complete
            and existing_sidecar_state is not None
            and caption_engine_name == "lmstudio"
        ):
            if location_shown_missing:
                gps_update_only = True
                reprocess_reasons.append("missing_location_shown")
            if location_shown_gps_dirty:
                gps_update_only = True
                reprocess_reasons.append("location_shown_ai_gps_stale")
        if (
            not needs_full
            and not people_update_only
            and not gps_update_only
            and not isinstance(existing_sidecar_state, dict)
        ):
            self.skipped += 1
            if self.args.verbose and not self.stdout_only:
                print(f"[{idx}/{len(self.files)}] skip  {image_path.name}")
            return

        if bool(effective.get("skip", False)):
            self.skipped += 1
            if self.args.verbose and not self.stdout_only:
                print(f"[{idx}/{len(self.files)}] skip  {image_path.name} (render_settings skip=true)")
            return

        if self.reprocess_mode not in ("unprocessed", "all"):
            _reasons_set = set(reprocess_reasons)
            _mode_match: bool
            if self.reprocess_mode == "new_only":
                _mode_match = existing_sidecar_state is None
            elif self.reprocess_mode == "errors_only":
                _mode_match = bool(_reasons_set & {"lmstudio_caption_error", "sidecar_incomplete"})
            elif self.reprocess_mode == "outdated":
                _mode_match = "sidecar_older_than_image" in _reasons_set
            elif self.reprocess_mode == "cast_changed":
                _mode_match = "cast_store_signature_changed" in _reasons_set
            elif self.reprocess_mode == "gps":
                _mode_match = gps_update_only
            else:
                _mode_match = True
            if not _mode_match:
                self.skipped += 1
                if self.args.verbose and not self.stdout_only:
                    print(f"[{idx}/{len(self.files)}] skip  {image_path.name} (reprocess_mode={self.reprocess_mode})")
                return

        if existing_sidecar_valid and not self.stdout_only:
            reason_text = _format_reprocess_reasons(reprocess_reasons)
            if needs_full and reason_text:
                print(
                    f"  [{idx}/{len(self.files)}]  {image_path.name}  [reprocess: {reason_text}]",
                    flush=True,
                )
            elif people_update_only and reason_text:
                print(
                    f"  [{idx}/{len(self.files)}]  {image_path.name}  [update: {reason_text}]",
                    flush=True,
                )
            elif (source_refresh_required or date_refresh_required) and reason_text:
                print(
                    f"  [{idx}/{len(self.files)}]  {image_path.name}  [refresh: {reason_text}]",
                    flush=True,
                )

        try:
            lock_path = _acquire_image_processing_lock(image_path)
        except RuntimeError as exc:
            if self.allow_concurrent_shards and "already processing" in str(exc):
                self.skipped += 1
                if self.args.verbose and not self.stdout_only:
                    print(f"[{idx}/{len(self.files)}] skip  {image_path.name} ({exc})")
            else:
                self.failures += 1
                self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")
            return

        try:
            if not needs_full and not people_update_only and not gps_update_only:
                if isinstance(existing_sidecar_state, dict):
                    self._process_refresh(
                        idx,
                        image_path,
                        sidecar_path,
                        effective,
                        settings_sig,
                        creator_tool,
                        date_estimation_enabled,
                        existing_sidecar_state,
                        current_cast_signature,
                    )
                return

            if not needs_full and people_update_only and not self.stdout_only:
                chain_gps = bool(gps_update_only)
                if not isinstance(existing_sidecar_state, dict):
                    needs_full = True
                else:
                    self._process_people_update(
                        idx,
                        image_path,
                        sidecar_path,
                        effective,
                        settings_sig,
                        creator_tool,
                        date_estimation_enabled,
                        existing_sidecar_state,
                        existing_xmp_people,
                        people_matcher,
                        current_cast_signature,
                        chain_gps,
                    )
                    if not chain_gps:
                        return
                    if not self.dry_run:
                        existing_sidecar_state = read_ai_sidecar_state(sidecar_path)
                        existing_xmp_people = read_person_in_image(sidecar_path)

            if not needs_full and gps_update_only and not self.stdout_only:
                if not isinstance(existing_sidecar_state, dict):
                    return
                self._process_gps_update(
                    idx,
                    image_path,
                    sidecar_path,
                    effective,
                    creator_tool,
                    existing_sidecar_state,
                    existing_xmp_people,
                )
                return

            self._process_full(
                idx,
                image_path,
                sidecar_path,
                effective,
                settings_sig,
                creator_tool,
                date_estimation_enabled,
                existing_sidecar_state,
                existing_xmp_people,
                people_matcher,
                current_cast_signature,
                archive_stitched_ocr_required,
                multi_scan_group_paths,
                multi_scan_group_signature,
            )
        finally:
            _release_image_processing_lock(lock_path)

    # ── Refresh fast-path ───────────────────────────────────────────────────

    def _process_refresh(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        creator_tool: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        current_cast_signature: str,
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return
        file_start = time.monotonic()
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
        try:
            review = load_ai_xmp_review(sidecar_path)
            refresh_ocr_text = _effective_sidecar_ocr_text(
                image_path,
                review if isinstance(review, dict) else None,
            )
            refresh_location = _effective_sidecar_location_payload(
                image_path,
                review if isinstance(review, dict) else None,
            )
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
                refresh_gps_lat = str(refresh_location.get("gps_latitude") or "").strip()
                refresh_gps_lon = str(refresh_location.get("gps_longitude") or "").strip()
                if not refresh_gps_lat:
                    refresh_gps_lat = _xmp_gps_to_decimal(review.get("gps_latitude"), axis="lat")
                if not refresh_gps_lon:
                    refresh_gps_lon = _xmp_gps_to_decimal(review.get("gps_longitude"), axis="lon")
                refresh_page_like = bool(review.get("subphotos")) or (
                    str((refresh_detections.get("caption") or {}).get("effective_engine") or "").strip()
                    == "page-summary"
                )
                text_layers = _resolve_xmp_text_layers(
                    image_path=image_path,
                    ocr_text=refresh_ocr_text,
                    page_like=refresh_page_like,
                    ocr_authority_source=str(review.get("ocr_authority_source") or ""),
                    author_text=str(review.get("author_text") or ""),
                    scene_text=str(review.get("scene_text") or ""),
                )
                xmp_title, xmp_title_source = _compute_xmp_title(
                    image_path=image_path,
                    explicit_title=str(review.get("title") or ""),
                    title_source=str(review.get("title_source") or ""),
                    author_text=str(text_layers.get("author_text") or ""),
                )
                stat = image_path.stat()
                refresh_subphotos = review.get("subphotos")
                refresh_analysis_mode = str(
                    (existing_sidecar_state or {}).get("analysis_mode")
                    or (
                        "page_subphotos"
                        if isinstance(refresh_subphotos, list) and refresh_subphotos
                        else "single_image"
                    )
                )
                refresh_album_title = _require_album_title_for_title_page(
                    image_path=image_path,
                    album_title=_resolve_title_page_album_title(
                        image_path=image_path,
                        album_title=(
                            str(review.get("album_title") or "").strip() or _resolve_album_title_hint(image_path)
                        ),
                        ocr_text=refresh_ocr_text,
                    ),
                    context="refresh",
                )
                date_engine = (
                    self._get_date_engine(effective)
                    if date_estimation_enabled and not _has_dc_date(_dc_date_value(review))
                    else None
                )
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
                if refresh_detections is None:
                    refresh_detections = {}
                refresh_detections["processing"] = {
                    "processor_signature": PROCESSOR_SIGNATURE,
                    "settings_signature": settings_sig,
                    "cast_store_signature": (
                        current_cast_signature if bool(effective.get("enable_people", True)) else ""
                    ),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "date_estimate_input_hash": _date_estimate_input_hash(
                        refresh_ocr_text,
                        refresh_album_title,
                    ),
                    "ocr_authority_signature": str((existing_sidecar_state or {}).get("ocr_authority_signature") or ""),
                    "ocr_authority_hash": str((existing_sidecar_state or {}).get("ocr_authority_hash") or ""),
                    "analysis_mode": refresh_analysis_mode,
                }
                write_xmp_sidecar(
                    sidecar_path,
                    creator_tool=creator_tool,
                    person_names=list(review.get("person_names") or []),
                    subjects=list(review.get("subjects") or []),
                    title=xmp_title,
                    title_source=xmp_title_source,
                    description=str(review.get("description") or ""),
                    album_title=refresh_album_title,
                    gps_latitude=refresh_gps_lat,
                    gps_longitude=refresh_gps_lon,
                    location_city=str(refresh_location.get("city") or ""),
                    location_state=str(refresh_location.get("state") or ""),
                    location_country=str(refresh_location.get("country") or ""),
                    location_sublocation=str(refresh_location.get("sublocation") or ""),
                    source_text=_build_dc_source(
                        refresh_album_title,
                        image_path,
                        _page_scan_filenames(image_path),
                    ),
                    ocr_text=refresh_ocr_text,
                    author_text=str(text_layers.get("author_text") or ""),
                    scene_text=str(text_layers.get("scene_text") or ""),
                    detections_payload=refresh_detections,
                    stitch_key=str(review.get("stitch_key") or ""),
                    ocr_authority_source=str(review.get("ocr_authority_source") or ""),
                    create_date=(str(review.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
                    dc_date=refresh_dc_date,
                    date_time_original=refresh_date_time_original,
                    history_when=_xmp_timestamp_from_path(image_path),
                    image_width=_get_image_dimensions(image_path)[0],
                    image_height=_get_image_dimensions(image_path)[1],
                    ocr_ran=bool(review.get("ocr_ran")),
                    people_detected=bool(review.get("people_detected")),
                    people_identified=bool(review.get("people_identified")),
                    ocr_lang=str(review.get("ocr_lang") or ""),
                    locations_shown=refresh_detections.get("locations_shown") if refresh_detections else None,
                )

            if not self.dry_run:
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

    # ── People-update fast-path ─────────────────────────────────────────────

    def _process_people_update(  # noqa: C901
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        creator_tool: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        chain_gps: bool,
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return

        file_start = time.monotonic()
        det = state.get("detections") or {}
        existing_people_rows = [r for r in list(det.get("people") or []) if isinstance(r, dict)]
        existing_caption_payload = dict(det.get("caption") or {})
        existing_ocr_text = _effective_sidecar_ocr_text(image_path, state)
        existing_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
        existing_object_rows = [r for r in list(det.get("objects") or []) if isinstance(r, dict)]
        existing_object_labels = [str(r.get("label") or "") for r in existing_object_rows if r.get("label")]
        existing_location = _effective_sidecar_location_payload(image_path, state)

        eta_str = _format_eta(self.completed_times, len(self.files) - idx + 1)
        eta_part = f"  {eta_str}" if eta_str else ""
        prefix = f"[{idx}/{len(self.files)}]{eta_part}  {_display_work_label(image_path)}"
        print(prefix, flush=True)
        _pu_stop, _pu_step = _progress_ticker(prefix)

        try:
            _pu_step("people")
            pu_people_matches = (
                _match_people_with_cast_store_retry(
                    people_matcher=people_matcher,
                    image_path=image_path,
                    source_path=image_path,
                    bbox_offset=(0, 0),
                    hint_text=existing_ocr_text,
                )
                if people_matcher
                else []
            )
            pu_faces_detected = (
                (
                    _v
                    if isinstance(
                        _v := getattr(people_matcher, "last_faces_detected", 0),
                        int,
                    )
                    else 0
                )
                if people_matcher
                else 0
            )
            pu_people_match_names = _dedupe([r.name for r in pu_people_matches])
            _pu_step(_format_people_step_label("people", pu_people_match_names))
            pu_person_names = _dedupe(pu_people_match_names + existing_xmp_people)
            pu_album_title = _resolve_album_title_hint(image_path)
            pu_printed_title = _resolve_album_printed_title_hint(image_path, self.printed_album_title_cache)
            pu_people_payload = _serialize_people_matches(pu_people_matches)
            pu_prompt_debug = None
            people_names_changed = pu_person_names != existing_xmp_people
            if not people_names_changed:
                pu_updated_det = {
                    **det,
                    "people": pu_people_payload or existing_people_rows,
                    "caption": existing_caption_payload,
                }
            else:
                caption_key = (
                    str(effective.get("caption_engine", self.defaults["caption_engine"])),
                    str(effective.get("caption_model", self.defaults["caption_model"])),
                    str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
                    int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
                    float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
                    str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
                    int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
                )
                pu_caption_engine = self.caption_engine_cache.get(caption_key)
                if pu_caption_engine is None:
                    pu_caption_engine = _init_caption_engine(
                        engine=caption_key[0],
                        model_name=caption_key[1],
                        caption_prompt=caption_key[2],
                        max_tokens=int(caption_key[3]),
                        temperature=float(caption_key[4]),
                        lmstudio_base_url=caption_key[5],
                        max_image_edge=int(caption_key[6]),
                        stream=True,
                    )
                    self.caption_engine_cache[caption_key] = pu_caption_engine
                pu_people_positions = _compute_people_positions(pu_people_matches, image_path)
                _pu_step("caption")
                pu_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
                with _prepare_ai_model_image(image_path) as pu_model_path:
                    pu_caption_out = pu_caption_engine.generate(
                        image_path=pu_model_path,
                        people=pu_person_names,
                        objects=existing_object_labels,
                        ocr_text=existing_ocr_text,
                        source_path=image_path,
                        album_title=pu_album_title,
                        printed_album_title=pu_printed_title,
                        people_positions=pu_people_positions,
                        debug_recorder=pu_prompt_debug.record,
                        debug_step="caption_refresh",
                    )
                    pu_faces_detected = (
                        (_v if isinstance(_v := getattr(people_matcher, "last_faces_detected", 0), int) else 0)
                        if people_matcher
                        else 0
                    )
                (
                    pu_local_people_present,
                    pu_local_estimated_people_count,
                ) = _estimate_people_from_detections(
                    people_matches=pu_people_matches,
                    people_names=pu_person_names,
                    object_labels=existing_object_labels,
                    faces_detected=pu_faces_detected,
                )
                (
                    pu_people_present,
                    pu_estimated_people_count,
                ) = _resolve_people_count_metadata(
                    requested_caption_engine=str(caption_key[0]),
                    caption_engine=pu_caption_engine,
                    model_image_path=pu_model_path,
                    people=pu_person_names,
                    objects=existing_object_labels,
                    ocr_text=existing_ocr_text,
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
                pu_ocr_model = str(
                    dict(det.get("ocr") or {}).get("model")
                    or (
                        effective.get("ocr_model", self.defaults["ocr_model"])
                        if str(effective.get("ocr_engine", self.defaults["ocr_engine"])).strip().lower()
                        in {"local", "lmstudio"}
                        else ""
                    )
                )
                pu_updated_det = _refresh_detection_model_metadata(
                    {
                        **det,
                        "people": pu_people_payload,
                        "caption": pu_caption_payload,
                    },
                    ocr_model=pu_ocr_model,
                    caption_model=(
                        str(pu_caption_engine.effective_model_name)
                        if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
                        else ""
                    ),
                )
            pu_subjects = _dedupe(
                existing_object_labels + existing_ocr_keywords + ([pu_album_title] if pu_album_title else [])
            )

            pu_people_detected = pu_faces_detected > 0 or len(pu_person_names) > 0
            pu_people_identified = len(pu_person_names) > 0

            if not self.dry_run:
                pu_album_title = _require_album_title_for_title_page(
                    image_path=image_path,
                    album_title=_resolve_title_page_album_title(
                        image_path=image_path,
                        album_title=pu_album_title,
                        ocr_text=existing_ocr_text,
                    ),
                    context="people update",
                )
                date_engine = (
                    self._get_date_engine(effective)
                    if date_estimation_enabled and not _has_dc_date(_dc_date_value(state))
                    else None
                )
                pu_dc_date = _resolve_dc_date(
                    existing_dc_date=_dc_date_value(state),
                    ocr_text=existing_ocr_text,
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
                    ocr_text=existing_ocr_text,
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
                current_cast_signature = str(people_matcher.store_signature())
                pu_proc = dict((pu_updated_det.get("processing") or {}))
                pu_proc["cast_store_signature"] = current_cast_signature
                if date_estimation_enabled or pu_dc_date:
                    pu_proc["date_estimate_input_hash"] = _date_estimate_input_hash(
                        existing_ocr_text,
                        pu_album_title,
                    )
                if existing_location:
                    pu_updated_det["location"] = existing_location
                pu_updated_det = {**pu_updated_det, "processing": pu_proc}
                _write_sidecar_and_record(
                    sidecar_path,
                    image_path,
                    creator_tool=creator_tool,
                    person_names=pu_person_names,
                    subjects=pu_subjects,
                    title=xmp_title,
                    title_source=xmp_title_source,
                    description=str(state.get("description") or ""),
                    album_title=pu_album_title,
                    location_payload=existing_location,
                    source_text=pu_source_text,
                    ocr_text=existing_ocr_text,
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

            _pu_stop()
            if not chain_gps:
                self.processed += 1
                self.completed_times.append(time.monotonic() - file_start)
                eta_str2 = _format_eta(self.completed_times, len(self.files) - idx)
                eta_part2 = f"  {eta_str2}" if eta_str2 else ""
                print(
                    f"[{idx}/{len(self.files)}]{eta_part2}  ok    {image_path.name}",
                    flush=True,
                )
        except Exception as exc:
            self.failures += 1
            _pu_stop()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    # ── GPS-update path ─────────────────────────────────────────────────────

    def _process_gps_update(
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        creator_tool: str,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
    ) -> None:
        state = existing_sidecar_state
        if not isinstance(state, dict):
            return
        file_start = time.monotonic()
        det = state.get("detections") or {}
        gps_ocr_text = _effective_sidecar_ocr_text(image_path, state)
        gps_ocr_keywords = list((det.get("ocr") or {}).get("keywords") or [])
        gps_people_names = _dedupe(
            [str(r.get("name") or "") for r in list(det.get("people") or []) if isinstance(r, dict) and r.get("name")]
        )
        gps_object_labels = [
            str(r.get("label") or "") for r in list(det.get("objects") or []) if isinstance(r, dict) and r.get("label")
        ]
        gps_album_title = str(state.get("album_title") or "").strip()
        gps_printed_title = _resolve_album_printed_title_hint(image_path, self.printed_album_title_cache)
        gps_existing_location_name = str((dict(det.get("location") or {})).get("query") or "").strip()

        eta_str = _format_eta(self.completed_times, len(self.files) - idx + 1)
        eta_part = f"  {eta_str}" if eta_str else ""
        prefix = f"[{idx}/{len(self.files)}]{eta_part}  {_display_work_label(image_path)}"
        print(prefix, flush=True)
        _gps_stop, _gps_step = _progress_ticker(prefix)

        try:
            caption_key = (
                str(effective.get("caption_engine", self.defaults["caption_engine"])),
                str(effective.get("caption_model", self.defaults["caption_model"])),
                str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
                int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
                float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
                str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
                int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
            )
            gps_caption_engine = self.caption_engine_cache.get(caption_key)
            if gps_caption_engine is None:
                gps_caption_engine = _init_caption_engine(
                    engine=caption_key[0],
                    model_name=caption_key[1],
                    caption_prompt=caption_key[2],
                    max_tokens=int(caption_key[3]),
                    temperature=float(caption_key[4]),
                    lmstudio_base_url=caption_key[5],
                    max_image_edge=int(caption_key[6]),
                    stream=True,
                )
                self.caption_engine_cache[caption_key] = gps_caption_engine

            gps_prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))
            _gps_step("location")
            with _prepare_ai_model_image(image_path) as gps_model_path:
                gps_latitude, gps_longitude, location_name = _resolve_location_metadata(
                    requested_caption_engine=str(caption_key[0]),
                    caption_engine=gps_caption_engine,
                    model_image_path=gps_model_path,
                    people=gps_people_names,
                    objects=gps_object_labels,
                    ocr_text=gps_ocr_text,
                    source_path=image_path,
                    album_title=gps_album_title,
                    printed_album_title=gps_printed_title,
                    people_positions={},
                    fallback_location_name=gps_existing_location_name,
                    prompt_debug=gps_prompt_debug,
                    debug_step="location_gps_step",
                )
                _gps_step("locations_shown")
                gps_locations_shown, gps_locations_shown_ran = _resolve_locations_shown(
                    requested_caption_engine=str(caption_key[0]),
                    caption_engine=gps_caption_engine,
                    model_image_path=gps_model_path,
                    ocr_text=gps_ocr_text,
                    source_path=image_path,
                    album_title=gps_album_title,
                    printed_album_title=gps_printed_title,
                    geocoder=self.geocoder,
                    prompt_debug=gps_prompt_debug,
                    debug_step="locations_shown_gps_step",
                    artifact_recorder=(lambda record: _append_geocode_artifact(image_path=image_path, record=record)),
                )
            gps_location_payload = _resolve_location_payload(
                geocoder=self.geocoder,
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
                location_name=location_name,
                artifact_recorder=(lambda record: _append_geocode_artifact(image_path=image_path, record=record)),
                artifact_step="location_gps_step",
            )
            gps_location_payload, _ = _apply_title_page_location_config(
                image_path=image_path,
                location_payload=gps_location_payload,
                title_page_location=self.title_page_location,
            )
            _emit_prompt_debug_artifact(gps_prompt_debug, dry_run=self.dry_run)

            if not self.dry_run:
                gps_updated_det = {**det}
                if gps_location_payload:
                    gps_updated_det["location"] = gps_location_payload
                elif "location" in gps_updated_det:
                    del gps_updated_det["location"]
                gps_updated_det["locations_shown"] = gps_locations_shown
                gps_updated_det["location_shown_ran"] = gps_locations_shown_ran
                gps_subjects = _dedupe(
                    gps_object_labels + gps_ocr_keywords + ([gps_album_title] if gps_album_title else [])
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
                    creator_tool=creator_tool,
                    person_names=list(existing_xmp_people),
                    subjects=gps_subjects,
                    title=xmp_title,
                    title_source=xmp_title_source,
                    description=str(state.get("description") or ""),
                    album_title=gps_album_title,
                    location_payload=gps_location_payload,
                    source_text=str(state.get("source_text") or ""),
                    ocr_text=gps_ocr_text,
                    ocr_lang=str(state.get("ocr_lang") or ""),
                    author_text=str(state.get("author_text") or ""),
                    scene_text=str(state.get("scene_text") or ""),
                    detections_payload=gps_updated_det,
                    stitch_key=str(state.get("stitch_key") or ""),
                    ocr_authority_source=str(state.get("ocr_authority_source") or ""),
                    create_date=(str(state.get("create_date") or "").strip() or read_embedded_create_date(image_path)),
                    dc_date=_dc_date_value(state),
                    date_time_original=str(state.get("date_time_original") or ""),
                    ocr_ran=bool(state.get("ocr_ran")),
                    people_detected=bool(state.get("people_detected")),
                    people_identified=bool(state.get("people_identified")),
                    title_page_location=self.title_page_location,
                )

            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            _gps_stop()
            eta_str2 = _format_eta(self.completed_times, len(self.files) - idx)
            eta_part2 = f"  {eta_str2}" if eta_str2 else ""
            print(
                f"[{idx}/{len(self.files)}]{eta_part2}  ok    {image_path.name}  [gps]",
                flush=True,
            )
        except Exception as exc:
            self.failures += 1
            _gps_stop()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")

    # ── Full processing path ────────────────────────────────────────────────

    def _process_full(  # noqa: C901
        self,
        idx: int,
        image_path: Path,
        sidecar_path: Path,
        effective: dict[str, Any],
        settings_sig: str,
        creator_tool: str,
        date_estimation_enabled: bool,
        existing_sidecar_state: dict | None,
        existing_xmp_people: list[str],
        people_matcher: Any,
        current_cast_signature: str,
        archive_stitched_ocr_required: bool,
        multi_scan_group_paths: list[Path],
        multi_scan_group_signature: str,
    ) -> None:
        file_start = time.monotonic()
        stop_ticker = None
        set_step = None
        if not self.stdout_only:
            eta_str = _format_eta(self.completed_times, len(self.files) - idx + 1)
            eta_part = f"  {eta_str}" if eta_str else ""
            prefix = f"[{idx}/{len(self.files)}]{eta_part}  {_display_work_label(image_path)}"
            print(prefix, flush=True)
            stop_ticker, set_step = _progress_ticker(prefix)
        album_title_hint = _resolve_album_title_hint(image_path)
        printed_album_title_hint = _resolve_album_printed_title_hint(image_path, self.printed_album_title_cache)
        upstream_page_state = _resolve_upstream_page_sidecar_state(image_path)
        upstream_context_ocr = str((upstream_page_state or {}).get("ocr_text") or "").strip()
        upstream_location_hint = _format_location_hint_from_state(upstream_page_state)
        if not album_title_hint:
            album_title_hint = str((upstream_page_state or {}).get("album_title") or "").strip()
        if not printed_album_title_hint:
            printed_album_title_hint = str((upstream_page_state or {}).get("album_title") or "").strip()
        prompt_debug = PromptDebugSession(image_path, label=_display_work_label(image_path))

        try:
            object_detector = None
            if bool(effective.get("enable_objects", True)):
                object_key = (
                    str(effective.get("model", self.defaults["model"])),
                    float(effective.get("object_threshold", self.defaults["object_threshold"])),
                )
                object_detector = self.object_detector_cache.get(object_key)
                if object_detector is None:
                    object_detector = _init_object_detector(
                        model_name=str(object_key[0]),
                        confidence=float(object_key[1]),
                    )
                    self.object_detector_cache[object_key] = object_detector

            caption_key = (
                str(effective.get("caption_engine", self.defaults["caption_engine"])),
                str(effective.get("caption_model", self.defaults["caption_model"])),
                str(effective.get("caption_prompt", self.defaults["caption_prompt"])),
                int(effective.get("caption_max_tokens", self.defaults["caption_max_tokens"])),
                float(effective.get("caption_temperature", self.defaults["caption_temperature"])),
                str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"])),
                int(effective.get("caption_max_edge", self.defaults["caption_max_edge"])),
            )
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
                    stream=not self.stdout_only,
                )
                self.caption_engine_cache[caption_key] = caption_engine

            ocr_key = (
                str(effective.get("ocr_engine", self.defaults["ocr_engine"])),
                str(effective.get("ocr_lang", self.defaults["ocr_lang"])),
                str(effective.get("ocr_model", self.defaults["ocr_model"])),
                normalize_lmstudio_base_url(
                    str(effective.get("lmstudio_base_url", self.defaults["lmstudio_base_url"]))
                ),
            )
            ocr_engine = self.ocr_engine_cache.get(ocr_key)
            if ocr_engine is None:
                ocr_engine = OCREngine(
                    engine=ocr_key[0],
                    language=ocr_key[1],
                    model_name=ocr_key[2],
                    base_url=ocr_key[3],
                )
                self.ocr_engine_cache[ocr_key] = ocr_engine

            scan_ocr_authority: ArchiveScanOCRAuthority | None = None
            if archive_stitched_ocr_required:
                scan_ocr_authority = _resolve_archive_scan_authoritative_ocr(
                    image_path=image_path,
                    group_paths=multi_scan_group_paths,
                    group_signature=multi_scan_group_signature,
                    cache=self.archive_scan_ocr_cache,
                    ocr_engine=ocr_engine,
                    step_fn=set_step,
                    stitched_image_dir=self.stitch_cap_dir,
                    debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
                )

            with prepare_image_layout(
                image_path,
                split_mode="off",
            ) as layout:
                person_names: list[str]
                subjects: list[str]
                description: str
                ocr_text: str
                payload: dict[str, Any]
                subphotos_xml: list[dict[str, Any]] | None = None
                analysis_mode = "single_image"
                _scan_filenames = _page_scan_filenames(image_path)
                printed_album_title_hint = album_title_hint

                _stitched_cap_path = scan_ocr_authority.stitched_image_path if scan_ocr_authority is not None else None
                derived_ocr_override = _effective_sidecar_ocr_text(image_path, existing_sidecar_state)
                analysis_target = _stitched_cap_path or (layout.content_path if layout.page_like else image_path)
                people_analysis_source = (
                    analysis_target
                    if scan_ocr_authority is not None
                    else (layout.content_path if layout.page_like else image_path)
                )
                analysis = _run_image_analysis(
                    image_path=analysis_target,
                    people_image_path=people_analysis_source,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine=str(caption_key[0]),
                    ocr_engine_name=ocr_key[0],
                    ocr_language=ocr_key[1],
                    people_source_path=people_analysis_source,
                    people_bbox_offset=(_bounds_offset(layout.content_bounds) if layout.page_like else (0, 0)),
                    caption_source_path=(image_path if layout.page_like else analysis_target),
                    album_title=album_title_hint,
                    printed_album_title=printed_album_title_hint,
                    geocoder=self.geocoder,
                    step_fn=set_step,
                    extra_people_names=existing_xmp_people,
                    is_page_scan=layout.page_like,
                    ocr_text_override=(
                        scan_ocr_authority.ocr_text
                        if scan_ocr_authority is not None
                        else (derived_ocr_override or None)
                    ),
                    context_ocr_text=upstream_context_ocr,
                    context_location_hint=upstream_location_hint,
                    prompt_debug=prompt_debug,
                    title_page_location=self.title_page_location,
                )
                resolved_album_title = analysis.album_title or album_title_hint
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
                ocr_text = analysis.ocr_text
                payload = _build_flat_payload(layout, analysis)
                analysis_mode = "page_flat" if layout.page_like else "single_image"
                ocr_authority_hash = str(scan_ocr_authority.ocr_hash) if scan_ocr_authority is not None else ""

                payload = _refresh_detection_model_metadata(
                    payload,
                    ocr_model=(
                        str(ocr_engine.effective_model_name)
                        if str(ocr_key[0]).strip().lower() in {"local", "lmstudio"}
                        else ""
                    ),
                    caption_model=(
                        str(caption_engine.effective_model_name)
                        if str(caption_key[0]).strip().lower() in {"local", "lmstudio"}
                        else ""
                    ),
                )

                _ocr_ran_flag = str(effective.get("ocr_engine", self.defaults["ocr_engine"])).lower() != "none"
                _people_detected_flag = analysis.faces_detected > 0 or len(person_names) > 0
                _people_identified_flag = len(person_names) > 0

                if not self.dry_run:
                    location_payload = dict(payload.get("location") or {}) if isinstance(payload, dict) else {}
                    effective_location_payload = _effective_sidecar_location_payload(image_path, existing_sidecar_state)
                    if effective_location_payload:
                        location_payload = effective_location_payload
                    if location_payload:
                        payload["location"] = location_payload
                    img_w, img_h = _get_image_dimensions(image_path)
                    final_album_title = _require_album_title_for_title_page(
                        image_path=image_path,
                        album_title=_resolve_title_page_album_title(
                            image_path=image_path,
                            album_title=(resolved_album_title or _resolve_album_title_hint(image_path)),
                            ocr_text=ocr_text,
                        ),
                        context="write",
                    )
                    date_engine = (
                        self._get_date_engine(effective)
                        if date_estimation_enabled and not _has_dc_date(_dc_date_value(existing_sidecar_state))
                        else None
                    )
                    final_dc_date = _resolve_dc_date(
                        existing_dc_date=_dc_date_value(existing_sidecar_state),
                        ocr_text=ocr_text,
                        album_title=final_album_title,
                        image_path=image_path,
                        date_engine=date_engine,
                        prompt_debug=prompt_debug,
                    )
                    final_date_time_original = _resolve_date_time_original(
                        dc_date=final_dc_date,
                        date_time_original=str((existing_sidecar_state or {}).get("date_time_original") or ""),
                    )
                    text_layers = _resolve_xmp_text_layers(
                        image_path=image_path,
                        ocr_text=ocr_text,
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
                    subphotos_xml = [
                        {
                            "index": i + 1,
                            "bounds": {
                                "x": round(r["x"] * img_w),
                                "y": round(r["y"] * img_h),
                                "width": round(r["w"] * img_w),
                                "height": round(r["h"] * img_h),
                            },
                            "description": r.get("author_text", ""),
                            "author_text": r.get("author_text", ""),
                            "scene_text": r.get("scene_text", ""),
                            "people": [],
                            "subjects": [],
                        }
                        for i, r in enumerate(analysis.image_regions or [])
                    ] or None
                    if people_matcher is not None:
                        current_cast_signature = str(people_matcher.store_signature())
                    stat = image_path.stat()
                    payload["processing"] = {
                        "processor_signature": PROCESSOR_SIGNATURE,
                        "settings_signature": settings_sig,
                        "cast_store_signature": (
                            current_cast_signature if bool(effective.get("enable_people", True)) else ""
                        ),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "date_estimate_input_hash": (
                            _date_estimate_input_hash(ocr_text, final_album_title)
                            if date_estimation_enabled or final_dc_date
                            else str((existing_sidecar_state or {}).get("date_estimate_input_hash") or "")
                        ),
                        "ocr_authority_signature": (
                            str(scan_ocr_authority.signature) if scan_ocr_authority is not None else ""
                        ),
                        "ocr_authority_hash": ocr_authority_hash,
                        "analysis_mode": str(analysis_mode),
                    }
                    _write_sidecar_and_record(
                        sidecar_path,
                        image_path,
                        creator_tool=creator_tool,
                        person_names=person_names,
                        subjects=subjects,
                        title=xmp_title,
                        title_source=xmp_title_source,
                        description=description,
                        album_title=final_album_title,
                        location_payload=location_payload,
                        source_text=_build_dc_source(
                            final_album_title,
                            image_path,
                            _scan_filenames,
                        ),
                        ocr_text=ocr_text,
                        ocr_lang=str(analysis.ocr_lang or ""),
                        author_text=str(text_layers.get("author_text") or ""),
                        scene_text=str(text_layers.get("scene_text") or ""),
                        detections_payload=payload,
                        subphotos=subphotos_xml,
                        ocr_authority_source=("archive_stitched" if scan_ocr_authority is not None else ""),
                        create_date=read_embedded_create_date(image_path),
                        dc_date=final_dc_date,
                        date_time_original=final_date_time_original,
                        ocr_ran=_ocr_ran_flag,
                        people_detected=_people_detected_flag,
                        people_identified=_people_identified_flag,
                        title_page_location=self.title_page_location,
                    )

            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            self.processed += 1
            self.completed_times.append(time.monotonic() - file_start)
            if stop_ticker is not None:
                stop_ticker()
            if self.stdout_only:
                caption_meta = dict(payload.get("caption") or {}) if isinstance(payload, dict) else {}
                fallback_error = str(caption_meta.get("error") or "").strip()
                if bool(caption_meta.get("fallback")) and fallback_error:
                    self.emit_error(
                        f"[{idx}/{len(self.files)}] warn  {image_path.name}: caption fallback: {fallback_error}"
                    )
                print(f"{image_path.name}: {description}" if description else image_path.name)
            else:
                eta_str = _format_eta(self.completed_times, len(self.files) - idx)
                eta_part = f"  {eta_str}" if eta_str else ""
                print(
                    f"[{idx}/{len(self.files)}]{eta_part}  ok    {image_path.name}",
                    flush=True,
                )
            _mirror_page_sidecars(image_path)
        except Exception as exc:
            self.failures += 1
            _emit_prompt_debug_artifact(prompt_debug, dry_run=self.dry_run)
            if stop_ticker is not None:
                stop_ticker()
            self.emit_error(f"[{idx}/{len(self.files)}] fail  {image_path.name}: {exc}")
