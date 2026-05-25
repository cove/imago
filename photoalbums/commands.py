from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _call_main(func) -> int:
    try:
        result = func()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        print(code)
        return 1
    if isinstance(result, int):
        return int(result)
    return 0


def run_ai_index(argv: list[str]) -> int:
    from .lib import ai_index

    return int(ai_index.run(argv) or 0)


def run_apply_metadata() -> int:
    import apply_metadata

    return _call_main(apply_metadata.main)


def run_create_metadata_tsv() -> int:
    import create_metadata_tsv

    return _call_main(create_metadata_tsv.main)


def run_metadata_map(*, paths: list[str], port: int) -> int:
    from . import map_server

    try:
        map_server.run_server(paths, port=port)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def run_compress_tiff() -> int:
    import compress_tiff

    return _call_main(compress_tiff.main)


def run_render() -> int:
    import stitch_oversized_pages

    return _call_main(stitch_oversized_pages.main)


def run_stitch_validate() -> int:
    import stitch_oversized_pages_validate

    return _call_main(stitch_oversized_pages_validate.main)


def run_watch_incoming(*, album_id: str = "") -> int:
    import incoming_scans_watcher

    return _call_main(lambda: incoming_scans_watcher.main(album_id=album_id))


def run_checksum_tree(*, base_dir: str, verify: bool) -> int:
    import sha3_tree_hashes

    argv = [str(base_dir)]
    if verify:
        argv.append("--verify")
    return int(sha3_tree_hashes.run(argv) or 0)


def _run_crop_regions_for_view(
    view_path,
    *,
    root,
    model_name,
    photos_dir,
    force,
    skip_restoration,
    force_restoration,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
    associate_captions,
    clear_pipeline_steps,
    detect_regions,
    read_pipeline_step,
    validate_region_set,
    write_pipeline_step,
    write_region_list,
    crop_page_regions,
) -> int:
    """Process a single view path through the crop-regions pipeline. Returns 1 on error, 0 on success."""
    try:
        _ensure_crop_regions_inputs(
            view_path,
            root=root,
            model_name=model_name,
            _has_xmp_regions=_has_xmp_regions,
            _image_dimensions=_image_dimensions,
            _read_regions_from_xmp=_read_regions_from_xmp,
            associate_captions=associate_captions,
            clear_pipeline_steps=clear_pipeline_steps,
            detect_regions=detect_regions,
            read_pipeline_step=read_pipeline_step,
            validate_region_set=validate_region_set,
            write_pipeline_step=write_pipeline_step,
            write_region_list=write_region_list,
        )
        n = crop_page_regions(
            view_path,
            photos_dir,
            force=force,
            skip_restoration=skip_restoration,
            force_restoration=force_restoration,
        )
        if n > 0:
            print(f"  Wrote {n} crop(s) to {photos_dir.name}/")
        return 0
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1


def run_crop_regions(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
    force: bool,
    skip_restoration: bool = False,
    force_restoration: bool = False,
) -> int:
    """Run only the crop-regions step for matching album page view JPEGs."""
    from pathlib import Path

    from .lib.ai_model_settings import default_view_region_model
    from .lib.ai_photo_crops import crop_page_regions
    from .lib.ai_view_regions import (
        _has_xmp_regions,
        _image_dimensions,
        _read_regions_from_xmp,
        associate_captions,
        detect_regions,
        validate_region_set,
    )
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step, write_region_list
    from .naming import archive_dir_for_album_dir, is_pages_dir
    from .stitch_oversized_pages import get_photos_dirname

    if skip_restoration and force_restoration:
        print("Error: --skip-restoration and --force-restoration cannot be used together", file=sys.stderr)
        return 2

    root = Path(photos_root)
    album_id_lower = album_id.casefold()
    model_name = default_view_region_model()

    view_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and is_pages_dir(d) and (not album_id or album_id_lower in d.name.casefold())
    )
    if not view_dirs:
        print(f"No _Pages directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    errors = 0
    for view_dir in view_dirs:
        archive_path = archive_dir_for_album_dir(view_dir)
        photos_dir = Path(get_photos_dirname(archive_path))

        candidates = _iter_page_view_targets(view_dir, page)

        for view_path in candidates:
            print(f"Processing {view_path.name}...")
            errors += _run_crop_regions_for_view(
                view_path,
                root=root,
                model_name=model_name,
                photos_dir=photos_dir,
                force=force,
                skip_restoration=skip_restoration,
                force_restoration=force_restoration,
                _has_xmp_regions=_has_xmp_regions,
                _image_dimensions=_image_dimensions,
                _read_regions_from_xmp=_read_regions_from_xmp,
                associate_captions=associate_captions,
                clear_pipeline_steps=clear_pipeline_steps,
                detect_regions=detect_regions,
                read_pipeline_step=read_pipeline_step,
                validate_region_set=validate_region_set,
                write_pipeline_step=write_pipeline_step,
                write_region_list=write_region_list,
                crop_page_regions=crop_page_regions,
            )

    return 1 if errors else 0


def _crop_regions_detect_decision(
    view_path: Path,
    *,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
    clear_pipeline_steps,
    validate_region_set,
) -> tuple[bool, str]:
    xmp_path = view_path.with_suffix(".xmp")
    needs_detect = not _view_regions_step_complete(xmp_path)
    if needs_detect or not _has_xmp_regions(xmp_path):
        return needs_detect, ""
    img_w, img_h = _image_dimensions(view_path)
    stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
    vresult = validate_region_set(stored, img_w=img_w, img_h=img_h)
    if vresult.valid:
        return False, ""
    reasons = ", ".join(f.reason for f in vresult.failures)
    clear_pipeline_steps(xmp_path, ["view_regions"])
    return True, f" (revalidate: {len(vresult.failures)} invalid region(s): {reasons})"


def _ensure_crop_regions_inputs(
    view_path: Path,
    *,
    root: Path,
    model_name: str,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
    associate_captions,
    clear_pipeline_steps,
    detect_regions,
    read_pipeline_step,
    validate_region_set,
    write_pipeline_step,
    write_region_list,
) -> None:
    xmp_path = view_path.with_suffix(".xmp")
    needs_detect, detect_reason = _crop_regions_detect_decision(
        view_path,
        _has_xmp_regions=_has_xmp_regions,
        _image_dimensions=_image_dimensions,
        _read_regions_from_xmp=_read_regions_from_xmp,
        clear_pipeline_steps=clear_pipeline_steps,
        validate_region_set=validate_region_set,
    )
    if not needs_detect:
        return
    if not detect_reason:
        detect_reason = (
            " (pipeline state present but regions are missing)"
            if read_pipeline_step(xmp_path, "view_regions") is not None
            else " (generating missing page regions)"
        )
    print(f"  detect-regions:{detect_reason}")
    img_w, img_h = _image_dimensions(view_path)
    album_context, page_caption, people_roster = _build_region_detection_context(view_path, root)
    regions = detect_regions(
        view_path,
        force=True,
        album_context=album_context,
        page_caption=page_caption,
        people_roster=people_roster,
    )
    _write_crop_regions_detection_result(
        xmp_path,
        regions,
        img_w=img_w,
        img_h=img_h,
        model_name=model_name,
        associate_captions=associate_captions,
        read_pipeline_step=read_pipeline_step,
        write_pipeline_step=write_pipeline_step,
        write_region_list=write_region_list,
    )


def _write_crop_regions_detection_result(
    xmp_path: Path,
    regions,
    *,
    img_w: int,
    img_h: int,
    model_name: str,
    associate_captions,
    read_pipeline_step,
    write_pipeline_step,
    write_region_list,
) -> None:
    if regions:
        regions_with_captions = associate_captions(regions, [], img_w)
        write_region_list(xmp_path, regions_with_captions, img_w, img_h)
        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
        print(f"  detect-regions: {len(regions)} region(s)")
        return
    write_region_list(xmp_path, [], img_w, img_h)
    existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
    if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
    print("  detect-regions: no regions")


def _is_derived_view(filename: str) -> bool:
    """Return True if filename is a derived _D##-##_V.jpg output (not a page view)."""
    import re

    return bool(re.search(r"_D\d{2}-\d{2}_V\b", filename))



def _write_view_regions_debug_artifact(prompt_debug, *, image_path: Path) -> Path | None:
    if prompt_debug is None or not prompt_debug.has_steps():
        return None
    from .lib.prompt_debug import debug_root_for_image_path

    debug_path = debug_root_for_image_path(image_path) / f"{image_path.stem}.view-regions.debug.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(prompt_debug.to_artifact(), ensure_ascii=False, indent=2), encoding="utf-8")
    return debug_path


def _is_title_page_view(path: Path) -> bool:
    return "_P01_" in path.name and not _is_derived_view(path.name)


def _iter_page_view_targets(view_dir: Path, page: str | None) -> list[Path]:
    candidates = sorted(path for path in view_dir.glob("*_V.jpg") if not _is_derived_view(path.name))
    candidates = [path for path in candidates if not _is_title_page_view(path)]
    if page is None:
        return candidates
    page_token = f"_P{str(page).zfill(2)}_"
    return [path for path in candidates if page_token in path.name]


def _iter_face_refresh_targets(view_dir: Path, photos_dir: Path, page: str | None) -> list[Path]:
    page_token = f"_P{str(page).zfill(2)}_" if page is not None else ""
    derived_view_targets = sorted(
        path
        for path in view_dir.glob("*_V.jpg")
        if _is_derived_view(path.name) and (not page_token or page_token in path.name)
    )
    crop_targets = []
    if photos_dir.is_dir():
        crop_targets = sorted(
            path
            for path in photos_dir.glob("*_V.jpg")
            if _is_derived_view(path.name) and (not page_token or page_token in path.name)
        )
    return [*_iter_page_view_targets(view_dir, page), *derived_view_targets, *crop_targets]


def _build_region_detection_context(view_path: Path, photos_root: Path) -> tuple[str, str, dict[str, str]]:
    from .lib.album_sets import find_archive_set_by_photos_root, read_people_roster
    from .lib.xmp_sidecar import read_ai_sidecar_state
    from .naming import parse_album_filename

    collection, year, book, page = parse_album_filename(view_path.name)
    album_context = ""
    if collection != "Unknown" and year != "Unknown":
        album_context = f"{collection} {year}"
        if str(book or "").strip():
            album_context = album_context + f", book {book}"
        if str(page or "").isdigit():
            album_context = album_context + f", page {int(page):02d}"
    sidecar_state = read_ai_sidecar_state(view_path.with_suffix(".xmp")) or {}
    page_caption = str(sidecar_state.get("description") or "").strip()
    album_set = find_archive_set_by_photos_root(photos_root)
    people_roster = read_people_roster(album_set)
    return album_context, page_caption, people_roster


def _view_regions_step_complete(xmp_path: Path) -> bool:
    from .lib.ai_view_regions import _has_xmp_regions
    from .lib.xmp_sidecar import read_pipeline_step

    pipeline_state = read_pipeline_step(xmp_path, "view_regions")
    if pipeline_state is None:
        return False
    if str(pipeline_state.get("result") or "").strip() in {"no_regions", "validation_failed", "failed"}:
        return True
    return _has_xmp_regions(xmp_path)


def run_face_refresh(*, album_id: str, photos_root: str, page: str | None, force: bool) -> int:
    from .lib.ai_render_face_refresh import FaceRefreshSkipped, RenderFaceRefreshSession
    from .naming import archive_dir_for_album_dir, is_pages_dir
    from .stitch_oversized_pages import get_photos_dirname

    root = Path(photos_root)
    album_id_lower = album_id.casefold()

    view_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and is_pages_dir(d) and (not album_id or album_id_lower in d.name.casefold())
    )
    if not view_dirs:
        print(f"No _Pages directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    targets: list[Path] = []
    for view_dir in view_dirs:
        archive_path = archive_dir_for_album_dir(view_dir)
        photos_dir = Path(get_photos_dirname(archive_path))
        targets.extend(_iter_face_refresh_targets(view_dir, photos_dir, page))

    if not targets:
        print("No matching rendered JPEGs found for face refresh.", file=sys.stderr)
        return 1

    session = RenderFaceRefreshSession(photos_root=root)
    session.set_files(targets)

    errors = 0
    for image_path in targets:
        try:
            session.refresh_face_regions(image_path, image_path.with_suffix(".xmp"), force=force)
        except FaceRefreshSkipped as exc:
            print(f"WARNING: {exc}")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


def _render_pipeline_render_page(
    *,
    group,
    primary_scan: Path,
    view_dir: Path,
    photos_dir: Path,
    current_page_token: str,
    archive: Path,
    deps: dict,
) -> None:
    if len(group) > 1:
        deps["stitch"](group, str(view_dir))
    else:
        deps["tif_to_jpg"](str(primary_scan), str(view_dir))
    for derived in deps["list_derived_images"](archive):
        if current_page_token in Path(derived).name:
            deps["derived_to_jpg"](derived, str(photos_dir))


def _check_skip_detect_regions(
    xmp_path: Path, view_path: Path, *, force: bool, skip_validation: bool, summary: dict, deps: dict
) -> bool:
    if force:
        deps["clear_pipeline_steps"](xmp_path, ["view_regions"])
        return False
    if not _view_regions_step_complete(xmp_path):
        return False
    if not deps["_has_xmp_regions"](xmp_path):
        summary["detect_regions_skipped"] += 1
        return True
    img_w, img_h = deps["_image_dimensions"](view_path)
    stored = deps["_read_regions_from_xmp"](xmp_path, img_w, img_h)
    vresult = deps["validate_region_set"](stored, img_w=img_w, img_h=img_h)
    if vresult.valid or skip_validation:
        summary["detect_regions_skipped"] += 1
        return True
    n_fail = len(vresult.failures)
    reasons = ", ".join(f.reason for f in vresult.failures)
    print(f"  detect-regions: revalidate: {n_fail} invalid region(s): {reasons}")
    deps["clear_pipeline_steps"](xmp_path, ["view_regions"])
    return False


def _write_detect_regions_result(
    xmp_path: Path,
    view_path: Path,
    regions: list,
    *,
    img_w: int,
    img_h: int,
    model_name: str,
    debug: bool,
    summary: dict,
    deps: dict,
) -> None:
    if regions:
        regions_with_captions = deps["associate_captions"](regions, [], img_w)
        deps["write_region_list"](xmp_path, regions_with_captions, img_w, img_h)
        deps["write_pipeline_step"](xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
        summary["detect_regions_found"] += 1
        print(f"  detect-regions: {len(regions)} region(s)")
    else:
        deps["write_region_list"](xmp_path, [], img_w, img_h)
        existing_step = deps["read_pipeline_step"](xmp_path, "view_regions") or {}
        if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
            deps["write_pipeline_step"](xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
        summary["detect_regions_no_regions"] += 1
        if debug:
            failed_debug_path = deps["_failed_regions_debug_path"](view_path)
            if failed_debug_path.is_file():
                print(f"  detect-regions: failed boxes -> {failed_debug_path}")
        print("  detect-regions: no regions")


def _render_pipeline_detect_regions(
    *,
    view_path: Path,
    xmp_path: Path,
    root: Path,
    model_name: str,
    force: bool,
    debug: bool,
    skip_validation: bool,
    summary: dict[str, int],
    deps: dict,
) -> None:
    if _check_skip_detect_regions(
        xmp_path, view_path, force=force, skip_validation=skip_validation, summary=summary, deps=deps
    ):
        return

    if not force and deps["read_pipeline_step"](xmp_path, "view_regions") is not None:
        summary["detect_regions_reran"] += 1
        print("  detect-regions: re-run (pipeline state present but regions are missing)")
    img_w, img_h = deps["_image_dimensions"](view_path)
    album_context, page_caption, people_roster = _build_region_detection_context(view_path, root)
    prompt_debug = deps["PromptDebugSession"](view_path, label=view_path.name) if debug else None
    try:
        regions = deps["detect_regions"](
            view_path,
            force=force,
            album_context=album_context,
            page_caption=page_caption,
            people_roster=people_roster,
            prompt_debug=prompt_debug,
            skip_validation=skip_validation,
            write_debug=debug,
        )
    finally:
        debug_path = _write_view_regions_debug_artifact(prompt_debug, image_path=view_path)
        if debug_path is not None:
            print(f"  detect-regions: debug -> {debug_path}")
    if debug:
        _print_render_pipeline_debug_paths(view_path, deps)
    _write_detect_regions_result(
        xmp_path,
        view_path,
        regions,
        img_w=img_w,
        img_h=img_h,
        model_name=model_name,
        debug=debug,
        summary=summary,
        deps=deps,
    )


def _print_render_pipeline_debug_paths(view_path: Path, deps: dict) -> None:
    accepted_debug_path = deps["_accepted_regions_debug_path"](view_path)
    if accepted_debug_path.is_file():
        print(f"  detect-regions: accepted boxes -> {accepted_debug_path}")
    raw_docling_debug_path = deps["_docling_raw_debug_path"](view_path)
    if raw_docling_debug_path.is_file():
        print(f"  detect-regions: docling debug -> {raw_docling_debug_path}")


def _render_pipeline_crop_regions(
    *,
    view_path: Path,
    photos_dir: Path,
    force: bool,
    force_restoration: bool,
    summary: dict[str, int],
    deps: dict,
) -> None:
    crop_stats = deps["CropPageStats"]()
    n = deps["crop_page_regions"](
        view_path,
        photos_dir,
        force=force,
        force_restoration=force_restoration,
        stats=crop_stats,
    )
    summary["crops_written"] += n
    summary["ignored_crop_regions"] += crop_stats.ignored_empty_regions
    if crop_stats.skipped_existing_outputs:
        summary["crop_steps_skipped"] += 1
    if crop_stats.reran_missing_outputs:
        summary["crop_steps_reran"] += 1
    if n > 0:
        print(f"  crop-regions: {n} crop(s)")


def _render_pipeline_face_refresh(
    *,
    view_dir: Path,
    photos_dir: Path,
    current_page: str,
    force: bool,
    summary: dict[str, int],
    face_session,
    deps: dict,
) -> None:
    refresh_targets = _iter_face_refresh_targets(view_dir, photos_dir, current_page)
    face_session.set_files(refresh_targets)
    for img_path in refresh_targets:
        try:
            face_session.refresh_face_regions(img_path, img_path.with_suffix(".xmp"), force=force)
        except deps["FaceRefreshSkipped"] as exc:
            print(f"  face-refresh: {exc}")
        except Exception as exc:
            print(f"  WARNING [face-refresh] {img_path.name}: {exc}", file=sys.stderr)
            summary["warnings"] += 1


def _record_render_pipeline_failure(
    *,
    failures: list[tuple[str, str, str]],
    summary: dict[str, int],
    page_label: str,
    step: str,
    exc: Exception,
    warning: bool = False,
) -> None:
    label = "WARNING" if warning else "ERROR"
    print(f"  {label} [{step}]: {exc}", file=sys.stderr)
    if warning:
        summary["warnings"] += 1
        return
    failures.append((page_label, step, str(exc)))
    summary["errors"] += 1
    summary["pages_failed"] += 1


def _render_pipeline_propagate_metadata(*, xmp_path, view_path, archive_sidecar, failures, summary, page_label, deps):
    try:
        if view_path.is_file() and archive_sidecar.is_file():
            deps["propagate_archive_copy_safe_fields"](xmp_path, archive_sidecar)
    except Exception as exc:
        _record_render_pipeline_failure(
            failures=failures,
            summary=summary,
            page_label=page_label,
            step="propagate-archive-metadata",
            exc=exc,
            warning=True,
        )


def _run_render_pipeline_locked_steps(
    *,
    group,
    primary_scan: Path,
    view_path: Path,
    xmp_path: Path,
    archive_sidecar: Path,
    view_dir: Path,
    photos_dir: Path,
    archive: Path,
    current_page: str,
    current_page_token: str,
    root: Path,
    model_name: str,
    force: bool,
    skip_crops: bool,
    force_restoration: bool,
    debug: bool,
    skip_validation: bool,
    page_label: str,
    summary: dict,
    failures: list,
    face_session,
    deps: dict,
) -> None:
    """Execute all per-page pipeline steps inside an acquired lock."""
    try:
        _render_pipeline_render_page(
            group=group,
            primary_scan=primary_scan,
            view_dir=view_dir,
            photos_dir=photos_dir,
            current_page_token=current_page_token,
            archive=archive,
            deps=deps,
        )
    except Exception as exc:
        _record_render_pipeline_failure(
            failures=failures, summary=summary, page_label=page_label, step="render", exc=exc
        )
        return

    _render_pipeline_propagate_metadata(
        xmp_path=xmp_path,
        view_path=view_path,
        archive_sidecar=archive_sidecar,
        failures=failures,
        summary=summary,
        page_label=page_label,
        deps=deps,
    )

    if _is_title_page_view(view_path):
        summary["detect_regions_skipped"] += 1
        print("  detect-regions: skipped title page (P01)")
        if not skip_crops:
            summary["crop_steps_skipped"] += 1
            print("  crop-regions: skipped title page (P01)")
    else:
        try:
            _render_pipeline_detect_regions(
                view_path=view_path,
                xmp_path=xmp_path,
                root=root,
                model_name=model_name,
                force=force,
                debug=debug,
                skip_validation=skip_validation,
                summary=summary,
                deps=deps,
            )
        except Exception as exc:
            _record_render_pipeline_failure(
                failures=failures, summary=summary, page_label=page_label, step="detect-regions", exc=exc
            )
            return
        if not skip_crops:
            try:
                _render_pipeline_crop_regions(
                    view_path=view_path,
                    photos_dir=photos_dir,
                    force=force,
                    force_restoration=force_restoration,
                    summary=summary,
                    deps=deps,
                )
            except Exception as exc:
                _record_render_pipeline_failure(
                    failures=failures, summary=summary, page_label=page_label, step="crop-regions", exc=exc
                )
                return

    try:
        _render_pipeline_face_refresh(
            view_dir=view_dir,
            photos_dir=photos_dir,
            current_page=current_page,
            force=force,
            summary=summary,
            face_session=face_session,
            deps=deps,
        )
    except Exception as exc:
        _record_render_pipeline_failure(
            failures=failures, summary=summary, page_label=page_label, step="face-refresh", exc=exc
        )
        return
    summary["pages_completed"] += 1


def _run_render_pipeline_page(
    *,
    archive: Path,
    view_dir: Path,
    photos_dir: Path,
    group,
    root: Path,
    model_name: str,
    force: bool,
    skip_crops: bool,
    force_restoration: bool,
    debug: bool,
    skip_validation: bool,
    summary: dict[str, int],
    failures: list[tuple[str, str, str]],
    face_session,
    deps: dict,
) -> None:
    primary_scan = Path(deps["_require_primary_scan"](group))
    view_path = deps["_view_page_output_path"](primary_scan, view_dir)
    xmp_path = view_path.with_suffix(".xmp")
    archive_sidecar = primary_scan.with_suffix(".xmp")
    current_page = str(primary_scan.stem.split("_P")[1].split("_")[0]).zfill(2)
    current_page_token = f"_P{current_page}_"
    page_label = view_path.name
    summary["pages_seen"] += 1
    print(f"Processing {page_label}...")

    try:
        _run_render_pipeline_locked_steps(
            group=group,
            primary_scan=primary_scan,
            view_path=view_path,
            xmp_path=xmp_path,
            archive_sidecar=archive_sidecar,
            view_dir=view_dir,
            photos_dir=photos_dir,
            archive=archive,
            current_page=current_page,
            current_page_token=current_page_token,
            root=root,
            model_name=model_name,
            force=force,
            skip_crops=skip_crops,
            force_restoration=force_restoration,
            debug=debug,
            skip_validation=skip_validation,
            page_label=page_label,
            summary=summary,
            failures=failures,
            face_session=face_session,
            deps=deps,
        )
    except Exception as exc:
        _record_render_pipeline_failure(
            failures=failures, summary=summary, page_label=page_label, step="page", exc=exc
        )


def run_render_pipeline(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
    force: bool,
    skip_crops: bool,
    force_restoration: bool = False,
    debug: bool = False,
    skip_validation: bool = False,
) -> int:
    """Run the full render pipeline for matching pages.

    Pipeline order per page:
      render -> detect-regions -> crop-regions -> face-refresh
    """
    from .lib.ai_model_settings import default_view_region_model
    from .lib.ai_photo_crops import CropPageStats, crop_page_regions
    from .lib.ai_render_face_refresh import FaceRefreshSkipped, RenderFaceRefreshSession
    from .lib.ai_view_regions import (
        _accepted_regions_debug_path,
        _docling_raw_debug_path,
        _failed_regions_debug_path,
        _has_xmp_regions,
        _image_dimensions,
        _read_regions_from_xmp,
        associate_captions,
        detect_regions,
        validate_region_set,
    )
    from .lib.prompt_debug import PromptDebugSession
    from .lib.xmp_sidecar import (
        clear_pipeline_steps,
        propagate_archive_copy_safe_fields,
        read_pipeline_step,
        write_pipeline_step,
        write_region_list,
    )
    from .stitch_oversized_pages import (
        _require_primary_scan,
        _view_page_output_path,
        derived_to_jpg,
        get_photos_dirname,
        get_view_dirname,
        list_archive_dirs,
        list_derived_images,
        list_page_scans,
        stitch,
        tif_to_jpg,
    )

    root = Path(photos_root)
    album_id_lower = album_id.casefold()
    model_name = default_view_region_model()

    # Collect archive directories to process
    archives = [Path(p) for p in list_archive_dirs(root) if not album_id or album_id_lower in Path(p).name.casefold()]
    if not archives:
        print(f"No archive directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str, str]] = []
    summary = {
        "pages_seen": 0,
        "pages_completed": 0,
        "pages_failed": 0,
        "warnings": 0,
        "errors": 0,
        "detect_regions_found": 0,
        "detect_regions_no_regions": 0,
        "detect_regions_skipped": 0,
        "detect_regions_reran": 0,
        "crop_steps_skipped": 0,
        "crop_steps_reran": 0,
        "crops_written": 0,
        "ignored_crop_regions": 0,
    }

    face_session = RenderFaceRefreshSession(photos_root=root)
    deps = {
        "CropPageStats": CropPageStats,
        "FaceRefreshSkipped": FaceRefreshSkipped,
        "PromptDebugSession": PromptDebugSession,
        "_accepted_regions_debug_path": _accepted_regions_debug_path,
        "_docling_raw_debug_path": _docling_raw_debug_path,
        "_failed_regions_debug_path": _failed_regions_debug_path,
        "_has_xmp_regions": _has_xmp_regions,
        "_image_dimensions": _image_dimensions,
        "_read_regions_from_xmp": _read_regions_from_xmp,
        "_require_primary_scan": _require_primary_scan,
        "_view_page_output_path": _view_page_output_path,
        "associate_captions": associate_captions,
        "clear_pipeline_steps": clear_pipeline_steps,
        "crop_page_regions": crop_page_regions,
        "derived_to_jpg": derived_to_jpg,
        "detect_regions": detect_regions,
        "list_derived_images": list_derived_images,
        "propagate_archive_copy_safe_fields": propagate_archive_copy_safe_fields,
        "read_pipeline_step": read_pipeline_step,
        "stitch": stitch,
        "tif_to_jpg": tif_to_jpg,
        "validate_region_set": validate_region_set,
        "write_pipeline_step": write_pipeline_step,
        "write_region_list": write_region_list,
    }

    for archive in archives:
        view_dir = Path(get_view_dirname(archive))
        photos_dir = Path(get_photos_dirname(archive))
        page_groups = list_page_scans(archive)
        if page is not None:
            page_groups = [g for g in page_groups if f"P{str(page).zfill(2)}" in Path(_require_primary_scan(g)).name]

        for group in page_groups:
            _run_render_pipeline_page(
                archive=archive,
                view_dir=view_dir,
                photos_dir=photos_dir,
                group=group,
                root=root,
                model_name=model_name,
                force=force,
                skip_crops=skip_crops,
                force_restoration=force_restoration,
                debug=debug,
                skip_validation=skip_validation,
                summary=summary,
                failures=failures,
                face_session=face_session,
                deps=deps,
            )

    print("\n===== PIPELINE SUMMARY =====")
    print(
        f"Pages: {summary['pages_seen']} total, {summary['pages_completed']} completed, {summary['pages_failed']} failed"
    )
    print(f"Warnings: {summary['warnings']}")
    print(f"Errors: {summary['errors']}")
    print(
        "Detect regions: "
        f"{summary['detect_regions_found']} found, "
        f"{summary['detect_regions_no_regions']} no-regions, "
        f"{summary['detect_regions_skipped']} skipped, "
        f"{summary['detect_regions_reran']} rerun"
    )
    print(
        "Crop regions: "
        f"{summary['crops_written']} crop(s) written, "
        f"{summary['ignored_crop_regions']} ignored-empty, "
        f"{summary['crop_steps_skipped']} skipped, "
        f"{summary['crop_steps_reran']} rerun"
    )
    if failures:
        print("\n===== PIPELINE FAILURE SUMMARY =====", file=sys.stderr)
        for page_name, step, msg in failures:
            print(f"  {page_name} [{step}]: {msg}", file=sys.stderr)
        return 1

    return 0


def print_pipeline_plan(
    steps: list,
    skip_ids: set[str],
    redo_ids: set[str],
    album_label: str,
    page_count: int,
) -> None:
    total = len(steps)
    print(f"Pipeline steps ({total} total):")
    for i, step in enumerate(steps, 1):
        annotation = ""
        if step.id in skip_ids:
            annotation = f"  (skipped: --skip {step.id})"
        elif step.id in redo_ids:
            annotation = "  (redo forced)"
        print(f"  [{i}] {step.id}{annotation}")
    if album_label:
        print(f"Album: {album_label}, {page_count} page(s)")
    else:
        print(f"{page_count} page(s)")


def _dispatch_pipeline_step(
    *,
    step,
    group,
    primary_scan,
    view_path,
    xmp_path,
    archive_sidecar,
    archive,
    view_dir,
    photos_dir,
    current_page,
    current_page_token,
    counters,
    step_just_ran,
    stale_dep,
    ai_runner,
    scan_ai_runner,
    ai_page_idx,
    face_session,
    root,
    model_name,
    force_this_step,
    debug,
    no_validation,
    skip_restoration,
    force_restoration,
    should_redo,
    deps,
) -> int:
    match step.id:
        case "scan-ai":
            _run_pipeline_scan_ai_step(
                primary_scan=primary_scan,
                xmp_path=xmp_path,
                scan_ai_runner=scan_ai_runner,
                force_this_step=force_this_step,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "render":
            if not force_this_step and view_path.is_file():
                counters["render"]["skipped"] += 1
                _print_outcome("skipped (already complete)", stale_dep)
            else:
                outcome = _run_step_render(
                    group=group,
                    primary_scan=primary_scan,
                    view_dir=view_dir,
                    photos_dir=photos_dir,
                    current_page_token=current_page_token,
                    stitch=deps["stitch"],
                    tif_to_jpg=deps["tif_to_jpg"],
                    derived_to_jpg=deps["derived_to_jpg"],
                    list_derived_images=deps["list_derived_images"],
                    archive=archive,
                )
                counters["render"]["run"] += 1
                step_just_ran.add("render")
                deps["write_pipeline_step"](xmp_path, "render")
                _print_outcome(outcome, stale_dep)
        case "detect-regions":
            _run_pipeline_detect_regions_step(
                view_path=view_path,
                xmp_path=xmp_path,
                root=root,
                model_name=model_name,
                force=force_this_step,
                debug=debug,
                no_validation=no_validation,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "crop-regions":
            _run_pipeline_crop_regions_step(
                view_path=view_path,
                xmp_path=xmp_path,
                photos_dir=photos_dir,
                force_this_step=force_this_step,
                skip_restoration=skip_restoration,
                force_restoration=force_restoration,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "face-refresh":
            _run_pipeline_face_refresh_step(
                view_dir=view_dir,
                photos_dir=photos_dir,
                current_page=current_page,
                face_session=face_session,
                should_redo=should_redo,
                xmp_path=xmp_path,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "immich-face-refresh":
            _run_pipeline_immich_face_refresh_step(
                primary_scan=primary_scan,
                view_dir=view_dir,
                photos_dir=photos_dir,
                current_page=current_page,
                xmp_path=xmp_path,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "ai-index":
            ai_page_idx = _run_pipeline_ai_index_step(
                ai_runner=ai_runner,
                ai_page_idx=ai_page_idx,
                view_path=view_path,
                xmp_path=xmp_path,
                force_this_step=force_this_step,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
        case "verify-crops":
            _run_pipeline_verify_crops_step(
                view_path=view_path,
                xmp_path=xmp_path,
                ai_runner=ai_runner,
                force_this_step=force_this_step,
                counters=counters,
                step_just_ran=step_just_ran,
                stale_dep=stale_dep,
                deps=deps,
            )
    return ai_page_idx


def _run_process_pipeline_step(
    *,
    step,
    step_idx: int,
    total_steps: int,
    page_label: str,
    group,
    primary_scan: Path,
    view_path: Path,
    xmp_path: Path,
    archive_sidecar: Path,
    archive: Path,
    view_dir: Path,
    photos_dir: Path,
    current_page: str,
    current_page_token: str,
    effective_redo_ids: set[str],
    step_just_ran: set[str],
    counters: dict[str, dict],
    ai_runner,
    scan_ai_runner,
    ai_page_idx: int,
    face_session,
    root: Path,
    model_name: str,
    force: bool,
    debug: bool,
    no_validation: bool,
    skip_restoration: bool,
    force_restoration: bool,
    deps: dict,
) -> int:
    prefix = f"[{step_idx}/{total_steps}] {step.id} {page_label}"
    pipeline_state = deps["read_pipeline_state"](xmp_path)
    should_redo = step.id in effective_redo_ids
    dep_ran = any(dep in step_just_ran for dep in step.depends_on)
    stale = False
    stale_dep = ""
    if not should_redo and not dep_ran:
        stale, stale_dep = _check_step_stale(step, pipeline_state)
    force_this_step = should_redo or dep_ran or stale

    print(f"{prefix}", end="", flush=True)

    try:
        ai_page_idx = _dispatch_pipeline_step(
            step=step,
            group=group,
            primary_scan=primary_scan,
            view_path=view_path,
            xmp_path=xmp_path,
            archive_sidecar=archive_sidecar,
            archive=archive,
            view_dir=view_dir,
            photos_dir=photos_dir,
            current_page=current_page,
            current_page_token=current_page_token,
            counters=counters,
            step_just_ran=step_just_ran,
            stale_dep=stale_dep,
            ai_runner=ai_runner,
            scan_ai_runner=scan_ai_runner,
            ai_page_idx=ai_page_idx,
            face_session=face_session,
            root=root,
            model_name=model_name,
            force_this_step=force_this_step,
            debug=debug,
            no_validation=no_validation,
            skip_restoration=skip_restoration,
            force_restoration=force_restoration,
            should_redo=should_redo,
            deps=deps,
        )
    except Exception as exc:
        counters[step.id]["failed"] += 1
        print(f" ... ERROR: {exc}", file=sys.stderr, flush=True)
    return ai_page_idx


def _run_pipeline_scan_ai_step(
    *,
    primary_scan: Path,
    xmp_path: Path,
    scan_ai_runner,
    force_this_step: bool,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> None:
    if not force_this_step and deps["read_pipeline_step"](xmp_path, "scan-ai") is not None:
        counters["scan-ai"]["skipped"] += 1
        _print_outcome("skipped (already complete)", stale_dep)
        return
    scan_ai_runner.force_processing = force_this_step
    scan_ai_runner.files = [primary_scan]
    fail_before = scan_ai_runner.failures
    scan_ai_runner._process_one(1, primary_scan)
    if scan_ai_runner.failures > fail_before:
        counters["scan-ai"]["failed"] += 1
        _print_outcome("ERROR", stale_dep)
    else:
        counters["scan-ai"]["run"] += 1
        step_just_ran.add("scan-ai")
        deps["write_pipeline_step"](xmp_path, "scan-ai")
        _print_outcome("done", stale_dep)


def _run_pipeline_detect_regions_step(
    *,
    view_path,
    xmp_path,
    root,
    model_name,
    force,
    debug,
    no_validation,
    counters,
    step_just_ran,
    stale_dep,
    deps,
) -> None:
    skipped, _ran = _run_step_detect_regions(
        view_path=view_path,
        xmp_path=xmp_path,
        root=root,
        model_name=model_name,
        force=force,
        debug=debug,
        skip_validation=no_validation,
        counters=counters,
        prompt_debug_cls=deps["PromptDebugSession"] if debug else None,
        write_region_list=deps["write_region_list"],
        read_pipeline_step=deps["read_pipeline_step"],
        write_pipeline_step=deps["write_pipeline_step"],
        clear_pipeline_steps=deps["clear_pipeline_steps"],
        detect_regions=deps["detect_regions"],
        associate_captions=deps["associate_captions"],
        validate_region_set=deps["validate_region_set"],
        _has_xmp_regions=deps["_has_xmp_regions"],
        _image_dimensions=deps["_image_dimensions"],
        _read_regions_from_xmp=deps["_read_regions_from_xmp"],
    )
    if skipped:
        counters["detect-regions"]["skipped"] += 1
        _print_outcome("skipped (already complete)", stale_dep)
    else:
        counters["detect-regions"]["run"] += 1
        step_just_ran.add("detect-regions")
        deps["write_pipeline_step"](xmp_path, "detect-regions")
        _print_outcome("done", stale_dep)


def _run_pipeline_crop_regions_step(
    *,
    view_path: Path,
    xmp_path: Path,
    photos_dir: Path,
    force_this_step: bool,
    skip_restoration: bool,
    force_restoration: bool,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> None:
    if _is_title_page_view(view_path):
        counters["crop-regions"]["skipped"] += 1
        _print_outcome("skipped (title page)", stale_dep)
        return
    if force_this_step:
        deps["clear_pipeline_steps"](xmp_path, ["crop-regions"])
    crop_stats = deps["CropPageStats"]()
    n = deps["crop_page_regions"](
        view_path,
        photos_dir,
        force=force_this_step,
        skip_restoration=skip_restoration,
        force_restoration=force_restoration,
        stats=crop_stats,
    )
    if crop_stats.skipped_existing_outputs and n == 0:
        counters["crop-regions"]["skipped"] += 1
        _print_outcome("skipped (already complete)", stale_dep)
        return
    counters["crop-regions"]["run"] += 1
    counters["crop-regions"]["detail"].append(f"{n} crops written")
    step_just_ran.add("crop-regions")
    deps["write_pipeline_step"](xmp_path, "crop-regions")
    _print_outcome("done", stale_dep)


def _run_pipeline_face_refresh_step(
    *,
    view_dir: Path,
    photos_dir: Path,
    current_page: str,
    face_session,
    should_redo: bool,
    xmp_path: Path,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> None:
    refresh_targets = _iter_face_refresh_targets(view_dir, photos_dir, current_page)
    face_session.set_files(refresh_targets)
    ran_any = False
    for img_path in refresh_targets:
        try:
            ran_any = (
                face_session.refresh_face_regions(img_path, img_path.with_suffix(".xmp"), force=should_redo) or ran_any
            )
        except deps["FaceRefreshSkipped"] as exc:
            log.debug("Face refresh skipped for %s: %s", img_path, exc)
    if ran_any:
        counters["face-refresh"]["run"] += 1
        step_just_ran.add("face-refresh")
        deps["write_pipeline_step"](xmp_path, "face-refresh")
        _print_outcome("done", stale_dep)
        return
    counters["face-refresh"]["skipped"] += 1
    _print_outcome("skipped (already complete)", "")


def _run_pipeline_immich_face_refresh_step(
    *,
    primary_scan: Path,
    view_dir: Path,
    photos_dir: Path,
    current_page: str,
    xmp_path: Path,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> None:
    base_url = str(os.environ.get(deps["IMMICH_URL_ENV"]) or "").rstrip("/")
    api_key = str(os.environ.get(deps["IMMICH_API_KEY_ENV"]) or "")
    if not base_url or not api_key:
        raise RuntimeError("IMMICH_URL and IMMICH_API_KEY are required for immich-face-refresh")

    refresh_targets = [primary_scan, *_iter_face_refresh_targets(view_dir, photos_dir, current_page)]
    updated = 0
    unmatched = 0
    for img_path in refresh_targets:
        assets = deps["fetch_assets_by_original_filename"](base_url, api_key, img_path.name)
        if not assets:
            unmatched += 1
            continue
        if len(assets) > 1:
            raise RuntimeError(f"Immich returned multiple assets for {img_path.name}")
        asset_id = str(assets[0].get("id") or "").strip()
        if not asset_id:
            raise RuntimeError(f"Immich asset for {img_path.name} did not include an id")
        faces = deps["fetch_asset_faces"](base_url, api_key, asset_id)
        named_faces = [
            face
            for face in faces
            if isinstance(face.get("person"), dict) and str(face["person"].get("name") or "").strip()
        ]
        names = deps["_dedupe_names"]([str(face["person"]["name"]).strip() for face in named_faces])
        regions = deps["_faces_to_regions"](named_faces)
        local_width, local_height = deps["_image_dimensions"](img_path)
        if local_width > 0 and local_height > 0:
            for region in regions:
                region["image_width"] = local_width
                region["image_height"] = local_height
        sidecar_path = img_path.with_suffix(".xmp")
        deps["merge_persons_xmp"](sidecar_path, names)
        deps["merge_face_regions_xmp"](sidecar_path, regions)
        updated += 1

    counters["immich-face-refresh"]["run"] += 1
    counters["immich-face-refresh"]["detail"].append(f"{updated} updated, {unmatched} unmatched")
    step_just_ran.add("immich-face-refresh")
    deps["write_pipeline_step"](xmp_path, "immich-face-refresh")
    _print_outcome("done", stale_dep)


def _run_pipeline_ai_index_step(
    *,
    ai_runner,
    ai_page_idx: int,
    view_path: Path,
    xmp_path: Path,
    force_this_step: bool,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> int:
    if force_this_step:
        ai_runner.force_processing = True
    ai_page_idx += 1
    skip_before = ai_runner.skipped
    fail_before = ai_runner.failures
    ai_runner._process_one(ai_page_idx, view_path)
    if ai_runner.failures > fail_before:
        counters["ai-index"]["failed"] += 1
        _print_outcome("ERROR", stale_dep)
    elif ai_runner.skipped > skip_before:
        counters["ai-index"]["skipped"] += 1
        _print_outcome("skipped (already complete)", stale_dep)
    else:
        counters["ai-index"]["run"] += 1
        step_just_ran.add("ai-index")
        deps["write_pipeline_step"](xmp_path, "ai-index")
        _print_outcome("done", stale_dep)
        _print_ai_index_discovery_summary(sidecar_path=xmp_path)
    return ai_page_idx


def _run_pipeline_verify_crops_step(
    *,
    view_path: Path,
    xmp_path: Path,
    ai_runner,
    force_this_step: bool,
    counters: dict[str, dict],
    step_just_ran: set[str],
    stale_dep: str,
    deps: dict,
) -> None:
    if not force_this_step and deps["read_pipeline_step"](xmp_path, "verify-crops") is not None:
        counters["verify-crops"]["skipped"] += 1
        _print_outcome("skipped (already complete)", stale_dep)
        return
    verify_result = deps["run_verify_crops_page"](
        view_path,
        model_name=str(ai_runner.defaults.get("caption_model") or ""),
        base_url=str(ai_runner.defaults.get("lmstudio_base_url") or ""),
        logger=lambda message: print(f"    {message}", flush=True),
    )
    deps["persist_verify_crops_state"](view_path, verify_result)
    counters["verify-crops"]["run"] += 1
    step_just_ran.add("verify-crops")
    counters["verify-crops"]["detail"].append(_format_verify_crops_detail(verify_result))
    _print_outcome("done", stale_dep)
    _print_verify_crops_summary(view_path, verify_result)


def run_process_pipeline(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
    skip_ids: list[str],
    redo_ids: list[str],
    step_id: str | None,
    force: bool,
    debug: bool,
    no_validation: bool,
    skip_restoration: bool,
    force_restoration: bool,
    log_lmstudio_tokens: bool = False,
    gps_only: bool = False,
    refresh_gps: bool = False,
) -> int:
    from .lib._caption_lmstudio import lmstudio_token_logging
    from .lib.pipeline import OPTIONAL_STEP_IDS, PIPELINE_STEPS, VALID_STEP_IDS
    from .lib.xmp_sidecar import (
        clear_pipeline_steps,
        read_pipeline_state,
        read_pipeline_step,
        write_pipeline_step,
        write_region_list,
    )
    from .stitch_oversized_pages import (
        _require_primary_scan,
        _view_page_output_path,
        derived_to_jpg,
        get_photos_dirname,
        get_view_dirname,
        list_archive_dirs,
        list_derived_images,
        list_page_scans,
        stitch,
        tif_to_jpg,
    )

    root = Path(photos_root)
    effective_skip_ids, effective_redo_ids = _effective_pipeline_step_ids(
        valid_step_ids=VALID_STEP_IDS,
        optional_step_ids=OPTIONAL_STEP_IDS,
        skip_ids=skip_ids,
        redo_ids=redo_ids,
        step_id=step_id,
        force=force,
        refresh_gps=refresh_gps,
    )

    # Collect archive directories
    archives = _matching_pipeline_archives(root, album_id=album_id, list_archive_dirs=list_archive_dirs)
    if not archives:
        print(f"No archive directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    # Collect all page groups
    all_pages = _collect_pipeline_pages(
        archives,
        page=page,
        _require_primary_scan=_require_primary_scan,
        get_photos_dirname=get_photos_dirname,
        get_view_dirname=get_view_dirname,
        list_page_scans=list_page_scans,
    )
    if not all_pages:
        print("No matching pages found.", file=sys.stderr)
        return 1

    active_steps = [s for s in PIPELINE_STEPS if s.id not in effective_skip_ids]
    total_steps = len(active_steps)
    album_label = album_id or ""
    print_pipeline_plan(PIPELINE_STEPS, effective_skip_ids, effective_redo_ids, album_label, len(all_pages))

    # Per-step counters: {step_id: {"run": int, "skipped": int, "failed": int, "detail": list}}
    counters: dict[str, dict] = {s.id: {"run": 0, "skipped": 0, "failed": 0, "detail": []} for s in PIPELINE_STEPS}

    # Track which steps have run this pass per page (for staleness cascade)
    from cast.immich_sync import (
        IMMICH_API_KEY_ENV,
        IMMICH_URL_ENV,
        _dedupe_names,
        _faces_to_regions,
        fetch_asset_faces,
        fetch_assets_by_original_filename,
    )
    from cast.xmp_writer import merge_face_regions_xmp, merge_persons_xmp

    from .lib.ai_model_settings import default_view_region_model
    from .lib.ai_photo_crops import CropPageStats, crop_page_regions
    from .lib.ai_render_face_refresh import FaceRefreshSkipped, RenderFaceRefreshSession
    from .lib.ai_verify_crops import persist_verify_crops_state, run_verify_crops_page
    from .lib.ai_view_regions import (
        _has_xmp_regions,
        _image_dimensions,
        _read_regions_from_xmp,
        associate_captions,
        detect_regions,
        validate_region_set,
    )
    from .lib.prompt_debug import PromptDebugSession

    model_name = default_view_region_model()
    face_session = RenderFaceRefreshSession(photos_root=root)

    # Build IndexRunners once so engine caches are shared across pages
    from .lib.ai_index_runner import IndexRunner

    ai_runner = IndexRunner(
        _pipeline_ai_runner_argv(root, gps_only=gps_only, refresh_gps=refresh_gps, force=force, debug=debug)
    )
    scan_ai_runner = IndexRunner(
        _pipeline_scan_ai_runner_argv(root, force=force, debug=debug)
    )
    deps = {
        "CropPageStats": CropPageStats,
        "FaceRefreshSkipped": FaceRefreshSkipped,
        "IMMICH_API_KEY_ENV": IMMICH_API_KEY_ENV,
        "IMMICH_URL_ENV": IMMICH_URL_ENV,
        "PromptDebugSession": PromptDebugSession,
        "_dedupe_names": _dedupe_names,
        "_faces_to_regions": _faces_to_regions,
        "_has_xmp_regions": _has_xmp_regions,
        "_image_dimensions": _image_dimensions,
        "_read_regions_from_xmp": _read_regions_from_xmp,
        "associate_captions": associate_captions,
        "clear_pipeline_steps": clear_pipeline_steps,
        "crop_page_regions": crop_page_regions,
        "derived_to_jpg": derived_to_jpg,
        "detect_regions": detect_regions,
        "fetch_asset_faces": fetch_asset_faces,
        "fetch_assets_by_original_filename": fetch_assets_by_original_filename,
        "list_derived_images": list_derived_images,
        "merge_face_regions_xmp": merge_face_regions_xmp,
        "merge_persons_xmp": merge_persons_xmp,
        "persist_verify_crops_state": persist_verify_crops_state,
        "read_pipeline_state": read_pipeline_state,
        "read_pipeline_step": read_pipeline_step,
        "run_verify_crops_page": run_verify_crops_page,
        "stitch": stitch,
        "tif_to_jpg": tif_to_jpg,
        "validate_region_set": validate_region_set,
        "write_pipeline_step": write_pipeline_step,
        "write_region_list": write_region_list,
    }

    with lmstudio_token_logging(log_lmstudio_tokens):
        _process_pipeline_pages(
            all_pages,
            active_steps=active_steps,
            total_steps=total_steps,
            effective_redo_ids=effective_redo_ids,
            counters=counters,
            ai_runner=ai_runner,
            scan_ai_runner=scan_ai_runner,
            face_session=face_session,
            root=root,
            model_name=model_name,
            force=force,
            debug=debug,
            no_validation=no_validation,
            skip_restoration=skip_restoration,
            force_restoration=force_restoration,
            _require_primary_scan=_require_primary_scan,
            _view_page_output_path=_view_page_output_path,
            deps=deps,
        )

    _print_process_pipeline_summary(active_steps, counters)

    any_failed = any(counters[s.id]["failed"] > 0 for s in active_steps)
    return 1 if any_failed else 0


def _effective_pipeline_step_ids(
    *,
    valid_step_ids: set[str],
    optional_step_ids: set[str] = frozenset(),
    skip_ids: list[str] = (),
    redo_ids: list[str],
    step_id: str | None,
    force: bool,
    refresh_gps: bool,
) -> tuple[set[str], set[str]]:
    effective_redo_ids = set(valid_step_ids) if force else set(redo_ids)
    if step_id:
        effective_skip_ids = set(sid for sid in valid_step_ids if sid != step_id)
    else:
        auto_skipped = optional_step_ids - effective_redo_ids
        effective_skip_ids = set(skip_ids) | auto_skipped
    if step_id != "face-refresh" and "face-refresh" not in redo_ids:
        effective_skip_ids.add("face-refresh")
    if step_id != "immich-face-refresh" and "immich-face-refresh" not in redo_ids:
        effective_skip_ids.add("immich-face-refresh")
    if step_id != "verify-crops" and "verify-crops" not in redo_ids:
        effective_skip_ids.add("verify-crops")
    if refresh_gps:
        effective_redo_ids.add("ai-index")
    return effective_skip_ids, effective_redo_ids


def _matching_pipeline_archives(root: Path, *, album_id: str, list_archive_dirs) -> list[Path]:
    album_id_lower = album_id.casefold()
    return [Path(p) for p in list_archive_dirs(root) if not album_id or album_id_lower in Path(p).name.casefold()]


def _collect_pipeline_pages(
    archives: list[Path],
    *,
    page: str | None,
    _require_primary_scan,
    get_photos_dirname,
    get_view_dirname,
    list_page_scans,
) -> list[tuple]:
    all_pages: list[tuple] = []
    for archive in archives:
        view_dir = Path(get_view_dirname(archive))
        photos_dir = Path(get_photos_dirname(archive))
        for group in _pipeline_page_groups(
            archive, page=page, _require_primary_scan=_require_primary_scan, list_page_scans=list_page_scans
        ):
            all_pages.append((archive, view_dir, photos_dir, group))
    return all_pages


def _pipeline_page_groups(archive: Path, *, page: str | None, _require_primary_scan, list_page_scans) -> list:
    page_groups = list_page_scans(archive)
    if page is None:
        return page_groups
    page_token = f"P{str(page).zfill(2)}"
    return [group for group in page_groups if page_token in Path(_require_primary_scan(group)).name]


def _pipeline_ai_runner_argv(root: Path, *, gps_only: bool, refresh_gps: bool, force: bool, debug: bool) -> list[str]:
    argv = ["--photos-root", str(root)]
    if gps_only or refresh_gps:
        argv += ["--reprocess-mode", "gps"]
    if force:
        argv.append("--force")
    if debug:
        argv.append("--debug")
    return argv


def _pipeline_scan_ai_runner_argv(root: Path, *, force: bool, debug: bool) -> list[str]:
    argv = ["--photos-root", str(root), "--include-archive"]
    if force:
        argv.append("--force")
    if debug:
        argv.append("--debug")
    return argv


def _process_pipeline_pages(
    all_pages: list[tuple],
    *,
    active_steps: list,
    total_steps: int,
    effective_redo_ids: set[str],
    counters: dict[str, dict],
    ai_runner,
    scan_ai_runner,
    face_session,
    root: Path,
    model_name: str,
    force: bool,
    debug: bool,
    no_validation: bool,
    skip_restoration: bool,
    force_restoration: bool,
    _require_primary_scan,
    _view_page_output_path,
    deps: dict,
) -> None:
    ai_page_idx = 0
    for archive, view_dir, photos_dir, group in all_pages:
        page_context = _pipeline_page_context(
            group,
            view_dir=view_dir,
            _require_primary_scan=_require_primary_scan,
            _view_page_output_path=_view_page_output_path,
        )
        step_just_ran: set[str] = set()
        for step_idx, step in enumerate(active_steps, 1):
            ai_page_idx = _run_process_pipeline_step(
                step=step,
                step_idx=step_idx,
                total_steps=total_steps,
                archive=archive,
                view_dir=view_dir,
                photos_dir=photos_dir,
                group=group,
                effective_redo_ids=effective_redo_ids,
                step_just_ran=step_just_ran,
                counters=counters,
                ai_runner=ai_runner,
                scan_ai_runner=scan_ai_runner,
                ai_page_idx=ai_page_idx,
                face_session=face_session,
                root=root,
                model_name=model_name,
                force=force,
                debug=debug,
                no_validation=no_validation,
                skip_restoration=skip_restoration,
                force_restoration=force_restoration,
                deps=deps,
                **page_context,
            )


def _pipeline_page_context(group, *, view_dir: Path, _require_primary_scan, _view_page_output_path) -> dict:
    primary_scan = Path(_require_primary_scan(group))
    view_path = _view_page_output_path(primary_scan, view_dir)
    current_page = str(primary_scan.stem.split("_P")[1].split("_")[0]).zfill(2)
    return {
        "primary_scan": primary_scan,
        "view_path": view_path,
        "xmp_path": view_path.with_suffix(".xmp"),
        "archive_sidecar": primary_scan.with_suffix(".xmp"),
        "current_page": current_page,
        "current_page_token": f"_P{current_page}_",
        "page_label": view_path.name,
    }


def _print_process_pipeline_summary(active_steps: list, counters: dict[str, dict]) -> None:
    print("\n===== PIPELINE SUMMARY =====")
    for step in active_steps:
        c = counters[step.id]
        detail = ", ".join(c["detail"]) if c["detail"] else ""
        detail_col = f"  ({detail})" if detail else ""
        print(f"  {step.id:<22} run={c['run']}  skipped={c['skipped']}  failed={c['failed']}{detail_col}")


def _check_step_stale(step, pipeline_state: dict) -> tuple[bool, str]:
    from .lib.xmp_sidecar import is_step_stale

    stale = is_step_stale(step.id, step.depends_on, pipeline_state)
    if stale:
        # Identify which dep triggered it
        from datetime import datetime

        def _parse_ts(entry: dict) -> str:
            # Support both "timestamp" (new schema) and "completed" (legacy schema)
            return str(entry.get("timestamp") or entry.get("completed") or "").strip()

        step_entry = pipeline_state.get(step.id) or {}
        step_ts_str = _parse_ts(step_entry)
        if not step_ts_str:
            return True, ""
        try:
            step_ts = datetime.fromisoformat(step_ts_str)
        except ValueError:
            return True, ""
        for dep_id in step.depends_on:
            dep_entry = pipeline_state.get(dep_id) or {}
            dep_ts_str = _parse_ts(dep_entry)
            if not dep_ts_str:
                continue
            try:
                dep_ts = datetime.fromisoformat(dep_ts_str)
            except ValueError as exc:
                log.debug("Invalid ISO timestamp in dependency %r: %s", dep_ts_str, exc)
                continue
            if dep_ts > step_ts:
                return True, dep_id
    return False, ""


def _print_outcome(outcome: str, stale_dep: str) -> None:
    if stale_dep:
        print(f" ... (re-run: {stale_dep} updated)", flush=True)
    else:
        print(f" ... {outcome}", flush=True)


def _print_ai_index_discovery_summary(*, sidecar_path: Path) -> None:
    from .lib.xmp_sidecar import read_ai_sidecar_state

    state = read_ai_sidecar_state(sidecar_path)
    if not isinstance(state, dict):
        return
    detections = dict(state.get("detections") or {})
    caption = str(state.get("description") or "").strip()
    if caption:
        print(f"    ai-index caption: {caption}", flush=True)
    dc_date = str(state.get("dc_date") or "").strip()
    if dc_date:
        print(f"    ai-index date: {dc_date}", flush=True)
    location_text = _ai_index_location_text(state)
    if location_text:
        print(f"    ai-index shown_location: {location_text}", flush=True)
    _print_ai_index_gps_summary(state)
    locations_shown = list(detections.get("locations_shown") or [])
    if locations_shown:
        names = _location_shown_names(locations_shown)
        if names:
            print(f"    ai-index locations_shown: {', '.join(names)}", flush=True)


def _location_shown_names(locations_shown: list) -> list[str]:
    names = []
    for row in locations_shown:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("city") or row.get("country_name") or "").strip()
        if name:
            names.append(name)
    return names


def _ai_index_location_text(state: dict) -> str:
    location_parts = [
        str(state.get("location_sublocation") or "").strip(),
        str(state.get("location_city") or "").strip(),
        str(state.get("location_state") or "").strip(),
        str(state.get("location_country") or "").strip(),
    ]
    return ", ".join(part for part in location_parts if part)


def _print_ai_index_gps_summary(state: dict) -> None:
    gps_lat = str(state.get("gps_latitude") or "").strip()
    gps_lon = str(state.get("gps_longitude") or "").strip()
    if gps_lat or gps_lon:
        print(f"    ai-index gps: {gps_lat or '?'} {gps_lon or '?'}", flush=True)


def _format_verify_crops_detail(verify_result: dict[str, object]) -> str:
    status = str(verify_result.get("status") or "")
    if status != "ok":
        missing = ", ".join(list(verify_result.get("missing_context") or []))
        return f"missing_context={missing or 'unknown'}"
    result_count = len(list(verify_result.get("results") or []))
    flagged = 0
    for row in list(verify_result.get("results") or []):
        review = dict(row.get("review") or {})
        if any(
            str((review.get(concern) or {}).get("verdict") or "") in {"bad", "uncertain"}
            for concern in ("caption", "gps", "shown_location", "date", "overall")
        ):
            flagged += 1
    return f"{result_count} reviewed, {flagged} flagged"


def _print_verify_crops_summary(view_path: Path, verify_result: dict[str, object]) -> None:
    status = str(verify_result.get("status") or "")
    if status != "ok":
        missing = ", ".join(list(verify_result.get("missing_context") or []))
        print(f"    verify-crops missing context: {missing}", flush=True)
        return
    for row in list(verify_result.get("results") or []):
        _print_verify_crops_row(row)
    artifact_path = str(verify_result.get("artifact_path") or "").strip()
    if artifact_path:
        print(f"    verify-crops artifact: {artifact_path}", flush=True)


def _print_verify_crops_row(row: dict) -> None:
    review = dict(row.get("review") or {})
    concerns = _collect_verify_crops_concerns(review)
    if concerns:
        crop_name = Path(str(row.get("crop_image_path") or "")).name
        print(f"    verify-crops {crop_name}: {'; '.join(concerns)}", flush=True)


def _collect_verify_crops_concerns(review: dict) -> list[str]:
    concerns: list[str] = []
    for concern in ("caption", "gps", "shown_location", "date", "overall"):
        verdict = str((review.get(concern) or {}).get("verdict") or "")
        if verdict in {"bad", "uncertain"}:
            reason = str((review.get(concern) or {}).get("failure_reason") or "").strip()
            concern_text = f"{concern}={verdict}"
            if reason:
                concern_text += f" ({reason})"
            concerns.append(concern_text)
    return concerns


def _run_step_render(
    *,
    group,
    primary_scan: Path,
    view_dir: Path,
    photos_dir: Path,
    current_page_token: str,
    stitch,
    tif_to_jpg,
    derived_to_jpg,
    list_derived_images,
    archive: Path,
) -> str:
    if len(group) > 1:
        stitch(group, str(view_dir))
    else:
        tif_to_jpg(str(primary_scan), str(view_dir))
    for derived in list_derived_images(archive):
        if current_page_token in Path(derived).name:
            derived_to_jpg(derived, str(photos_dir))
    return "done"


def _run_step_detect_regions(
    *,
    view_path: Path,
    xmp_path: Path,
    root: Path,
    model_name: str,
    force: bool,
    debug: bool,
    skip_validation: bool,
    counters: dict,
    prompt_debug_cls,
    write_region_list,
    read_pipeline_step,
    write_pipeline_step,
    clear_pipeline_steps,
    detect_regions,
    associate_captions,
    validate_region_set,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
) -> tuple[bool, bool]:
    """Returns (skipped, ran)."""
    if _is_title_page_view(view_path):
        return True, False

    if force:
        clear_pipeline_steps(xmp_path, ["view_regions"])
    elif _view_regions_step_complete(xmp_path):
        if not _has_xmp_regions(xmp_path):
            return True, False
        img_w, img_h = _image_dimensions(view_path)
        stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
        vresult = validate_region_set(stored, img_w=img_w, img_h=img_h)
        if vresult.valid or skip_validation:
            return True, False
        clear_pipeline_steps(xmp_path, ["view_regions"])

    img_w, img_h = _image_dimensions(view_path)
    album_context, page_caption, people_roster = _build_region_detection_context(view_path, root)
    prompt_debug = prompt_debug_cls(view_path, label=view_path.name) if prompt_debug_cls else None
    try:
        regions = detect_regions(
            view_path,
            force=force,
            album_context=album_context,
            page_caption=page_caption,
            people_roster=people_roster,
            prompt_debug=prompt_debug,
            skip_validation=skip_validation,
            write_debug=debug,
        )
    finally:
        _write_view_regions_debug_artifact(prompt_debug, image_path=view_path)

    if regions:
        regions_with_captions = associate_captions(regions, [], img_w)
        write_region_list(xmp_path, regions_with_captions, img_w, img_h)
        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
        counters["detect-regions"]["detail"].append(f"{len(regions)} regions")
    else:
        write_region_list(xmp_path, [], img_w, img_h)
        existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
        if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
            write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})

    return False, True


def run_migrate_page_dir_refs(*, photos_root: str, verify_only: bool = False) -> int:
    from .lib.page_reference_migration import find_sidecars_with_view_references, migrate_album_page_references

    root = Path(photos_root)
    if verify_only:
        matches = find_sidecars_with_view_references(root)
        if matches:
            print(f"Found {len(matches)} sidecar(s) still containing _View under {root}", file=sys.stderr)
            for path in matches[:20]:
                print(f"  {path}", file=sys.stderr)
            if len(matches) > 20:
                print(f"  ... {len(matches) - 20} more", file=sys.stderr)
            return 1
        print(f"No .xmp files under {root} contain _View")
        return 0

    result = migrate_album_page_references(root)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_migrate_caption_layout(*, photos_root: str, verify_only: bool = False) -> int:
    from .lib.caption_layout_migration import (
        find_sidecars_with_legacy_caption_layout,
        migrate_album_caption_layout,
    )

    root = Path(photos_root)
    if verify_only:
        matches = find_sidecars_with_legacy_caption_layout(root)
        if matches:
            print(
                f"Found {len(matches)} sidecar(s) still using the legacy caption layout under {root}", file=sys.stderr
            )
            for path in matches[:20]:
                print(f"  {path}", file=sys.stderr)
            if len(matches) > 20:
                print(f"  ... {len(matches) - 20} more", file=sys.stderr)
            return 1
        print(f"No .xmp files under {root} use the legacy caption layout")
        return 0

    result = migrate_album_caption_layout(root)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_migrate_pipeline_records(*, photos_root: str, verify_only: bool = False) -> int:
    from .lib.xmp_sidecar import find_sidecars_with_legacy_pipeline_records, migrate_tree_pipeline_records

    root = Path(photos_root)
    if verify_only:
        matches = find_sidecars_with_legacy_pipeline_records(root)
        if matches:
            print(f"Found {len(matches)} sidecar(s) with legacy pipeline records under {root}", file=sys.stderr)
            for path in matches[:20]:
                print(f"  {path}", file=sys.stderr)
            if len(matches) > 20:
                print(f"  ... {len(matches) - 20} more", file=sys.stderr)
            return 1
        print(f"No .xmp files under {root} contain legacy pipeline records")
        return 0

    result = migrate_tree_pipeline_records(root)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_repair_crop_source(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
    verify_only: bool = False,
) -> int:
    from .lib.crop_source_repair import (
        find_crop_sidecars_needing_source_repair,
        repair_album_crop_sources,
    )

    root = Path(photos_root)
    if verify_only:
        matches = find_crop_sidecars_needing_source_repair(root, album_id=album_id, page=page)
        if matches:
            print(
                f"Found {len(matches)} crop sidecar(s) needing dc:source/AlbumTitle repair under {root}",
                file=sys.stderr,
            )
            for path in matches[:20]:
                print(f"  {path}", file=sys.stderr)
            if len(matches) > 20:
                print(f"  ... {len(matches) - 20} more", file=sys.stderr)
            return 1
        print(f"No crop sidecars under {root} need dc:source/AlbumTitle repair")
        return 0

    result = repair_album_crop_sources(root, album_id=album_id, page=page)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                "album_id": album_id,
                "page": str(page or ""),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_repair_crop_numbers(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
) -> int:
    from .lib.crop_number_repair import repair_album_crop_numbers

    root = Path(photos_root)
    result = repair_album_crop_numbers(root, album_id=album_id, page=page)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                "album_id": album_id,
                "page": str(page or ""),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_repair_page_derived_views(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
) -> int:
    from .lib.page_derived_repair import repair_page_derived_views

    root = Path(photos_root)
    result = repair_page_derived_views(root, album_id=album_id, page=page)
    print(
        json.dumps(
            {
                "photos_root": str(root),
                "album_id": album_id,
                "page": str(page or ""),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_detect_view_regions(
    *,
    album_id: str,
    photos_root: str,
    page: str | None,
    force: bool,
    redo_no_regions: bool = False,
    debug: bool = False,
    skip_validation: bool = False,
) -> int:
    from .lib.ai_model_settings import default_view_region_model
    from .lib.ai_view_regions import (
        _accepted_regions_debug_path,
        _docling_raw_debug_path,
        _failed_regions_debug_path,
        _has_xmp_regions,
        _image_dimensions,
        _read_regions_from_xmp,
        associate_captions,
        detect_regions,
        validate_region_set,
    )
    from .lib.prompt_debug import PromptDebugSession
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step, write_region_list
    from .naming import is_pages_dir

    root = Path(photos_root)
    album_id_lower = album_id.casefold()
    model_name = default_view_region_model()

    view_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and is_pages_dir(d) and album_id_lower in d.name.casefold()
    )
    if not view_dirs:
        print(f"No _Pages directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    errors = 0
    for view_dir in view_dirs:
        for view_path in _iter_page_view_targets(view_dir, page):
            ran = _detect_view_regions_for_path(
                view_path,
                root=root,
                force=force,
                redo_no_regions=redo_no_regions,
                debug=debug,
                skip_validation=skip_validation,
                model_name=model_name,
                _has_xmp_regions=_has_xmp_regions,
                _image_dimensions=_image_dimensions,
                _read_regions_from_xmp=_read_regions_from_xmp,
                _accepted_regions_debug_path=_accepted_regions_debug_path,
                _docling_raw_debug_path=_docling_raw_debug_path,
                _failed_regions_debug_path=_failed_regions_debug_path,
                associate_captions=associate_captions,
                clear_pipeline_steps=clear_pipeline_steps,
                detect_regions=detect_regions,
                read_pipeline_step=read_pipeline_step,
                validate_region_set=validate_region_set,
                write_pipeline_step=write_pipeline_step,
                write_region_list=write_region_list,
                PromptDebugSession=PromptDebugSession,
            )
            if ran is False:
                errors += 1

    return 1 if errors else 0


def _detect_view_regions_for_path(
    view_path: Path,
    *,
    root: Path,
    force: bool,
    redo_no_regions: bool,
    debug: bool,
    skip_validation: bool,
    model_name: str,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
    _accepted_regions_debug_path,
    _docling_raw_debug_path,
    _failed_regions_debug_path,
    associate_captions,
    clear_pipeline_steps,
    detect_regions,
    read_pipeline_step,
    validate_region_set,
    write_pipeline_step,
    write_region_list,
    PromptDebugSession,
) -> bool | None:
    xmp_path = view_path.with_suffix(".xmp")
    try:
        decision = _detect_view_regions_decision(
            view_path,
            force=force,
            redo_no_regions=redo_no_regions,
            skip_validation=skip_validation,
            _has_xmp_regions=_has_xmp_regions,
            _image_dimensions=_image_dimensions,
            _read_regions_from_xmp=_read_regions_from_xmp,
            clear_pipeline_steps=clear_pipeline_steps,
            read_pipeline_step=read_pipeline_step,
            validate_region_set=validate_region_set,
        )
        if decision is None:
            return None
        redetect_reason, force_detect = decision

        print(f"Processing {view_path.name}{redetect_reason}...")
        if not force and read_pipeline_step(xmp_path, "view_regions") is not None:
            print(f"  Re-running {view_path.name} (pipeline state present but regions are missing)")

        img_w, img_h = _image_dimensions(view_path)
        album_context, page_caption, people_roster = _build_region_detection_context(view_path, root)
        prompt_debug = PromptDebugSession(view_path, label=view_path.name) if debug else None
        try:
            regions = detect_regions(
                view_path,
                force=force_detect,
                album_context=album_context,
                page_caption=page_caption,
                people_roster=people_roster,
                prompt_debug=prompt_debug,
                skip_validation=skip_validation,
                write_debug=debug,
            )
        finally:
            debug_path = _write_view_regions_debug_artifact(prompt_debug, image_path=view_path)
            if debug_path is not None:
                print(f"  Debug request/response log: {debug_path}")

        if debug:
            _print_view_regions_debug_paths(
                view_path,
                _accepted_regions_debug_path=_accepted_regions_debug_path,
                _docling_raw_debug_path=_docling_raw_debug_path,
            )

        if not regions:
            _handle_no_detected_view_regions(
                view_path,
                xmp_path,
                redetect_reason=redetect_reason,
                img_w=img_w,
                img_h=img_h,
                model_name=model_name,
                _failed_regions_debug_path=_failed_regions_debug_path,
                read_pipeline_step=read_pipeline_step,
                write_pipeline_step=write_pipeline_step,
                write_region_list=write_region_list,
            )
            return None

        regions_with_captions = associate_captions(regions, [], img_w)
        write_region_list(xmp_path, regions_with_captions, img_w, img_h)
        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
        print(f"  Wrote {len(regions)} region(s) to {xmp_path.name}")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return False


def _detect_view_regions_decision(
    view_path: Path,
    *,
    force: bool,
    redo_no_regions: bool,
    skip_validation: bool,
    _has_xmp_regions,
    _image_dimensions,
    _read_regions_from_xmp,
    clear_pipeline_steps,
    read_pipeline_step,
    validate_region_set,
) -> tuple[str, bool] | None:
    xmp_path = view_path.with_suffix(".xmp")
    if force:
        clear_pipeline_steps(xmp_path, ["view_regions"])
        return "", True
    if not _view_regions_step_complete(xmp_path):
        return "", False
    if not _has_xmp_regions(xmp_path):
        if not redo_no_regions:
            return None
        step = read_pipeline_step(xmp_path, "view_regions")
        result = step.get("result") if step else None
        reason = " (redo: no_regions)" if result == "no_regions" else " (redo: regionlist missing)"
        clear_pipeline_steps(xmp_path, ["view_regions"])
        return reason, True
    img_w, img_h = _image_dimensions(view_path)
    stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
    validation = validate_region_set(stored, img_w=img_w, img_h=img_h)
    if validation.valid or skip_validation:
        return None
    reasons = ", ".join(f.reason for f in validation.failures)
    clear_pipeline_steps(xmp_path, ["view_regions"])
    return f" (revalidate: {len(validation.failures)} invalid region(s): {reasons})", True


def _print_view_regions_debug_paths(
    view_path: Path,
    *,
    _accepted_regions_debug_path,
    _docling_raw_debug_path,
) -> None:
    accepted_debug_path = _accepted_regions_debug_path(view_path)
    if accepted_debug_path.is_file():
        print(f"  Accepted boxes debug image: {accepted_debug_path}")
    raw_docling_debug_path = _docling_raw_debug_path(view_path)
    if raw_docling_debug_path.is_file():
        print(f"  Docling raw debug: {raw_docling_debug_path}")


def _handle_no_detected_view_regions(
    view_path: Path,
    xmp_path: Path,
    *,
    redetect_reason: str,
    img_w: int,
    img_h: int,
    model_name: str,
    _failed_regions_debug_path,
    read_pipeline_step,
    write_pipeline_step,
    write_region_list,
) -> bool:
    if redetect_reason:
        _print_failed_regions_debug_path(view_path, _failed_regions_debug_path=_failed_regions_debug_path)
        print("  No regions detected; will retry next run.")
        return True
    write_region_list(xmp_path, [], img_w, img_h)
    existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
    if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
    _print_failed_regions_debug_path(view_path, _failed_regions_debug_path=_failed_regions_debug_path)
    print("  No regions detected.")
    return False


def _print_failed_regions_debug_path(view_path: Path, *, _failed_regions_debug_path) -> None:
    failed_debug_path = _failed_regions_debug_path(view_path)
    if failed_debug_path.is_file():
        print(f"  Failed boxes debug image: {failed_debug_path}")


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python -m photoalbums ...")
