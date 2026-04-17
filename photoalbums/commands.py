from __future__ import annotations

import json
import sys
from pathlib import Path

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


def run_ctm(argv: list[str]) -> int:
    from .lib import ai_ctm_restoration
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step
    from .naming import parse_album_filename
    from .stitch_oversized_pages import (
        _require_primary_scan,
        _view_page_output_path,
        get_photos_dirname,
        get_view_dirname,
        list_archive_dirs,
        list_page_scans,
    )

    def _match_archives(photos_root_text: str, album_id_text: str) -> list[Path]:
        archives = [Path(path) for path in list_archive_dirs(Path(photos_root_text))]
        if not album_id_text:
            return archives
        return [path for path in archives if path.name == f"{album_id_text}_Archive"]

    def _resolve_ctm_source_image(primary_scan: Path) -> Path:
        view_dir = Path(get_view_dirname(primary_scan.parent))
        view_path = _view_page_output_path(primary_scan, view_dir)
        if not view_path.is_file():
            raise RuntimeError(f"CTM source stitched image not found: {view_path}")
        return view_path

    if not argv:
        print("Error: missing CTM command")
        return 2
    command = str(argv[0]).strip().lower()
    if command not in {"generate", "review"}:
        print(f"Error: unknown CTM command: {command}")
        return 2
    args = list(argv[1:])
    force = False
    per_photo = False
    album_id = ""
    page = ""
    photos_root = "."
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--force":
            force = True
            index += 1
            continue
        if token == "--per-photo":
            per_photo = True
            index += 1
            continue
        if token == "--album-id" and index + 1 < len(args):
            album_id = args[index + 1]
            index += 2
            continue
        if token == "--page" and index + 1 < len(args):
            page = args[index + 1]
            index += 2
            continue
        if token == "--photos-root" and index + 1 < len(args):
            photos_root = args[index + 1]
            index += 2
            continue
        print(f"Error: unknown argument: {token}")
        return 2

    selected = _match_archives(photos_root, album_id)
    if not selected:
        if album_id:
            print(f"Error: no archive matched album_id={album_id!r}")
        else:
            print(f"Error: no archive directories found under photos_root={photos_root!r}")
        return 1

    matched: list[Path] = []
    for archive in selected:
        for group in list_page_scans(archive):
            primary = Path(_require_primary_scan(group))
            if page and f"P{int(page):02d}" not in primary.name:
                continue
            matched.append(primary)
    if not matched:
        print("Error: no matching archive scan pages found")
        return 1

    if command == "generate":
        for scan in matched:
            targets: list[tuple[Path, Path, str]] = []
            if per_photo:
                _, _, _, page_str = parse_album_filename(scan.name)
                photos_dir = Path(get_photos_dirname(scan.parent))
                if photos_dir.is_dir():
                    targets = [
                        (crop, crop.with_suffix(".xmp"), crop.name)
                        for crop in sorted(photos_dir.glob(f"*_P{int(page_str):02d}_D*-00_V.jpg"))
                    ]
            else:
                targets = [(_resolve_ctm_source_image(scan), scan.with_suffix(".xmp"), scan.name)]

            for source_image, sidecar_path, display_name in targets:
                if force:
                    clear_pipeline_steps(sidecar_path, ["ctm"])
                else:
                    pipeline_state = read_pipeline_step(sidecar_path, "ctm")
                    if pipeline_state is not None:
                        completed = str(pipeline_state.get("completed") or "").strip()
                        if completed:
                            print(
                                f"Skipping {display_name} CTM generation (already completed at {completed}; use --force to rerun)"
                            )
                        else:
                            print(
                                f"Skipping {display_name} CTM generation (pipeline state present; use --force to rerun)"
                            )
                        continue

                output_sidecar, result = ai_ctm_restoration.generate_and_store_ctm(
                    source_image,
                    archive_sidecar_path=sidecar_path,
                    force=force,
                )
                write_pipeline_step(output_sidecar, "ctm", model=result.model_name or None)
                print(
                    json.dumps(
                        {
                            "image": display_name,
                            "source_image": str(source_image),
                            "archive_xmp": str(output_sidecar),
                            **result.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                )
        return 0

    for scan in matched:
        state = ai_ctm_restoration.read_ctm_from_archive_xmp(scan.with_suffix(".xmp"))
        print(
            json.dumps(
                {"image": scan.name, "archive_xmp": str(scan.with_suffix(".xmp")), "ctm": state}, ensure_ascii=False
            )
        )
    return 0


def run_ctm_apply(*, album_id: str, photos_root: str, page: str | None, force: bool) -> int:
    from .lib import ai_ctm_restoration
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step
    from .stitch_oversized_pages import (
        _require_primary_scan,
        _view_page_output_path,
        get_photos_dirname,
        get_view_dirname,
        list_archive_dirs,
        list_page_scans,
    )

    def _match_archives(photos_root_text: str, album_id_text: str) -> list[Path]:
        archives = [Path(path) for path in list_archive_dirs(Path(photos_root_text))]
        if not album_id_text:
            return archives
        return [path for path in archives if path.name == f"{album_id_text}_Archive"]

    def _apply_if_needed(*, jpeg_path: Path, sidecar_path: Path, ctm_source_path: Path) -> None:
        ctm_state = ai_ctm_restoration.read_ctm_from_archive_xmp(ctm_source_path)
        if ctm_state is None:
            return
        if force:
            clear_pipeline_steps(sidecar_path, ["ctm_applied"])
        else:
            existing = read_pipeline_step(sidecar_path, "ctm_applied")
            if existing is not None:
                print(f"Skipping {jpeg_path.name} CTM apply (pipeline state present; use --force to rerun)")
                return
        matrix = ctm_state.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != 9:
            return
        ai_ctm_restoration.apply_ctm_to_jpeg(jpeg_path, matrix)
        write_pipeline_step(sidecar_path, "ctm_applied", model=str(ctm_state.get("model_name") or "").strip() or None)

    selected = _match_archives(photos_root, album_id)
    if not selected:
        if album_id:
            print(f"Error: no archive matched album_id={album_id!r}")
        else:
            print(f"Error: no archive directories found under photos_root={photos_root!r}")
        return 1

    matched: list[Path] = []
    for archive in selected:
        for group in list_page_scans(archive):
            primary = Path(_require_primary_scan(group))
            if page and f"P{int(page):02d}" not in primary.name:
                continue
            matched.append(primary)
    if not matched:
        print("Error: no matching archive scan pages found")
        return 1

    for scan in matched:
        archive_sidecar = scan.with_suffix(".xmp")
        view_dir = Path(get_view_dirname(scan.parent))
        view_path = _view_page_output_path(scan, view_dir)
        if view_path.is_file():
            _apply_if_needed(
                jpeg_path=view_path,
                sidecar_path=view_path.with_suffix(".xmp"),
                ctm_source_path=archive_sidecar,
            )

        photos_dir = Path(get_photos_dirname(scan.parent))
        page_token = f"_P{int(page):02d}_" if page else f"_P{int(scan.stem.split('_P')[1].split('_')[0]):02d}_"
        if not photos_dir.is_dir():
            continue
        for crop_path in sorted(photos_dir.glob(f"*{page_token}D*-00_V.jpg")):
            crop_xmp = crop_path.with_suffix(".xmp")
            _apply_if_needed(
                jpeg_path=crop_path,
                sidecar_path=crop_xmp,
                ctm_source_path=crop_xmp,
            )
    return 0


def run_stitch_validate() -> int:
    import stitch_oversized_pages_validate

    return _call_main(stitch_oversized_pages_validate.main)


def run_watch_incoming() -> int:
    import incoming_scans_watcher

    return _call_main(incoming_scans_watcher.main)


def run_checksum_tree(*, base_dir: str, verify: bool) -> int:
    import sha3_tree_hashes

    argv = [str(base_dir)]
    if verify:
        argv.append("--verify")
    return int(sha3_tree_hashes.run(argv) or 0)


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
    from .lib.ai_view_regions import (
        _has_xmp_regions,
        _image_dimensions,
        _read_regions_from_xmp,
        associate_captions,
        detect_regions,
        validate_region_set,
        validate_regions_for_write,
    )
    from .lib.ai_photo_crops import crop_page_regions
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step, write_region_list
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
        if d.is_dir() and d.name.endswith("_View") and (not album_id or album_id_lower in d.name.casefold())
    )
    if not view_dirs:
        print(f"No _View directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    errors = 0
    for view_dir in view_dirs:
        # Derive _Photos directory from the _Archive sibling name
        archive_name = view_dir.name.replace("_View", "_Archive")
        archive_path = root / archive_name
        photos_dir = Path(get_photos_dirname(archive_path))

        candidates = _iter_page_view_targets(view_dir, page)

        for view_path in candidates:
            print(f"Processing {view_path.name}...")
            try:
                xmp_path = view_path.with_suffix(".xmp")
                needs_detect = not _view_regions_step_complete(xmp_path)
                detect_reason = ""
                if not needs_detect and _has_xmp_regions(xmp_path):
                    # Always revalidate stored regions
                    img_w, img_h = _image_dimensions(view_path)
                    stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
                    vresult = validate_region_set(stored, img_w=img_w, img_h=img_h)
                    if not vresult.valid:
                        n_fail = len(vresult.failures)
                        reasons = ", ".join(f.reason for f in vresult.failures)
                        detect_reason = f" (revalidate: {n_fail} invalid region(s): {reasons})"
                        clear_pipeline_steps(xmp_path, ["view_regions"])
                        needs_detect = True
                if needs_detect:
                    if not detect_reason:
                        if read_pipeline_step(xmp_path, "view_regions") is not None:
                            detect_reason = " (pipeline state present but regions are missing)"
                        else:
                            detect_reason = " (generating missing page regions)"
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
                    if regions:
                        captions: list[dict] = []
                        regions_with_captions = associate_captions(regions, captions, img_w)
                        write_region_list(xmp_path, regions_with_captions, img_w, img_h)
                        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
                        print(f"  detect-regions: {len(regions)} region(s)")
                    else:
                        write_region_list(xmp_path, [], img_w, img_h)
                        existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
                        if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
                            write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
                        print(f"  detect-regions: no regions")
                n = crop_page_regions(
                    view_path,
                    photos_dir,
                    force=force,
                    skip_restoration=skip_restoration,
                    force_restoration=force_restoration,
                )
                if n > 0:
                    print(f"  Wrote {n} crop(s) to {photos_dir.name}/")
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                errors += 1

    return 1 if errors else 0


def _is_derived_view(filename: str) -> bool:
    """Return True if filename is a derived _D##-##_V.jpg output (not a page view)."""
    import re

    return bool(re.search(r"_D\d{2}-\d{2}_V\b", filename))


def _acquire_page_pipeline_lock(page_image_path: Path) -> Path:
    from .lib.ai_processing_locks import _acquire_image_processing_lock

    return _acquire_image_processing_lock(page_image_path)


def _release_page_pipeline_lock(lock_path: Path | None) -> None:
    from .lib.ai_processing_locks import _release_image_processing_lock

    _release_image_processing_lock(lock_path)


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
    from .stitch_oversized_pages import get_photos_dirname

    root = Path(photos_root)
    album_id_lower = album_id.casefold()

    view_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name.endswith("_View") and (not album_id or album_id_lower in d.name.casefold())
    )
    if not view_dirs:
        print(f"No _View directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    targets: list[Path] = []
    for view_dir in view_dirs:
        archive_name = view_dir.name.replace("_View", "_Archive")
        archive_path = root / archive_name
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
      render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply
    """
    from .lib.ai_ctm_restoration import apply_ctm_to_jpeg, read_ctm_from_archive_xmp
    from .lib.ai_photo_crops import CropPageStats, crop_page_regions
    from .lib.ai_render_face_refresh import FaceRefreshSkipped, RenderFaceRefreshSession
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
        get_photos_dirname,
        get_view_dirname,
        list_archive_dirs,
        list_derived_images,
        list_page_scans,
        stitch,
        tif_to_jpg,
        derived_to_jpg,
    )

    root = Path(photos_root)
    album_id_lower = album_id.casefold()
    model_name = default_view_region_model()

    # Collect archive directories to process
    archives = [
        Path(p)
        for p in list_archive_dirs(root)
        if not album_id or album_id_lower in Path(p).name.casefold()
    ]
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
        "ctm_applied_outputs": 0,
    }

    face_session = RenderFaceRefreshSession(photos_root=root)

    for archive in archives:
        view_dir = Path(get_view_dirname(archive))
        photos_dir = Path(get_photos_dirname(archive))

        # Collect page scan groups to process
        page_groups = list_page_scans(archive)
        if page is not None:
            page_groups = [g for g in page_groups if f"P{str(page).zfill(2)}" in Path(_require_primary_scan(g)).name]

        for group in page_groups:
            primary_scan = Path(_require_primary_scan(group))
            view_path = _view_page_output_path(primary_scan, view_dir)
            xmp_path = view_path.with_suffix(".xmp")
            archive_sidecar = primary_scan.with_suffix(".xmp")
            page_label = view_path.name
            summary["pages_seen"] += 1
            print(f"Processing {page_label}...")
            lock_path: Path | None = None

            try:
                lock_path = _acquire_page_pipeline_lock(view_path)

                # Step: render
                try:
                    if len(group) > 1:
                        stitch(group, str(view_dir))
                    else:
                        tif_to_jpg(str(primary_scan), str(view_dir))
                    # Also render derived images for this page
                    page_token = f"_P{str(primary_scan.stem.split('_P')[1].split('_')[0]).zfill(2)}_"
                    for derived in list_derived_images(archive):
                        if page_token in Path(derived).name or not page:
                            derived_to_jpg(derived, str(view_dir))
                except Exception as exc:
                    print(f"  ERROR [render]: {exc}", file=sys.stderr)
                    failures.append((page_label, "render", str(exc)))
                    summary["errors"] += 1
                    summary["pages_failed"] += 1
                    continue

                # Step: propagate-archive-metadata
                try:
                    if view_path.is_file() and archive_sidecar.is_file():
                        propagate_archive_copy_safe_fields(xmp_path, archive_sidecar)
                except Exception as exc:
                    print(f"  WARNING [propagate-archive-metadata]: {exc}", file=sys.stderr)
                    summary["warnings"] += 1

                if _is_title_page_view(view_path):
                    summary["detect_regions_skipped"] += 1
                    if not skip_crops:
                        summary["crop_steps_skipped"] += 1
                    print("  detect-regions: skipped title page (P01)")
                    if not skip_crops:
                        print("  crop-regions: skipped title page (P01)")
                else:
                    # Step: detect-regions
                    try:
                        skip_detect = False
                        if force:
                            clear_pipeline_steps(xmp_path, ["view_regions"])
                        elif _view_regions_step_complete(xmp_path):
                            if _has_xmp_regions(xmp_path):
                                # Always revalidate stored regions on every run
                                img_w, img_h = _image_dimensions(view_path)
                                stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
                                vresult = validate_region_set(stored, img_w=img_w, img_h=img_h)
                                if vresult.valid or skip_validation:
                                    skip_detect = True
                                    summary["detect_regions_skipped"] += 1
                                else:
                                    n_fail = len(vresult.failures)
                                    reasons = ", ".join(f.reason for f in vresult.failures)
                                    print(f"  detect-regions: revalidate: {n_fail} invalid region(s): {reasons}")
                                    clear_pipeline_steps(xmp_path, ["view_regions"])
                            else:
                                skip_detect = True
                                summary["detect_regions_skipped"] += 1
                        if not skip_detect:
                            if not force and read_pipeline_step(xmp_path, "view_regions") is not None:
                                summary["detect_regions_reran"] += 1
                                print(f"  detect-regions: re-run (pipeline state present but regions are missing)")
                            img_w, img_h = _image_dimensions(view_path)
                            album_context, page_caption, people_roster = _build_region_detection_context(view_path, root)
                            prompt_debug = PromptDebugSession(view_path, label=view_path.name) if debug else None
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
                                debug_path = _write_view_regions_debug_artifact(prompt_debug, image_path=view_path)
                                if debug_path is not None:
                                    print(f"  detect-regions: debug -> {debug_path}")
                            if debug:
                                accepted_debug_path = _accepted_regions_debug_path(view_path)
                                if accepted_debug_path.is_file():
                                    print(f"  detect-regions: accepted boxes -> {accepted_debug_path}")
                                raw_docling_debug_path = _docling_raw_debug_path(view_path)
                                if raw_docling_debug_path.is_file():
                                    print(f"  detect-regions: docling debug -> {raw_docling_debug_path}")
                            # detect_regions validates internally and returns only the valid kept set
                            if regions:
                                captions: list[dict] = []
                                regions_with_captions = associate_captions(regions, captions, img_w)
                                write_region_list(xmp_path, regions_with_captions, img_w, img_h)
                                write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
                                summary["detect_regions_found"] += 1
                                print(f"  detect-regions: {len(regions)} region(s)")
                            else:
                                write_region_list(xmp_path, [], img_w, img_h)
                                existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
                                if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
                                    write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
                                summary["detect_regions_no_regions"] += 1
                                if debug:
                                    failed_debug_path = _failed_regions_debug_path(view_path)
                                    if failed_debug_path.is_file():
                                        print(f"  detect-regions: failed boxes -> {failed_debug_path}")
                                print(f"  detect-regions: no regions")
                    except Exception as exc:
                        print(f"  ERROR [detect-regions]: {exc}", file=sys.stderr)
                        failures.append((page_label, "detect-regions", str(exc)))
                        summary["errors"] += 1
                        summary["pages_failed"] += 1
                        continue

                # Step: crop-regions
                if not skip_crops and not _is_title_page_view(view_path):
                    try:
                        crop_stats = CropPageStats()
                        n = crop_page_regions(
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
                    except Exception as exc:
                        print(f"  ERROR [crop-regions]: {exc}", file=sys.stderr)
                        failures.append((page_label, "crop-regions", str(exc)))
                        summary["errors"] += 1
                        summary["pages_failed"] += 1
                        continue

                # Step: face-refresh (page view, derived views, and crops)
                try:
                    refresh_targets = _iter_face_refresh_targets(view_dir, photos_dir, page)
                    face_session.set_files(refresh_targets)
                    for img_path in refresh_targets:
                        try:
                            face_session.refresh_face_regions(img_path, img_path.with_suffix(".xmp"), force=force)
                        except FaceRefreshSkipped as exc:
                            print(f"  face-refresh: {exc}")
                        except Exception as exc:
                            print(f"  WARNING [face-refresh] {img_path.name}: {exc}", file=sys.stderr)
                            summary["warnings"] += 1
                except Exception as exc:
                    print(f"  ERROR [face-refresh]: {exc}", file=sys.stderr)
                    failures.append((page_label, "face-refresh", str(exc)))
                    summary["errors"] += 1
                    summary["pages_failed"] += 1
                    continue

                # Step: ctm-apply (page view)
                try:
                    def _apply_ctm_if_needed(jpeg_path: Path, sidecar_path: Path, ctm_source: Path) -> bool:
                        ctm_state = read_ctm_from_archive_xmp(ctm_source)
                        if ctm_state is None:
                            return False
                        if force:
                            clear_pipeline_steps(sidecar_path, ["ctm_applied"])
                        else:
                            if read_pipeline_step(sidecar_path, "ctm_applied") is not None:
                                return False
                        matrix = ctm_state.get("matrix")
                        if not isinstance(matrix, list) or len(matrix) != 9:
                            return False
                        apply_ctm_to_jpeg(jpeg_path, matrix)
                        write_pipeline_step(sidecar_path, "ctm_applied", model=str(ctm_state.get("model_name") or "").strip() or None)
                        return True

                    if view_path.is_file():
                        if _apply_ctm_if_needed(view_path, xmp_path, archive_sidecar):
                            summary["ctm_applied_outputs"] += 1
                    if photos_dir.is_dir():
                        page_token = f"_P{str(primary_scan.stem.split('_P')[1].split('_')[0]).zfill(2)}_"
                        for crop in sorted(photos_dir.glob(f"*{page_token}D*-00_V.jpg")):
                            crop_xmp = crop.with_suffix(".xmp")
                            if _apply_ctm_if_needed(crop, crop_xmp, crop_xmp):
                                summary["ctm_applied_outputs"] += 1
                except Exception as exc:
                    print(f"  ERROR [ctm-apply]: {exc}", file=sys.stderr)
                    failures.append((page_label, "ctm-apply", str(exc)))
                    summary["errors"] += 1
                    summary["pages_failed"] += 1
                    continue

                summary["pages_completed"] += 1

            except Exception as exc:
                print(f"  ERROR [page-lock]: {exc}", file=sys.stderr)
                failures.append((page_label, "page-lock", str(exc)))
                summary["errors"] += 1
                summary["pages_failed"] += 1
            finally:
                _release_page_pipeline_lock(lock_path)

    print("\n===== PIPELINE SUMMARY =====")
    print(f"Pages: {summary['pages_seen']} total, {summary['pages_completed']} completed, {summary['pages_failed']} failed")
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
    print(f"CTM apply: {summary['ctm_applied_outputs']} output(s) updated")

    if failures:
        print("\n===== PIPELINE FAILURE SUMMARY =====", file=sys.stderr)
        for page_name, step, msg in failures:
            print(f"  {page_name} [{step}]: {msg}", file=sys.stderr)
        return 1

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
    from .lib.ai_model_settings import default_view_region_model
    from .lib.prompt_debug import PromptDebugSession
    from .lib.xmp_sidecar import clear_pipeline_steps, read_pipeline_step, write_pipeline_step, write_region_list

    root = Path(photos_root)
    album_id_lower = album_id.casefold()
    model_name = default_view_region_model()

    view_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.endswith("_View") and album_id_lower in d.name.casefold()
    )
    if not view_dirs:
        print(f"No _View directories found matching '{album_id}' under {root}", file=sys.stderr)
        return 1

    errors = 0
    for view_dir in view_dirs:
        candidates = _iter_page_view_targets(view_dir, page)
        for view_path in candidates:
            xmp_path = view_path.with_suffix(".xmp")
            try:
                redetect_reason = ""
                force_detect = force
                if force:
                    clear_pipeline_steps(xmp_path, ["view_regions"])
                elif _view_regions_step_complete(xmp_path):
                    if not _has_xmp_regions(xmp_path):
                        if not redo_no_regions:
                            continue
                        step = read_pipeline_step(xmp_path, "view_regions")
                        result = step.get("result") if step else None
                        redetect_reason = " (redo: no_regions)" if result == "no_regions" else " (redo: regionlist missing)"
                        clear_pipeline_steps(xmp_path, ["view_regions"])
                        force_detect = True
                    else:
                        img_w, img_h = _image_dimensions(view_path)
                        stored = _read_regions_from_xmp(xmp_path, img_w, img_h)
                        validation = validate_region_set(stored, img_w=img_w, img_h=img_h)
                        if validation.valid or skip_validation:
                            continue
                        reasons = ", ".join(f.reason for f in validation.failures)
                        redetect_reason = f" (revalidate: {len(validation.failures)} invalid region(s): {reasons})"
                        clear_pipeline_steps(xmp_path, ["view_regions"])
                        force_detect = True

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
                    accepted_debug_path = _accepted_regions_debug_path(view_path)
                    if accepted_debug_path.is_file():
                        print(f"  Accepted boxes debug image: {accepted_debug_path}")
                    raw_docling_debug_path = _docling_raw_debug_path(view_path)
                    if raw_docling_debug_path.is_file():
                        print(f"  Docling raw debug: {raw_docling_debug_path}")

                if not regions:
                    if redetect_reason:
                        failed_debug_path = _failed_regions_debug_path(view_path)
                        if failed_debug_path.is_file():
                            print(f"  Failed boxes debug image: {failed_debug_path}")
                        print("  No regions detected; will retry next run.")
                        continue
                    write_region_list(xmp_path, [], img_w, img_h)
                    existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
                    if str(existing_step.get("result") or "") not in {"no_regions", "validation_failed", "failed"}:
                        write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "no_regions"})
                    failed_debug_path = _failed_regions_debug_path(view_path)
                    if failed_debug_path.is_file():
                        print(f"  Failed boxes debug image: {failed_debug_path}")
                    print("  No regions detected.")
                    continue

                regions_with_captions = associate_captions(regions, [], img_w)
                write_region_list(xmp_path, regions_with_captions, img_w, img_h)
                write_pipeline_step(xmp_path, "view_regions", model=model_name, extra={"result": "regions_found"})
                print(f"  Wrote {len(regions)} region(s) to {xmp_path.name}")
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
