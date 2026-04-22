from __future__ import annotations

import argparse

AI_STEP_LINES = [
    "AI pipeline steps:",
    "1. Load per-archive render settings overrides (if present).",
    "2. Match known people from Cast face embeddings (optional).",
    "3. Detect visual objects with YOLO (optional).",
    "4. Generate OCR text and sentence caption in one combined model call.",
    "5. Estimate GPS location from image content and context (re-runnable via --reprocess-mode=gps or `ai gps`).",
    "6. Geocode location name to GPS coordinates via Nominatim.",
    "7. Write XMP sidecar metadata (including processing state).",
]


def _import_commands():
    try:
        from . import commands

        return commands
    except Exception:
        import commands

        return commands


def _strip_remainder(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def _print_ai_steps() -> None:
    for line in AI_STEP_LINES:
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photoalbums.py",
        description="Unified command surface for photo album archive workflows.",
    )
    subparsers = parser.add_subparsers(dest="group", required=True)

    ai_parser = subparsers.add_parser(
        "ai",
        add_help=False,
        help="AI-assisted indexing (people -> objects -> combined OCR+caption -> XMP).",
    )
    ai_parser.add_argument(
        "-h",
        "--help",
        dest="ai_help",
        action="store_true",
        help="Show ai index help.",
    )

    metadata_parser = subparsers.add_parser("metadata", help="Metadata commands")
    metadata_sub = metadata_parser.add_subparsers(dest="metadata_kind", required=True)
    metadata_sub.add_parser("apply", help="Apply standardized metadata tags to TIFF scans")
    metadata_sub.add_parser("tsv", help="Deprecated metadata.tsv export command")
    map_parser = metadata_sub.add_parser(
        "map", help="Launch a local Web UI map to manually drag-and-drop GPS locations"
    )
    map_parser.add_argument(
        "paths",
        nargs="+",
        help="Directories or .xmp files to load into the map",
    )
    map_parser.add_argument(
        "--port",
        type=int,
        default=8095,
        help="Port to run the HTTP map server on (default 8095)",
    )

    subparsers.add_parser("compress", help="Compress TIFF scans in-place where needed")

    render_parser = subparsers.add_parser("render", help="Render album page outputs (skips existing valid outputs)")
    render_sub = render_parser.add_subparsers(dest="render_kind", required=False)
    render_sub.add_parser("validate", help="Validate source scan stitchability without writing outputs")

    subparsers.add_parser("watch", help="Watch for incoming scans and register pending events")

    detect_vr_parser = subparsers.add_parser(
        "detect-view-regions",
        help="Detect photo regions in view JPGs and write MWG-RS XMP region metadata.",
    )
    detect_vr_parser.add_argument(
        "album_id",
        nargs="?",
        default="",
        help="Album folder name fragment (e.g. 'Egypt_1975_B00'); omit for all albums",
    )
    detect_vr_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    detect_vr_parser.add_argument("--page", default=None, help="Page number to process (e.g. '26'); omit for all pages")
    detect_vr_parser.add_argument("--force", action="store_true", help="Re-run detection even if regions already exist")
    detect_vr_parser.add_argument(
        "--redo-no-regions",
        action="store_true",
        help="Re-run detection on pages previously marked as having no regions",
    )
    detect_vr_parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-image region detection request/response debug artifacts.",
    )
    detect_vr_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip strict region validations (e.g. for docling)",
    )

    crop_regions_parser = subparsers.add_parser(
        "crop-regions",
        help="Crop detected MWG-RS photo regions from view JPEGs and write to _Photos/ directory.",
    )
    crop_regions_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    crop_regions_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    crop_regions_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    crop_regions_parser.add_argument(
        "--force", action="store_true", help="Re-crop even if pipeline state already recorded"
    )
    crop_regions_parser.add_argument(
        "--skip-restoration", action="store_true", help="Skip AI photo restoration step (faster on CPU)"
    )
    crop_regions_parser.add_argument(
        "--force-restoration",
        action="store_true",
        help="Re-run photo restoration on existing crop outputs without forcing the full crop step.",
    )

    face_refresh_parser = subparsers.add_parser(
        "face-refresh",
        help="Refresh face regions on rendered JPEG sidecars using Cast-backed buffalo_l.",
    )
    face_refresh_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    face_refresh_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    face_refresh_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    face_refresh_parser.add_argument(
        "--force", action="store_true", help="Re-run face refresh even if pipeline state already recorded"
    )

    render_pipeline_parser = subparsers.add_parser(
        "render-pipeline",
        help="Run the full render pipeline (detect-regions, crop-regions, ...) for a matching album.",
    )
    render_pipeline_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    render_pipeline_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    render_pipeline_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    render_pipeline_parser.add_argument("--force", action="store_true", help="Re-run all pipeline steps")
    render_pipeline_parser.add_argument("--skip-crops", action="store_true", help="Skip the crop-regions step")
    render_pipeline_parser.add_argument(
        "--force-restoration",
        action="store_true",
        help="Re-run photo restoration on existing crop outputs during crop-regions.",
    )
    render_pipeline_parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-image region detection request/response debug artifacts.",
    )
    render_pipeline_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip strict region validations (e.g. for docling)",
    )

    checksum_parser = subparsers.add_parser("checksum", help="Checksum manifest commands")
    checksum_sub = checksum_parser.add_subparsers(dest="checksum_kind", required=True)
    checksum_tree = checksum_sub.add_parser("tree", help="Generate or verify SHA256 tree manifests")
    checksum_tree.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Path to the Photo Albums directory tree (default: current directory).",
    )
    checksum_tree.add_argument(
        "--verify",
        "--check",
        dest="verify",
        action="store_true",
        help="Verify hashes against existing manifests instead of generating them.",
    )

    migrate_refs_parser = subparsers.add_parser(
        "migrate-page-dir-refs",
        help="Rewrite page-side XMP references from *_View to *_Pages under a Photo Albums root.",
    )
    migrate_refs_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    migrate_refs_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail if any .xmp files under the root still contain _View.",
    )

    migrate_caption_parser = subparsers.add_parser(
        "migrate-caption-layout",
        help="Rewrite page and crop XMP sidecars to the current caption and parent-OCR layout.",
    )
    migrate_caption_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    migrate_caption_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail if any .xmp files under the root still use the legacy caption layout.",
    )

    migrate_pipeline_parser = subparsers.add_parser(
        "migrate-pipeline-records",
        help="Rewrite legacy pipeline step records ({completed: ts}) to the new schema ({timestamp, result, input_hash}) across a directory tree.",
    )
    migrate_pipeline_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    migrate_pipeline_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail if any .xmp files under the root still contain legacy pipeline records.",
    )

    repair_crop_source_parser = subparsers.add_parser(
        "repair-crop-source",
        help="Rewrite crop XMP imago:AlbumTitle and dc:source from archive scan lineage without rerunning crops.",
    )
    repair_crop_source_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    repair_crop_source_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    repair_crop_source_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    repair_crop_source_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail if any crop sidecars under the selection still need dc:source/AlbumTitle repair.",
    )

    repair_crop_numbers_parser = subparsers.add_parser(
        "repair-crop-numbers",
        help="Rename crop JPEG/XMP pairs to canonical derived numbers after the page's archive-derived max.",
    )
    repair_crop_numbers_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    repair_crop_numbers_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    repair_crop_numbers_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")

    repair_page_derived_views_parser = subparsers.add_parser(
        "repair-page-derived-views",
        help="Force-regenerate _Pages derived views from _Archive and remove orphan page-derived outputs.",
    )
    repair_page_derived_views_parser.add_argument(
        "album_id", nargs="?", default="", help="Album folder name fragment; omit for all albums"
    )
    repair_page_derived_views_parser.add_argument(
        "--photos-root", required=True, help="Path to the Photo Albums root directory"
    )
    repair_page_derived_views_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")

    process_parser = subparsers.add_parser(
        "process",
        help="Run the full pipeline (render → propagate-metadata → detect-regions → crop-regions → face-refresh → ai-index).",
    )
    process_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    process_parser.add_argument("--album", default="", dest="album_id", help="Album folder name fragment; omit for all albums")
    process_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    process_parser.add_argument(
        "--skip",
        dest="skip_ids",
        action="append",
        default=[],
        metavar="STEP",
        help="Skip a pipeline step by id (repeatable); mutually exclusive with --step",
    )
    process_parser.add_argument(
        "--redo",
        dest="redo_ids",
        action="append",
        default=[],
        metavar="STEP",
        help="Force-rerun a pipeline step by id (repeatable)",
    )
    process_parser.add_argument(
        "--step",
        dest="step_id",
        default=None,
        metavar="STEP",
        help="Run exactly one step; mutually exclusive with --skip",
    )
    process_parser.add_argument("--force", action="store_true", help="Force-rerun all steps (shorthand for --redo on all)")
    process_parser.add_argument("--debug", action="store_true", help="Write debug artifacts for detect-regions and ai-index")
    process_parser.add_argument("--no-validation", action="store_true", help="Skip strict region validations")
    process_parser.add_argument("--skip-restoration", action="store_true", help="Skip AI photo restoration in crop-regions")
    process_parser.add_argument(
        "--force-restoration", action="store_true", help="Re-run photo restoration in crop-regions"
    )
    process_parser.add_argument(
        "--gps-only", action="store_true", help="Forward --reprocess-mode=gps to ai-index step"
    )
    process_parser.add_argument(
        "--refresh-gps",
        action="store_true",
        help="Run the full pipeline but force ai-index to rerun GPS geocoding via --reprocess-mode=gps",
    )
    process_parser.add_argument("--list-steps", action="store_true", help="Print the step registry and exit")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    commands = _import_commands()

    if args.group == "process":
        from .lib.pipeline import PIPELINE_STEPS, VALID_STEP_IDS, validate_step_ids

        if bool(getattr(args, "list_steps", False)):
            for i, step in enumerate(PIPELINE_STEPS, 1):
                print(f"  [{i}] {step.id}  —  {step.label}")
            return 0

        if args.step_id and args.skip_ids:
            parser.error("--step and --skip are mutually exclusive")

        if args.step_id:
            validate_step_ids([args.step_id], flag="--step")
        if args.skip_ids:
            validate_step_ids(args.skip_ids, flag="--skip")
        if args.redo_ids:
            validate_step_ids(args.redo_ids, flag="--redo")

        return commands.run_process_pipeline(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            skip_ids=list(args.skip_ids or []),
            redo_ids=list(args.redo_ids or []),
            step_id=args.step_id,
            force=bool(getattr(args, "force", False)),
            debug=bool(getattr(args, "debug", False)),
            no_validation=bool(getattr(args, "no_validation", False)),
            skip_restoration=bool(getattr(args, "skip_restoration", False)),
            force_restoration=bool(getattr(args, "force_restoration", False)),
            gps_only=bool(getattr(args, "gps_only", False)),
            refresh_gps=bool(getattr(args, "refresh_gps", False)),
        )

    allow_extras = args.group == "ai"
    if extras and not allow_extras:
        parser.error("Unrecognized arguments: " + " ".join(extras))

    if args.group == "ai":
        forwarded = list(extras or [])
        if forwarded and forwarded[0] == "steps":
            _print_ai_steps()
            return 0
        if forwarded and forwarded[0] == "gps":
            forwarded = ["--reprocess-mode", "gps", *forwarded[1:]]
        if forwarded and forwarded[0] == "index":
            forwarded = forwarded[1:]
        elif forwarded and not str(forwarded[0]).startswith("-"):
            parser.error(f"Unknown ai command: {forwarded[0]}")
        forwarded = _strip_remainder(forwarded)
        if bool(getattr(args, "ai_help", False)):
            forwarded = ["--help", *forwarded]
        return commands.run_ai_index(forwarded)

    if args.group == "metadata":
        if args.metadata_kind == "apply":
            return commands.run_apply_metadata()
        if args.metadata_kind == "tsv":
            return commands.run_create_metadata_tsv()
        if args.metadata_kind == "map":
            return commands.run_metadata_map(paths=args.paths, port=args.port)

    if args.group == "compress":
        return commands.run_compress_tiff()

    if args.group == "render":
        if not getattr(args, "render_kind", None):
            return commands.run_render()
        if args.render_kind == "validate":
            return commands.run_stitch_validate()

    if args.group == "watch":
        return commands.run_watch_incoming()

    if args.group == "detect-view-regions":
        return commands.run_detect_view_regions(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            force=args.force,
            redo_no_regions=bool(getattr(args, "redo_no_regions", False)),
            debug=bool(getattr(args, "debug", False)),
            skip_validation=bool(getattr(args, "no_validation", False)),
        )

    if args.group == "crop-regions":
        return commands.run_crop_regions(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            force=args.force,
            skip_restoration=bool(getattr(args, "skip_restoration", False)),
            force_restoration=bool(getattr(args, "force_restoration", False)),
        )

    if args.group == "face-refresh":
        return commands.run_face_refresh(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            force=args.force,
        )

    if args.group == "render-pipeline":
        return commands.run_render_pipeline(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            force=args.force,
            skip_crops=bool(getattr(args, "skip_crops", False)),
            force_restoration=bool(getattr(args, "force_restoration", False)),
            debug=bool(getattr(args, "debug", False)),
            skip_validation=bool(getattr(args, "no_validation", False)),
        )

    if args.group == "checksum":
        if args.checksum_kind == "tree":
            return commands.run_checksum_tree(base_dir=args.base_dir, verify=bool(args.verify))

    if args.group == "migrate-page-dir-refs":
        return commands.run_migrate_page_dir_refs(
            photos_root=args.photos_root,
            verify_only=bool(getattr(args, "verify_only", False)),
        )

    if args.group == "migrate-caption-layout":
        return commands.run_migrate_caption_layout(
            photos_root=args.photos_root,
            verify_only=bool(getattr(args, "verify_only", False)),
        )

    if args.group == "migrate-pipeline-records":
        return commands.run_migrate_pipeline_records(
            photos_root=args.photos_root,
            verify_only=bool(getattr(args, "verify_only", False)),
        )

    if args.group == "repair-crop-source":
        return commands.run_repair_crop_source(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            verify_only=bool(getattr(args, "verify_only", False)),
        )

    if args.group == "repair-crop-numbers":
        return commands.run_repair_crop_numbers(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
        )

    if args.group == "repair-page-derived-views":
        return commands.run_repair_page_derived_views(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
        )

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
