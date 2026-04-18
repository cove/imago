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

    ctm_parser = subparsers.add_parser("ctm", help="Generate or review CTM restoration metadata")
    ctm_sub = ctm_parser.add_subparsers(dest="ctm_kind", required=True)
    for name in ("generate", "review"):
        sub = ctm_sub.add_parser(name, help=f"{name.title()} CTM restoration metadata")
        sub.add_argument("--album-id", help="Album identifier without _Archive suffix; omit for the full album set")
        sub.add_argument("--page", help="Optional page number")
        sub.add_argument("--photos-root", default=".", help="Photo Albums root")
        if name == "generate":
            sub.add_argument("--force", action="store_true", help="Regenerate even if CTM already exists")
            sub.add_argument("--per-photo", action="store_true", help="Generate CTMs for crop JPEGs in _Photos/")

    ctm_apply_parser = subparsers.add_parser("ctm-apply", help="Apply stored CTM matrices to rendered JPEGs")
    ctm_apply_parser.add_argument(
        "album_id", nargs="?", default="", help="Album identifier without _Archive suffix; omit for the full album set"
    )
    ctm_apply_parser.add_argument("--photos-root", required=True, help="Path to the Photo Albums root directory")
    ctm_apply_parser.add_argument("--page", default=None, help="Page number to process; omit for all pages")
    ctm_apply_parser.add_argument("--force", action="store_true", help="Re-apply even if pipeline state exists")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    commands = _import_commands()

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

    if args.group == "ctm":
        forwarded = [str(args.ctm_kind)]
        if getattr(args, "album_id", None):
            forwarded += ["--album-id", str(args.album_id)]
        if getattr(args, "page", None):
            forwarded += ["--page", str(args.page)]
        forwarded += ["--photos-root", str(args.photos_root)]
        if bool(getattr(args, "force", False)):
            forwarded.append("--force")
        if bool(getattr(args, "per_photo", False)):
            forwarded.append("--per-photo")
        return commands.run_ctm(forwarded)

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

    if args.group == "ctm-apply":
        return commands.run_ctm_apply(
            album_id=args.album_id,
            photos_root=args.photos_root,
            page=args.page,
            force=args.force,
        )

    if args.group == "checksum":
        if args.checksum_kind == "tree":
            return commands.run_checksum_tree(base_dir=args.base_dir, verify=bool(args.verify))

    if args.group == "migrate-page-dir-refs":
        return commands.run_migrate_page_dir_refs(
            photos_root=args.photos_root,
            verify_only=bool(getattr(args, "verify_only", False)),
        )

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
