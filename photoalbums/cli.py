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

    if args.group == "render":
        if not getattr(args, "render_kind", None):
            return commands.run_render()
        if args.render_kind == "validate":
            return commands.run_stitch_validate()

    if args.group == "watch":
        return commands.run_watch_incoming()

    if args.group == "checksum":
        if args.checksum_kind == "tree":
            return commands.run_checksum_tree(base_dir=args.base_dir, verify=bool(args.verify))

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
