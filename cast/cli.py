from __future__ import annotations

import argparse
import os
from pathlib import Path

from .server import run as run_web
from .storage import TextFaceStore


def default_store_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cast",
        description="Shared face identity storage and web review UI.",
    )
    parser.add_argument(
        "--store-dir",
        default=str(default_store_dir()),
        help="Directory containing people.json, faces.jsonl, review_queue.jsonl",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize text storage files.")

    web = subparsers.add_parser("web", help="Run local Cast web UI.")
    web.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    web.add_argument("--port", type=int, default=8093, help="Bind port (default: 8093)")
    web.add_argument(
        "--lmstudio-url",
        default="",
        help=(
            "Base URL of a running LM Studio instance. "
            "Defaults to the standard imago LM Studio URL. "
            "Pass an empty string to disable description rewrites."
        ),
    )

    label = subparsers.add_parser(
        "label-photos",
        help="Interactively label people in photos and save identities to XMP sidecars.",
    )
    label.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for photos (recursively). Defaults to current directory.",
    )
    label.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label photos that already have PersonInImage in their XMP sidecar.",
    )
    label.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.tif,.tiff,.png",
        help="Comma-separated photo extensions to scan (default: .jpg,.jpeg,.tif,.tiff,.png).",
    )

    immich = subparsers.add_parser(
        "immich-sync",
        help=(
            "Sync named face bounding boxes from Immich into local XMP sidecars "
            "and the cast store."
        ),
    )
    immich.add_argument(
        "--immich-url",
        default=os.environ.get("IMMICH_URL", ""),
        help="Immich base URL, e.g. http://immich.local:2283 (or set IMMICH_URL env var).",
    )
    immich.add_argument(
        "--api-key",
        default=os.environ.get("IMMICH_API_KEY", ""),
        help="Immich API key from Account Settings → API Keys (or set IMMICH_API_KEY env var).",
    )
    immich.add_argument(
        "--photos-root",
        default=".",
        help="Root directory of local photo albums to update. Defaults to current directory.",
    )
    immich.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing any files.",
    )
    immich.add_argument(
        "--skip-castdb",
        action="store_true",
        help="Do not add new Immich people names to the cast store.",
    )
    immich.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.tif,.tiff,.png",
        help="Comma-separated photo extensions to match (default: .jpg,.jpeg,.tif,.tiff,.png).",
    )
    immich.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    immich_cast_import = subparsers.add_parser(
        "immich-cast-import",
        help="Import high-confidence visible face crops from Immich manual person regions into Cast.",
    )
    immich_cast_import.add_argument(
        "--immich-url",
        default=os.environ.get("IMMICH_URL", ""),
        help="Immich base URL, e.g. http://immich.local:2283 (or set IMMICH_URL env var).",
    )
    immich_cast_import.add_argument(
        "--api-key",
        default=os.environ.get("IMMICH_API_KEY", ""),
        help="Immich API key from Account Settings → API Keys (or set IMMICH_API_KEY env var).",
    )
    immich_cast_import.add_argument(
        "--photos-root",
        default=".",
        help="Root directory of local photo albums to inspect. Defaults to current directory.",
    )
    immich_cast_import.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.tif,.tiff,.png",
        help="Comma-separated photo extensions to match (default: .jpg,.jpeg,.tif,.tiff,.png).",
    )

    return parser


def cmd_init(store: TextFaceStore) -> int:
    store.ensure_files()
    print(f"Initialized cast store in: {store.root_dir}")
    print(f"- {store.people_path.name}")
    print(f"- {store.faces_path.name}")
    print(f"- {store.review_path.name}")
    return 0


def _parse_extensions(raw: str) -> tuple[str, ...]:
    return tuple(
        e.strip() if e.strip().startswith(".") else f".{e.strip()}"
        for e in raw.split(",")
        if e.strip()
    )


def cmd_immich_sync(args: argparse.Namespace, store: TextFaceStore, parser: argparse.ArgumentParser) -> int:
    import logging

    from .immich_sync import IMMICH_API_KEY_ENV, IMMICH_URL_ENV, sync_immich_faces

    if bool(getattr(args, "verbose", False)):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    immich_url = str(getattr(args, "immich_url", "") or "").rstrip("/")
    api_key = str(getattr(args, "api_key", "") or "")
    if not immich_url:
        parser.error(f"--immich-url is required (or set {IMMICH_URL_ENV}).")
    if not api_key:
        parser.error(f"--api-key is required (or set {IMMICH_API_KEY_ENV}).")

    stats = sync_immich_faces(
        immich_url,
        api_key,
        Path(str(getattr(args, "photos_root", ".") or ".")),
        store,
        dry_run=bool(getattr(args, "dry_run", False)),
        update_castdb=not bool(getattr(args, "skip_castdb", False)),
        extensions=_parse_extensions(str(args.extensions)),
    )
    print(f"People synced to cast store : {stats['people_synced']}")
    print(f"Assets matched to local files: {stats['assets_matched']}")
    print(f"XMP sidecars updated         : {stats['xmp_updated']}")
    print(f"Assets with no named faces   : {stats['xmp_skipped']}")
    return 0


def cmd_immich_cast_import(args: argparse.Namespace, store: TextFaceStore, parser: argparse.ArgumentParser) -> int:
    from .immich_sync import IMMICH_API_KEY_ENV, IMMICH_URL_ENV, import_immich_cast_faces

    immich_url = str(getattr(args, "immich_url", "") or "").rstrip("/")
    api_key = str(getattr(args, "api_key", "") or "")
    if not immich_url:
        parser.error(f"--immich-url is required (or set {IMMICH_URL_ENV}).")
    if not api_key:
        parser.error(f"--api-key is required (or set {IMMICH_API_KEY_ENV}).")

    stats = import_immich_cast_faces(
        immich_url,
        api_key,
        Path(str(getattr(args, "photos_root", ".") or ".")),
        store,
        extensions=_parse_extensions(str(args.extensions)),
    )
    print(f"Assets matched exactly : {stats['assets_matched']}")
    print(f"Faces imported to Cast : {stats['faces_imported']}")
    print(f"Face review seeds added: {stats['review_seeds']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TextFaceStore(Path(args.store_dir))
    store.ensure_files()

    if args.command == "init":
        return cmd_init(store)
    if args.command == "web":
        run_web(
            host=str(args.host),
            port=int(args.port),
            store=store,
            lmstudio_url=str(getattr(args, "lmstudio_url", "") or ""),
        )
        return 0
    if args.command == "label-photos":
        from .ingest import FaceIngestor
        from .label_photos import run_label_photos

        ingestor = FaceIngestor(store, require_primary_model=True)
        run_label_photos(
            Path(args.directory),
            store,
            ingestor=ingestor,
            overwrite=bool(args.overwrite),
            extensions=_parse_extensions(str(args.extensions)),
        )
        return 0
    if args.command == "immich-sync":
        return cmd_immich_sync(args, store, parser)
    if args.command == "immich-cast-import":
        return cmd_immich_cast_import(args, store, parser)

    parser.error("Unknown command.")
    return 2
