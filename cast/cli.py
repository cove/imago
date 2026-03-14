from __future__ import annotations

import argparse
from pathlib import Path

from .server import run as run_web
from .storage import TextFaceStore


def default_store_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cast.py",
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
    return parser


def cmd_init(store: TextFaceStore) -> int:
    store.ensure_files()
    print(f"Initialized cast store in: {store.root_dir}")
    print(f"- {store.people_path.name}")
    print(f"- {store.faces_path.name}")
    print(f"- {store.review_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TextFaceStore(Path(args.store_dir))
    store.ensure_files()

    if args.command == "init":
        return cmd_init(store)
    if args.command == "web":
        run_web(host=str(args.host), port=int(args.port), store=store)
        return 0
    if args.command == "label-photos":
        from .label_photos import run_label_photos
        from .ingest import FaceIngestor
        extensions = tuple(
            e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in str(args.extensions).split(",")
            if e.strip()
        )
        ingestor = FaceIngestor(store, require_primary_model=True)
        run_label_photos(
            Path(args.directory),
            store,
            ingestor=ingestor,
            overwrite=bool(args.overwrite),
            extensions=extensions,
        )
        return 0

    parser.error("Unknown command.")
    return 2
