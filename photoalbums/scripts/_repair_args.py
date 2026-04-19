from __future__ import annotations

import argparse


def build_repair_parser(
    *,
    description: str,
    default_photos_root: str,
    include_page: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--photos-root",
        default=default_photos_root,
        help="Photo Albums root directory.",
    )
    parser.add_argument(
        "--album",
        default="",
        help="Optional substring filter against the parent album directory name.",
    )
    if include_page:
        parser.add_argument("--page", default="", help="Optional page number filter.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Write changes in place. Omit for a dry run.",
    )
    return parser
