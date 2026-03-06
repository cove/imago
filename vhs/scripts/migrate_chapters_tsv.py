#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import METADATA_DIR
from vhs_pipeline.metadata import convert_all_ffmetadata_to_chapters_tsv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate metadata/*/chapters.tsv from chapters.ffmetadata files.",
    )
    parser.add_argument(
        "--metadata-root",
        default=str(METADATA_DIR),
        help="Metadata directory root (default: common.METADATA_DIR).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chapters.tsv files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    convert_all_ffmetadata_to_chapters_tsv(
        metadata_root=Path(args.metadata_root),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
