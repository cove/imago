from __future__ import annotations

import argparse
import os
from pathlib import Path

from .ai_model_settings import default_lmstudio_base_url, default_ocr_model
from ..common import PHOTO_ALBUMS_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CREATOR_TOOL = "https://github.com/cove/imago"
DEFAULT_CAST_STORE = Path(__file__).resolve().parents[2] / "cast" / "data"


def _explicit_cli_flags(argv: list[str] | None) -> set[str]:
    flags: set[str] = set()
    for item in list(argv or []):
        text = str(item or "")
        if not text.startswith("--"):
            continue
        flags.add(text.split("=", 1)[0])
    return flags


def _resolve_caption_prompt(prompt_text: str, prompt_file: str) -> str:
    file_text = str(prompt_file or "").strip()
    if file_text:
        path = Path(file_text).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise SystemExit(f"Caption prompt file does not exist: {path}") from exc
        except OSError as exc:
            raise SystemExit(f"Could not read caption prompt file {path}: {exc}") from exc
    return str(prompt_text or "").strip()


def _absolute_cli_path(path_text: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path_text).expanduser())))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index photo album images with cast people matching, YOLO objects, OCR, and XMP sidecars.",
    )
    parser.add_argument(
        "--photos-root",
        default=str(PHOTO_ALBUMS_DIR),
        help="Photo Albums root directory.",
    )
    parser.add_argument("--cast-store", default=str(DEFAULT_CAST_STORE), help="Cast store directory.")
    parser.add_argument("--creator-tool", default=DEFAULT_CREATOR_TOOL, help="XMP CreatorTool value.")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument(
        "--object-threshold",
        type=float,
        default=0.30,
        help="Object detection confidence.",
    )
    parser.add_argument(
        "--people-threshold",
        type=float,
        default=0.72,
        help="Face similarity threshold.",
    )
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size in pixels.")
    parser.add_argument(
        "--ocr-engine",
        choices=["none", "local", "lmstudio"],
        default="none",
        help="OCR backend.",
    )
    parser.add_argument(
        "--ocr-model",
        default=default_ocr_model(),
        help="Optional model id/path used by the selected OCR engine.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="OCR language.")
    parser.add_argument(
        "--caption-engine",
        choices=["none", "lmstudio"],
        default="lmstudio",
        help="Caption backend for XMP description.",
    )
    parser.add_argument(
        "--caption-model",
        default="",
        help="Optional model id/path used by the selected caption engine.",
    )
    parser.add_argument(
        "--caption-prompt",
        dest="caption_prompt",
        default="",
        help="Exact prompt text for model captioning. When set, built-in prompt hints are disabled.",
    )
    parser.add_argument(
        "--caption-prompt-file",
        dest="caption_prompt_file",
        default="",
        help="Read exact model caption prompt text from a file. Overrides --caption-prompt when set.",
    )
    parser.add_argument(
        "--local-prompt",
        dest="caption_prompt",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-prompt-file",
        dest="caption_prompt_file",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-prompt",
        dest="caption_prompt",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-prompt-file",
        dest="caption_prompt_file",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default=default_lmstudio_base_url(),
        help="Base URL for the LM Studio OpenAI-compatible API.",
    )
    parser.add_argument(
        "--caption-max-tokens",
        type=int,
        default=96,
        help="Max new tokens for caption models.",
    )
    parser.add_argument(
        "--caption-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for local captioning.",
    )
    parser.add_argument(
        "--caption-max-edge",
        type=int,
        default=0,
        help="Optional long-edge cap, in pixels, applied only during caption generation.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional processing limit.")
    parser.add_argument(
        "--photo",
        default="",
        help="Process a single photo file. Bypasses discovery and implies --force.",
    )
    parser.add_argument(
        "--album",
        default="",
        help="Filter to photos whose parent directory name contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--photo-offset",
        type=int,
        default=0,
        help="Skip first N discovered images. Use with --max-images to process a range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore manifest and process all files. Equivalent to --reprocess-mode=all.",
    )
    parser.add_argument(
        "--reprocess-mode",
        default="unprocessed",
        choices=["unprocessed", "new_only", "errors_only", "outdated", "cast_changed", "gps", "all"],
        help=(
            "Controls which images are processed. "
            "'unprocessed' (default): images with missing or stale sidecar. "
            "'new_only': only images with no manifest entry (never indexed). "
            "'errors_only': only images whose sidecar contains a processing error. "
            "'outdated': only images where the sidecar is older than the image file. "
            "'cast_changed': only images needing people re-detection when the cast store changes. "
            "'gps': re-run only the GPS location estimate step for already-indexed images. "
            "'all': force reprocess everything (same as --force)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write sidecar/manifest.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated caption text to stdout only. Implies --dry-run and forced reprocessing.",
    )
    parser.add_argument(
        "--include-view",
        action="store_true",
        help="Include files in rendered *_Pages and *_Photos folders.",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include files in *_Archive folders.",
    )
    parser.add_argument("--disable-people", action="store_true", help="Disable cast people matching.")
    parser.add_argument("--disable-objects", action="store_true", help="Disable object detection.")
    parser.add_argument(
        "--ignore-render-settings",
        action="store_true",
        help="Ignore per-archive render_settings.json overrides.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--stitch-scans",
        action="store_true",
        help=(
            "Deprecated. Multi-scan archive page OCR now uses a temporary stitched composite during normal processing."
        ),
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help="Comma-separated file extensions to include.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split discovered files into N deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process when --shard-count is greater than 1.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help=(
            "Comma-separated list of step names to force re-run unconditionally "
            "(e.g. 'caption', 'ocr,people'). Downstream steps are also marked stale."
        ),
    )
    return parser.parse_args(argv)
