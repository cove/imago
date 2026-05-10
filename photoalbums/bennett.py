"""Bennett photo scanning pipeline.

Watches the Bennett Photos directory for incoming_scan.tif, runs Docling to
detect individual photos, and saves each cropped region as Bennett_####_a.jpg.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

BENNETT_PHOTOS_DIR_ENV = "BENNETT_PHOTOS_DIR"
BENNETT_PHOTOS_SUBPATH = Path("Bennett Photos")
INCOMING_NAME = "incoming_scan.tif"
_BENNETT_FILE_RE = re.compile(r"^Bennett_(\d{4})_a\.jpg$", re.IGNORECASE)
_POLL_INTERVAL_SECONDS = 2.0
_STABILITY_WAIT_SECONDS = 2.0


def get_bennett_dir() -> Path:
    configured = str(os.environ.get(BENNETT_PHOTOS_DIR_ENV, "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    from .common import _onedrive_roots
    for root in _onedrive_roots(Path.home()):
        candidate = root / BENNETT_PHOTOS_SUBPATH
        if candidate.exists():
            return candidate
    return Path.home() / "OneDrive" / BENNETT_PHOTOS_SUBPATH


def _next_bennett_number(output_dir: Path) -> int:
    highest = 0
    if not output_dir.is_dir():
        return 1
    for entry in output_dir.iterdir():
        if not entry.is_file():
            continue
        m = _BENNETT_FILE_RE.match(entry.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def _open_tif_rgb(tif_path: Path):
    from PIL import Image

    from .lib.image_limits import allow_large_pillow_images
    allow_large_pillow_images(Image)
    img = Image.open(str(tif_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _run_docling_on_scan(tif_path: Path, img_w: int, img_h: int) -> list:
    from .lib._docling_pipeline import run_docling_pipeline
    from .lib.ai_model_settings import (
        default_docling_backend,
        default_docling_device,
        default_docling_preset,
        default_docling_retries,
    )
    result = run_docling_pipeline(
        tif_path,
        img_w,
        img_h,
        default_docling_preset(),
        backend=default_docling_backend(),
        device=default_docling_device(),
        retries=default_docling_retries(),
    )
    return result.regions


def _crop_region(image, region):
    img_w, img_h = image.size
    left = max(0, region.x)
    top = max(0, region.y)
    right = min(img_w, region.x + region.width)
    bottom = min(img_h, region.y + region.height)
    return image.crop((left, top, right, bottom))


def process_scan(tif_path: Path, output_dir: Path) -> list[Path]:
    print(f"Opening {tif_path.name}...")
    image = _open_tif_rgb(tif_path)
    img_w, img_h = image.size
    print(f"  Image size: {img_w}x{img_h}")

    print("  Running Docling region detection...")
    regions = _run_docling_on_scan(tif_path, img_w, img_h)
    if not regions:
        print("  No photo regions detected.")
        return []

    print(f"  Detected {len(regions)} region(s).")
    next_number = _next_bennett_number(output_dir)
    written: list[Path] = []
    for i, region in enumerate(regions):
        crop = _crop_region(image, region)
        w, h = crop.size
        if w < 1 or h < 1:
            print(f"  Skipping zero-size region {i + 1}.")
            continue
        number = next_number + len(written)
        output_path = output_dir / f"Bennett_{number:04d}_a.jpg"
        crop.save(str(output_path), "JPEG", quality=95)
        print(f"  Wrote {output_path.name}")
        written.append(output_path)

    return written


def _is_file_stable(path: Path) -> bool:
    try:
        size_before = path.stat().st_size
        time.sleep(_STABILITY_WAIT_SECONDS)
        return path.is_file() and path.stat().st_size == size_before
    except FileNotFoundError:
        return False


def watch(*, output_dir: Path | None = None) -> int:
    if output_dir is None:
        output_dir = get_bennett_dir()
    incoming = output_dir / INCOMING_NAME
    print(f"Watching {output_dir} for {INCOMING_NAME}...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            if incoming.is_file() and _is_file_stable(incoming):
                _process_incoming_scan(incoming, output_dir)
            else:
                time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def _process_incoming_scan(incoming: Path, output_dir: Path) -> None:
    try:
        written = process_scan(incoming, output_dir)
        incoming.unlink()
        print(f"Done. Wrote {len(written)} crop(s). Removed {INCOMING_NAME}.")
    except Exception as exc:
        print(f"ERROR processing {INCOMING_NAME} failed due to: {exc}", file=sys.stderr)


def _run_process_command(output_dir: Path) -> int:
    incoming = output_dir / INCOMING_NAME
    if not incoming.is_file():
        print(f"No {INCOMING_NAME} found in {output_dir}", file=sys.stderr)
        return 1
    try:
        written = process_scan(incoming, output_dir)
        incoming.unlink()
        print(f"Done. Wrote {len(written)} crop(s).")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m photoalbums.bennett",
        description="Bennett photo scanning pipeline.",
    )
    parser.add_argument("--output-dir", default=None, metavar="DIR", help="Override output directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("watch", help=f"Watch for {INCOMING_NAME} and process when it appears")
    subparsers.add_parser("process", help=f"Process {INCOMING_NAME} once (one-shot)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else get_bennett_dir()

    if args.command == "watch":
        return watch(output_dir=output_dir)
    if args.command == "process":
        return _run_process_command(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
