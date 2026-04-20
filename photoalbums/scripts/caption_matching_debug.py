"""Diagnostic: run LM Studio caption matching on a single album page image.

Runs the standard Docling layout pipeline, generates the numbered overlay,
calls the configured LM Studio model for caption assignment, and prints the
merged result. Useful for inspecting caption quality on a specific page.

Usage:
  python -m photoalbums.scripts.caption_matching_debug --image path/to/image.jpg
  python -m photoalbums.scripts.caption_matching_debug --debug-image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEST_IMAGE = "Family_1980-1985_B08_P16_V.jpg"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"


def _find_image(name: str) -> Path | None:
    candidates = [Path(name), REPO_ROOT / name, *REPO_ROOT.rglob(name)]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _run_docling(image_path: Path):
    from photoalbums.lib._docling_pipeline import run_docling_pipeline  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(image_path) as img:
        img_w, img_h = img.size
    print(f"  Image size: {img_w}×{img_h}")
    print("  Running standard Docling pipeline...")
    result = run_docling_pipeline(
        image_path,
        img_w=img_w,
        img_h=img_h,
        preset="granite_docling",
        backend="auto_inline",
        device="cpu",
        retries=1,
    )
    return result.regions, img_w, img_h


def _print_regions(regions: list, label: str) -> None:
    print(f"\n  {label} — {len(regions)} region(s):")
    for i, r in enumerate(regions, start=1):
        hint = f" | caption_hint={r.caption_hint!r}" if r.caption_hint else ""
        print(f"    [{i}] x={r.x} y={r.y} w={r.width} h={r.height}{hint}")


def _write_debug_image(image_path: Path, regions: list) -> Path:
    from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel
    from photoalbums.lib.image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except OSError:
        font = ImageFont.load_default()
    for i, r in enumerate(regions, start=1):
        box = (r.x, r.y, r.x + r.width, r.y + r.height)
        draw.rectangle(box, outline="red", width=3)
        draw.text((r.x + 4, r.y + 4), str(i), fill="red", font=font)
    out_path = image_path.parent / f"{image_path.stem}_debug.png"
    img.save(out_path, format="PNG")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LM Studio caption matching for an album page image")
    parser.add_argument("--image", default=DEFAULT_TEST_IMAGE, help="Path to album page image")
    parser.add_argument("--lmstudio-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument("--model", default="", help="LM Studio model name (empty = use ai_models.toml default)")
    parser.add_argument("--debug-image", action="store_true", help="Write debug PNG with numbered bounding boxes")
    args = parser.parse_args()

    image_path = _find_image(args.image)
    if image_path is None:
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"\n=== Caption matching debug ===")
    print(f"Image: {image_path}")

    print("\n[1] Docling layout detection")
    try:
        regions, img_w, img_h = _run_docling(image_path)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1
    _print_regions(regions, "Raw output (model order)")

    from photoalbums.lib._caption_matching import call_lmstudio_caption_matching, assign_captions_from_lmstudio  # pylint: disable=import-outside-toplevel
    from photoalbums.lib.ai_model_settings import default_caption_matching_model  # pylint: disable=import-outside-toplevel
    from photoalbums.lib.ai_view_regions import _write_region_association_overlay_image  # pylint: disable=import-outside-toplevel

    overlay_path = _write_region_association_overlay_image(image_path, regions)
    print(f"\n[2] Region-association overlay: {overlay_path or '(write failed)'}")

    model = args.model or default_caption_matching_model()
    print(f"\n[3] LM Studio caption matching (url={args.lmstudio_url}, model={model or 'auto'})")
    captions = call_lmstudio_caption_matching(overlay_path or image_path, base_url=args.lmstudio_url, model=model, timeout=300.0)
    if not captions:
        print("  No captions returned (LM Studio offline or parse error)")
    else:
        print(f"  Response: {captions}")

    print("\n[4] Merged result")
    merged = assign_captions_from_lmstudio(regions, captions)
    for i, r in enumerate(merged, start=1):
        caption = r.caption_hint or "(no caption)"
        print(f"  region-{r.index + 1}: x={r.x} y={r.y} w={r.width} h={r.height} | {caption!r}")

    if args.debug_image:
        print("\n[5] Writing debug image...")
        try:
            out_path = _write_debug_image(image_path, regions)
            print(f"  Written: {out_path}")
        except Exception as exc:
            print(f"  ERROR writing debug image: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
