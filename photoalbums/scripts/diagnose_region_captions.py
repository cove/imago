"""Diagnostic for the empty-mwg-rs:Name issue.

Usage:
    python -m photoalbums.scripts.diagnose_region_captions <path-to-page-view.jpg>

Reads the XMP next to the page view and the embedded ai-index detections,
then prints a summary of:
  - the regions written to the XMP (Name, PhotoNumber, CaptionHint)
  - the metadata step's pipeline state (input_hash, result, timestamp)
  - the photo_captions stored in the detections payload
  - whether the association-overlay debug image exists at the expected path

The output should make it obvious whether the metadata step ran, whether it
returned per-photo captions, and whether _update_region_captions_from_metadata
had the inputs it needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from photoalbums.lib.ai_view_regions import _region_association_overlay_path
from photoalbums.lib.xmp_sidecar import (
    read_ai_sidecar_state,
    read_pipeline_step,
    read_region_list,
)


def _img_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image
    from photoalbums.lib.image_limits import allow_large_pillow_images

    allow_large_pillow_images(Image)
    with Image.open(image_path) as img:
        return img.width, img.height


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    image_path = Path(argv[1]).resolve()
    if not image_path.is_file():
        print(f"image not found: {image_path}")
        return 1

    xmp_path = image_path.with_suffix(".xmp")
    print(f"image: {image_path}")
    print(f"xmp:   {xmp_path}")
    print(f"xmp exists: {xmp_path.is_file()}")
    if not xmp_path.is_file():
        return 1

    img_w, img_h = _img_dims(image_path)
    print(f"image dims: {img_w}x{img_h}")

    # 1. Regions
    regions = read_region_list(xmp_path, img_w, img_h)
    print()
    print(f"-- regions ({len(regions)}) --")
    for r in regions:
        print(
            f"  idx={r['index']} "
            f"Name={r.get('caption')!r} "
            f"PhotoNumber={r.get('photo_number')} "
            f"CaptionHint={r.get('caption_hint')!r} "
            f"PersonNames={r.get('person_names')}"
        )

    # 2. Pipeline state for the metadata step
    state = read_pipeline_step(xmp_path, "metadata")
    print()
    print("-- ai-index/metadata pipeline state --")
    print(json.dumps(state, indent=2, default=str) if state else "  (none)")

    # 3. Detections payload (should contain photo_captions if step is fresh)
    sidecar_state = read_ai_sidecar_state(xmp_path) or {}
    detections = dict(sidecar_state.get("detections") or {})
    photo_captions = detections.get("photo_captions")
    print()
    print("-- detections.photo_captions --")
    print(
        json.dumps(photo_captions, indent=2, default=str)
        if photo_captions is not None
        else "  (key missing - the cached metadata step output predates the fix)"
    )

    caption_meta = detections.get("caption") or {}
    print()
    print("-- detections.caption --")
    print(json.dumps(caption_meta, indent=2, default=str))

    # 4. Overlay path
    overlay_path = _region_association_overlay_path(image_path)
    print()
    print(f"-- association-overlay --")
    print(f"  expected: {overlay_path}")
    print(f"  exists:   {overlay_path.is_file()}")
    if not overlay_path.is_file():
        debug_dir = overlay_path.parent
        if debug_dir.is_dir():
            siblings = [p.name for p in debug_dir.iterdir() if "association-overlay" in p.name]
            print(f"  siblings in {debug_dir}: {siblings}")
        else:
            print(f"  debug dir {debug_dir} does not exist")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
