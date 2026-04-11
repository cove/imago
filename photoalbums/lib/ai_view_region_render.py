"""Debug rendering of detected photo regions over a view JPG.

Draws coloured labelled bounding boxes on a downscaled copy of the image.
Used for visual validation of region detection results before splitting.
"""

from __future__ import annotations

import io
from pathlib import Path

from .image_limits import allow_large_pillow_images

_MAX_EDGE = 1500

_PALETTE = [
    (255, 60, 60),  # red
    (60, 255, 60),  # lime
    (60, 220, 255),  # cyan
    (255, 230, 0),  # yellow
    (220, 60, 255),  # magenta
    (255, 140, 0),  # orange
    (255, 255, 255),  # white
    (0, 160, 255),  # deepskyblue
]


def render_regions_debug(
    image_path: str | Path,
    regions: list[dict],
    output_path: str | Path,
) -> bytes:
    """Draw region bounding boxes on a downscaled copy of the view image.

    regions is a list of dicts as returned by xmp_sidecar.read_region_list:
      {index, x, y, width, height, caption, ...} — pixel coords, top-left origin.

    Saves annotated JPEG to output_path and returns the JPEG bytes.
    The original image is not modified.
    """
    from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)

    path = Path(image_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(str(path)).convert("RGB")
    try:
        orig_w, orig_h = img.size
        longest = max(orig_w, orig_h)
        if longest > _MAX_EDGE:
            scale = _MAX_EDGE / longest
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            scale = 1.0

        draw_w, draw_h = img.size
        outline_width = max(2, draw_w // 400)

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=max(12, draw_w // 60))
        except TypeError:
            font = ImageFont.load_default()

        for reg in regions:
            idx = int(reg.get("index") or 0)
            colour = _PALETTE[idx % len(_PALETTE)]

            # Scale pixel coords to downscaled image
            rx = int(round(reg["x"] * scale))
            ry = int(round(reg["y"] * scale))
            rw = int(round(reg["width"] * scale))
            rh = int(round(reg["height"] * scale))

            # Clamp to image bounds
            x0 = max(0, min(draw_w - 1, rx))
            y0 = max(0, min(draw_h - 1, ry))
            x1 = max(x0, min(draw_w - 1, rx + rw))
            y1 = max(y0, min(draw_h - 1, ry + rh))

            draw.rectangle([x0, y0, x1, y1], outline=colour, width=outline_width)

            caption = str(reg.get("caption") or "").strip()
            label = f"#{idx + 1}"
            if caption:
                max_cap = 30
                label += f" {caption[:max_cap]}{'…' if len(caption) > max_cap else ''}"

            # Draw label background and text
            try:
                bbox = draw.textbbox((x0 + outline_width, y0 + outline_width), label, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 180))
            except AttributeError:
                bbox = None
            draw.text((x0 + outline_width, y0 + outline_width), label, fill=colour, font=font)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        jpeg_bytes = buf.getvalue()

        out_path.write_bytes(jpeg_bytes)
        return jpeg_bytes
    finally:
        img.close()
