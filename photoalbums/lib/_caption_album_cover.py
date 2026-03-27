from __future__ import annotations

from pathlib import Path

from ._caption_album_inference import infer_album_context
from ._caption_album_types import AlbumContext
from ._caption_text import clean_text


def _looks_like_uniform_cover_color(image_path: str | Path) -> bool:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
    except Exception:
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return False

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > 512:
        scale = 512.0 / float(longest)
        resized = cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    pixels = resized.reshape(-1, 3).astype("float32")
    if pixels.size == 0:
        return False

    mean = pixels.mean(axis=0)
    delta = np.abs(pixels - mean)
    uniform_ratio = float((delta.max(axis=1) <= 45.0).mean())
    blue_dominant = bool(mean[0] >= 95.0 and mean[0] >= mean[1] + 18.0 and mean[0] >= mean[2] + 18.0)
    white_dominant = bool(float(mean.min()) >= 170.0)
    return uniform_ratio >= 0.72 and (blue_dominant or white_dominant)


def looks_like_album_cover(
    image_path: str | Path,
    *,
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> bool:
    text = clean_text(ocr_text)
    if not text:
        return False
    context = album_context or infer_album_context(image_path=image_path, ocr_text=text, allow_ocr=True)
    if not context.kind:
        return False
    return _looks_like_uniform_cover_color(image_path)
