from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def allow_large_pillow_images(image_module=None) -> Any | None:
    try:
        image = image_module
        if image is None:
            from PIL import Image as image  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        log.debug("PIL not available: %s", exc)
        return None
    image.MAX_IMAGE_PIXELS = None
    return image


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    with Image.open(str(image_path)) as img:
        return img.size


try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except Exception as exc:
    log.debug("PIL not available at module load: %s", exc)
