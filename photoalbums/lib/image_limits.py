from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def allow_large_pillow_images(image_module=None):
    try:
        image = image_module
        if image is None:
            from PIL import Image as image  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        log.debug("PIL not available: %s", exc)
        return None
    image.MAX_IMAGE_PIXELS = None
    return image


try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except Exception as exc:
    log.debug("PIL not available at module load: %s", exc)
