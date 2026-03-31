from __future__ import annotations


def allow_large_pillow_images(image_module=None):
    try:
        image = image_module
        if image is None:
            from PIL import Image as image  # pylint: disable=import-outside-toplevel
    except Exception:
        return None
    image.MAX_IMAGE_PIXELS = None
    return image
