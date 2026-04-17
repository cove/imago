"""Photo restoration using RealRestorer diffusion pipeline.

Optional dependency: requires RealRestorer installed from
https://github.com/yfyang007/RealRestorer.git
If not installed, restore_photo() returns the original image unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger(__name__)

_UNAVAILABLE = object()
_pipeline = None

# Tunable: adjust prompt to match the degradation type of the photos being processed.
_RESTORE_PROMPT = "Please restore this low-quality image, recovering its normal brightness and clarity."


def _get_pipeline():
    global _pipeline
    if _pipeline is _UNAVAILABLE:
        return None
    if _pipeline is not None:
        return _pipeline

    try:
        from diffusers import RealRestorerPipeline
        import torch
    except ImportError:
        log.warning(
            "RealRestorer is not installed; photo restoration will be skipped. "
            "See https://github.com/yfyang007/RealRestorer.git for install instructions."
        )
        _pipeline = _UNAVAILABLE
        return None

    try:
        pipe = RealRestorerPipeline.from_pretrained(
            "RealRestorer/RealRestorer",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    except Exception as exc:
        log.warning("RealRestorer pipeline setup failed: %s", exc)
        _pipeline = _UNAVAILABLE
        return None

    _pipeline = pipe
    return _pipeline


def restore_photo(image: "PILImage.Image") -> "PILImage.Image":
    """Restore a cropped photo using RealRestorer.

    Accepts an in-memory PIL Image and returns a restored PIL Image.
    Returns the original image unchanged if RealRestorer is unavailable
    or if inference fails.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return image

    try:
        result = pipe(
            image=image,
            prompt=_RESTORE_PROMPT,
            num_inference_steps=28,
            guidance_scale=3.0,
            seed=42,
            size_level=1024,
        )
        return result.images[0]
    except Exception as exc:
        log.warning("RealRestorer inference failed: %s", exc)
        return image
