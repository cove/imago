"""Photo restoration using RealRestorer diffusion pipeline.

Optional dependency: requires RealRestorer installed from
https://github.com/yfyang007/RealRestorer.git
If not installed, restore_photo() returns the original image unchanged.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger(__name__)

_UNAVAILABLE = object()
_pipeline = None
REAL_RESTORER_MODEL_NAME = "RealRestorer/RealRestorer"
REAL_RESTORER_REPO_BYTES = 41_800_000_000

RESTORE_RESULT_RESTORED = "restored"
RESTORE_RESULT_UNAVAILABLE = "unavailable"
RESTORE_RESULT_FAILED = "failed"

# Tunable: adjust prompt to match the degradation type of the photos being processed.
_RESTORE_PROMPT = "Please restore this low-quality image, recovering its normal brightness and clarity."


def _resolve_runtime(torch) -> tuple[object, str]:
    if bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
        return torch.bfloat16, "cuda"
    if bool(getattr(getattr(getattr(torch, "backends", None), "mps", None), "is_available", lambda: False)()):
        return torch.float32, "mps"
    return torch.float32, "cpu"


def _installed_ram_bytes() -> int | None:
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys)

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(page_count)


def _format_bytes_gb(value: int) -> str:
    return f"{value / 1_000_000_000:.1f} GB"


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

    installed_ram = _installed_ram_bytes()
    if installed_ram is not None and REAL_RESTORER_REPO_BYTES >= installed_ram:
        log.warning(
            "RealRestorer pipeline setup refused: model repo size %s is greater than or equal to installed RAM %s",
            _format_bytes_gb(REAL_RESTORER_REPO_BYTES),
            _format_bytes_gb(installed_ram),
        )
        _pipeline = _UNAVAILABLE
        return None

    try:
        torch_dtype, device = _resolve_runtime(torch)
        pipe = RealRestorerPipeline.from_pretrained(
            REAL_RESTORER_MODEL_NAME,
            torch_dtype=torch_dtype,
        )
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    except Exception as exc:
        log.warning("RealRestorer pipeline setup failed: %s", exc)
        _pipeline = _UNAVAILABLE
        return None

    _pipeline = pipe
    return _pipeline


def restore_photo_with_result(image: "PILImage.Image") -> tuple["PILImage.Image", str]:
    """Restore a cropped photo using RealRestorer and report the outcome.

    Returns ``(image, result)`` where ``result`` is one of:
    - ``restored`` when inference succeeded
    - ``unavailable`` when RealRestorer is unavailable
    - ``failed`` when inference failed and the original image is preserved
    """
    pipe = _get_pipeline()
    if pipe is None:
        return image, RESTORE_RESULT_UNAVAILABLE

    try:
        result = pipe(
            image=image,
            prompt=_RESTORE_PROMPT,
            num_inference_steps=28,
            guidance_scale=3.0,
            seed=42,
            size_level=1024,
        )
        return result.images[0], RESTORE_RESULT_RESTORED
    except Exception as exc:
        log.warning("RealRestorer inference failed: %s", exc)
        return image, RESTORE_RESULT_FAILED


def restore_photo(image: "PILImage.Image") -> "PILImage.Image":
    """Restore a cropped photo using RealRestorer.

    Accepts an in-memory PIL Image and returns a restored PIL Image.
    Returns the original image unchanged if RealRestorer is unavailable
    or if inference fails.
    """
    restored_image, _ = restore_photo_with_result(image)
    return restored_image
