from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_REMBG_REMOVE = None
_REMBG_NEW_SESSION = None
_REMBG_IMPORT_ERROR = None
_REMBG_SESSION = None


def _load_rembg() -> tuple[Any, Any]:
    global _REMBG_REMOVE, _REMBG_NEW_SESSION, _REMBG_IMPORT_ERROR
    if _REMBG_REMOVE is not None and _REMBG_NEW_SESSION is not None:
        return _REMBG_REMOVE, _REMBG_NEW_SESSION
    if _REMBG_IMPORT_ERROR is not None:
        raise RuntimeError(_REMBG_IMPORT_ERROR)
    try:
        from rembg import new_session, remove  # type: ignore[import]
    except Exception as exc:
        _REMBG_IMPORT_ERROR = f"Unable to import rembg: {type(exc).__name__}: {exc}"
        raise RuntimeError(_REMBG_IMPORT_ERROR) from exc
    _REMBG_REMOVE = remove
    _REMBG_NEW_SESSION = new_session
    return _REMBG_REMOVE, _REMBG_NEW_SESSION


def _load_onnxruntime() -> Any | None:
    try:
        import onnxruntime as ort  # type: ignore[import]
    except Exception:
        return None
    return ort


def _rembg_providers() -> list[str]:
    ort = _load_onnxruntime()
    if ort is None:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_rembg_session() -> Any:
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    _remove, new_session = _load_rembg()
    _REMBG_SESSION = new_session(providers=_rembg_providers())
    return _REMBG_SESSION


def _coerce_removed_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        from PIL import Image as PILImage  # type: ignore[import]
    except Exception:
        PILImage = None
    if PILImage is not None and isinstance(value, PILImage.Image):
        return np.asarray(value)
    raise RuntimeError("rembg returned an unsupported image type.")


def build_rembg_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("Cannot preprocess an empty image.")
    remove, _new_session = _load_rembg()
    session = _get_rembg_session()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    removed = _coerce_removed_array(remove(image_rgb, session=session))
    if removed.ndim != 3:
        raise RuntimeError("rembg returned an invalid image shape.")
    if removed.shape[:2] != image_rgb.shape[:2]:
        raise RuntimeError("rembg changed the image dimensions unexpectedly.")
    if removed.shape[2] == 4:
        foreground = removed[:, :, :3].astype(np.float32)
        alpha = removed[:, :, 3:4].astype(np.float32) / 255.0
        background = np.full_like(foreground, 255.0)
        composited = (foreground * alpha) + (background * (1.0 - alpha))
        removed_rgb = np.clip(composited, 0.0, 255.0).astype(np.uint8)
    elif removed.shape[2] == 3:
        removed_rgb = removed.astype(np.uint8)
    else:
        raise RuntimeError("rembg returned an unsupported channel count.")
    return cv2.cvtColor(removed_rgb, cv2.COLOR_RGB2BGR)
