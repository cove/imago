from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..naming import BASE_PAGE_NAME_RE, DERIVED_NAME_RE, SCAN_NAME_RE

PAGE_SPLIT_MODES = {"auto", "off"}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_MIN_REGION_AREA_RATIO = 0.015
_MAX_REGION_COUNT = 12
_MAX_WHOLE_PAGE_RATIO = 0.92


@dataclass(frozen=True)
class LayoutBounds:
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class PreparedSubPhoto:
    index: int
    bounds: LayoutBounds
    path: Path


@dataclass
class PreparedImageLayout:
    kind: str
    split_mode: str
    content_bounds: LayoutBounds
    content_path: Path
    original_path: Path
    page_like: bool
    footer_trimmed: bool
    split_applied: bool
    fallback_used: bool
    subphotos: list[PreparedSubPhoto]


def normalize_page_split_mode(value: object, default: str = "off") -> str:
    text = str(value or "").strip().lower()
    if text in PAGE_SPLIT_MODES:
        return text
    fallback = str(default or "off").strip().lower()
    if fallback in PAGE_SPLIT_MODES:
        return fallback
    return "off"


def classify_image_kind(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    name = path.name
    stem = path.stem
    parent_names = {parent.name for parent in path.parents}
    in_view = any(name.endswith("_View") for name in parent_names)

    if suffix not in _IMAGE_EXTENSIONS:
        return "single_image"
    if DERIVED_NAME_RE.search(name):
        return "detail"
    if suffix in {".tif", ".tiff"} and SCAN_NAME_RE.search(name):
        return "page_scan"
    if "_stitched" in stem.lower():
        return "page_view"
    if in_view and BASE_PAGE_NAME_RE.fullmatch(stem):
        return "page_view"
    return "single_image"


def is_page_like_kind(kind: str) -> bool:
    return kind in {"page_scan", "page_view"}


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency optional in tests
        raise RuntimeError(
            "opencv-python is required for page layout analysis."
        ) from exc
    return cv2


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency optional in tests
        raise RuntimeError("numpy is required for page layout analysis.") from exc
    return np


def _load_image_bgr(image_path: Path):
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(axis=2), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _write_png(path: Path, image) -> Path:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write temp PNG: {path}")
    return path


def _detect_footer_bounds(image) -> tuple[LayoutBounds, bool]:
    np = _require_numpy()
    cv2 = _require_cv2()

    height, width = image.shape[:2]
    if height < 200 or width < 200:
        return LayoutBounds(0, 0, width, height), False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = (gray > 24).mean(axis=1)
    window = max(9, min(31, height // 45))
    kernel = np.ones(window, dtype=float) / float(window)
    smooth = np.convolve(signal, kernel, mode="same")

    min_trim = max(20, int(height * 0.05))
    max_trim = max(min_trim, int(height * 0.35))
    search_start = int(height * 0.55)
    content_bottom = height

    for row in range(search_start, height - min_trim):
        footer_mean = float(smooth[row:].mean()) if row < len(smooth) else 1.0
        above_start = max(0, row - max(10, window * 2))
        above_mean = float(smooth[above_start:row].mean()) if row > above_start else 0.0
        trimmed = height - row
        if trimmed < min_trim or trimmed > max_trim:
            continue
        if above_mean >= 0.16 and footer_mean <= 0.10:
            footer = gray[row:, :]
            if footer.size and float(footer.mean()) <= 45.0:
                content_bottom = row
                break

    if content_bottom >= height:
        return LayoutBounds(0, 0, width, height), False
    return LayoutBounds(0, 0, width, content_bottom), True


def _bounds_area(bounds: LayoutBounds) -> int:
    return max(0, int(bounds.width)) * max(0, int(bounds.height))


def _intersects_or_touches(a: LayoutBounds, b: LayoutBounds, gap: int = 12) -> bool:
    ax1 = a.x + a.width
    ay1 = a.y + a.height
    bx1 = b.x + b.width
    by1 = b.y + b.height
    return not (
        ax1 + gap < b.x or bx1 + gap < a.x or ay1 + gap < b.y or by1 + gap < a.y
    )


def _contains(a: LayoutBounds, b: LayoutBounds, padding: int = 10) -> bool:
    return (
        a.x - padding <= b.x
        and a.y - padding <= b.y
        and a.x + a.width + padding >= b.x + b.width
        and a.y + a.height + padding >= b.y + b.height
    )


def _merge_boxes(boxes: list[LayoutBounds]) -> list[LayoutBounds]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[LayoutBounds] = []
        while merged:
            current = merged.pop(0)
            combined = current
            kept: list[LayoutBounds] = []
            for other in merged:
                if (
                    _intersects_or_touches(combined, other)
                    or _contains(combined, other)
                    or _contains(other, combined)
                ):
                    x0 = min(combined.x, other.x)
                    y0 = min(combined.y, other.y)
                    x1 = max(combined.x + combined.width, other.x + other.width)
                    y1 = max(combined.y + combined.height, other.y + other.height)
                    combined = LayoutBounds(x0, y0, x1 - x0, y1 - y0)
                    changed = True
                else:
                    kept.append(other)
            next_boxes.append(combined)
            merged = kept
        merged = next_boxes
    return merged


def _sort_boxes(boxes: list[LayoutBounds]) -> list[LayoutBounds]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda item: (item.y, item.x))
    median_height = sorted(item.height for item in sorted_boxes)[len(sorted_boxes) // 2]
    row_tolerance = max(20, int(median_height * 0.45))
    rows: list[list[LayoutBounds]] = []
    anchors: list[int] = []
    for box in sorted_boxes:
        placed = False
        for idx, anchor in enumerate(anchors):
            if abs(box.y - anchor) <= row_tolerance:
                rows[idx].append(box)
                anchors[idx] = int(
                    round(sum(item.y for item in rows[idx]) / len(rows[idx]))
                )
                placed = True
                break
        if not placed:
            rows.append([box])
            anchors.append(box.y)
    out: list[LayoutBounds] = []
    for row in rows:
        row.sort(key=lambda item: item.x)
        out.extend(row)
    return out


def _detect_photo_regions(image, content_bounds: LayoutBounds) -> list[LayoutBounds]:
    cv2 = _require_cv2()

    x0 = int(content_bounds.x)
    y0 = int(content_bounds.y)
    x1 = x0 + int(content_bounds.width)
    y1 = y0 + int(content_bounds.height)
    content = image[y0:y1, x0:x1]
    if content.size == 0:
        return []

    height, width = content.shape[:2]
    content_area = max(1, height * width)
    min_area = max(2500, int(content_area * _MIN_REGION_AREA_RATIO))
    min_width = max(60, int(width * 0.10))
    min_height = max(60, int(height * 0.10))

    gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(edges, thresh)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.dilate(mask, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[LayoutBounds] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw < min_width or rh < min_height:
            continue
        box_area = rw * rh
        if box_area >= int(content_area * _MAX_WHOLE_PAGE_RATIO):
            continue
        fill_ratio = area / float(max(1, box_area))
        if fill_ratio < 0.35:
            continue
        aspect_ratio = rw / float(max(1, rh))
        if aspect_ratio < 0.20 or aspect_ratio > 5.0:
            continue
        boxes.append(LayoutBounds(x0 + rx, y0 + ry, rw, rh))

    boxes = _merge_boxes(boxes)
    boxes = [
        box
        for box in boxes
        if _bounds_area(box) < int(_bounds_area(content_bounds) * _MAX_WHOLE_PAGE_RATIO)
    ]
    boxes = _sort_boxes(boxes)
    return boxes[:_MAX_REGION_COUNT]


@contextmanager
def prepare_image_layout(
    image_path: str | Path,
    *,
    split_mode: str = "auto",
) -> Iterator[PreparedImageLayout]:
    path = Path(image_path)
    kind = classify_image_kind(path)
    page_like = is_page_like_kind(kind)
    normalized_mode = normalize_page_split_mode(split_mode)

    if not page_like:
        yield PreparedImageLayout(
            kind=kind,
            split_mode="off",
            content_bounds=LayoutBounds(0, 0, 0, 0),
            content_path=path,
            original_path=path,
            page_like=False,
            footer_trimmed=False,
            split_applied=False,
            fallback_used=False,
            subphotos=[],
        )
        return

    image = _load_image_bgr(path)
    height, width = image.shape[:2]
    content_bounds = LayoutBounds(0, 0, width, height)
    footer_trimmed = False
    if kind == "page_view":
        content_bounds, footer_trimmed = _detect_footer_bounds(image)

    with tempfile.TemporaryDirectory(prefix="imago-page-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        content = image[
            content_bounds.y : content_bounds.y + content_bounds.height,
            content_bounds.x : content_bounds.x + content_bounds.width,
        ]
        content_path = _write_png(tmp_dir / "content.png", content)

        subphotos: list[PreparedSubPhoto] = []
        split_applied = False
        fallback_used = False

        if normalized_mode == "auto":
            for idx, bounds in enumerate(
                _detect_photo_regions(image, content_bounds), 1
            ):
                crop = image[
                    bounds.y : bounds.y + bounds.height,
                    bounds.x : bounds.x + bounds.width,
                ]
                if crop.size == 0:
                    continue
                crop_path = _write_png(tmp_dir / f"photo_{idx:02d}.png", crop)
                subphotos.append(
                    PreparedSubPhoto(index=idx, bounds=bounds, path=crop_path)
                )
            split_applied = bool(subphotos)
            if not subphotos:
                fallback_used = True
                fallback_path = content_path if kind == "page_view" else path
                subphotos = [
                    PreparedSubPhoto(
                        index=1,
                        bounds=content_bounds,
                        path=fallback_path,
                    )
                ]

        yield PreparedImageLayout(
            kind=kind,
            split_mode=normalized_mode,
            content_bounds=content_bounds,
            content_path=content_path,
            original_path=path,
            page_like=True,
            footer_trimmed=footer_trimmed,
            split_applied=split_applied,
            fallback_used=fallback_used,
            subphotos=subphotos,
        )
