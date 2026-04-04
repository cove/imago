from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from .ai_album_titles import _scan_name_match
from .ai_ocr import extract_keywords
from .ai_page_layout import PreparedImageLayout
from .ai_render_settings import find_archive_dir_for_image
from .xmp_sidecar import _dedupe
from ..naming import SCAN_TIFF_RE, parse_album_filename

from .ai_index_analysis import (
    ArchiveScanOCRAuthority,
    ImageAnalysis,
    _get_image_dimensions,
    _prepare_ai_model_image,
)


def _hash_text(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()


def _scan_page_key(image_path: Path) -> str | None:
    """Return a page-level grouping key for _S# scan files (same P##, different S##).

    Returns None for files that don't match the scan naming pattern.
    """
    match = _scan_name_match(image_path)
    if not match:
        return None
    return (
        f"{match.group('collection')}_{match.group('year')}_B{match.group('book')}_P{match.group('page')}"
    ).casefold()


def _scan_number(image_path: Path) -> int:
    """Return the S## scan number for ordering within a page group."""
    match = _scan_name_match(image_path)
    if not match:
        return 0
    try:
        return int(match.group("scan"))
    except (ValueError, IndexError):
        return 0


def _scan_group_paths(image_path: Path) -> list[Path]:
    page_key = _scan_page_key(image_path)
    if page_key is None:
        return [image_path]
    group_paths = [
        path
        for path in image_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and _scan_page_key(path) == page_key
    ]
    group_paths.sort(key=_scan_number)
    return group_paths or [image_path]


def _scan_group_signature(group_paths: list[Path]) -> str:
    parts: list[str] = []
    for path in group_paths:
        stat = path.stat()
        parts.append(f"{path.name}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}")
    return _hash_text("|".join(parts))


def _resolve_archive_scan_authoritative_ocr(
    *,
    image_path: Path,
    group_paths: list[Path],
    group_signature: str,
    cache: dict[str, ArchiveScanOCRAuthority],
    ocr_engine=None,
    step_fn=None,
    stitched_image_dir: Path | None = None,
    debug_recorder=None,
    debug_step: str = "ocr_authority",
) -> ArchiveScanOCRAuthority:
    page_key = _scan_page_key(image_path)
    if page_key is None or len(group_paths) < 2:
        raise RuntimeError(f"Authoritative stitched OCR requires a multi-scan archive page: {image_path}")
    cached = cache.get(page_key)
    if cached is not None and cached.signature == group_signature and (ocr_engine is None or bool(cached.ocr_hash)):
        return cached

    from ..stitch_oversized_pages import (  # pylint: disable=import-outside-toplevel
        build_stitched_image,
        get_view_dirname,
    )

    collection, year, book, page = parse_album_filename(image_path.name)
    view_jpg: Path | None = None
    if collection != "Unknown":
        view_dir = Path(get_view_dirname(image_path.parent))
        candidate = view_dir / f"{collection}_{year}_B{book}_P{int(page):02d}_V.jpg"
        if candidate.is_file():
            view_jpg = candidate

    def _run_authoritative_ocr(source_path: Path) -> tuple[str, tuple[str, ...], str]:
        if ocr_engine is None or ocr_engine.engine == "none":
            return "", (), ""
        if step_fn:
            step_fn("ocr")
        with _prepare_ai_model_image(source_path) as model_image_path:
            ocr_text = ocr_engine.read_text(
                model_image_path,
                debug_recorder=debug_recorder,
                debug_step=debug_step,
            )
        return ocr_text, tuple(extract_keywords(ocr_text, max_keywords=15)), _hash_text(ocr_text)

    stitched_cap_path: Path | None = view_jpg
    ocr_text = ""
    ocr_keywords: tuple[str, ...] = ()
    ocr_hash = ""

    if view_jpg is not None:
        ocr_text, ocr_keywords, ocr_hash = _run_authoritative_ocr(view_jpg)

    if view_jpg is None:
        if step_fn:
            step_fn("stitch")

        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - dependency optional in tests
            raise RuntimeError("opencv-python is required for stitched archive OCR.") from exc

        with tempfile.TemporaryDirectory(prefix="imago-archive-ocr-") as tmp_dir_name:
            stitched = build_stitched_image([str(path) for path in group_paths])
            tmp_path = Path(tmp_dir_name) / f"{group_paths[0].stem}_ocr_stitched.jpg"
            wrote_temp_image = False
            if hasattr(cv2, "imwrite"):
                wrote_temp_image = bool(cv2.imwrite(str(tmp_path), stitched))
            else:
                try:
                    from PIL import Image  # pylint: disable=import-outside-toplevel

                    rgb_image = stitched[:, :, ::-1] if len(stitched.shape) == 3 else stitched
                    Image.fromarray(rgb_image).save(tmp_path, format="JPEG", quality=95)
                    wrote_temp_image = True
                except Exception:
                    wrote_temp_image = False
            if not wrote_temp_image:
                raise RuntimeError(f"Could not write temporary stitched OCR image: {tmp_path}")
            cap_wrote = False
            ocr_source_path = tmp_path
            if stitched_image_dir is not None:
                cap_path = stitched_image_dir / f"{group_paths[0].stem}_stitched.jpg"
                if hasattr(cv2, "imwrite"):
                    cap_wrote = bool(cv2.imwrite(str(cap_path), stitched))
                else:
                    try:
                        from PIL import Image  # pylint: disable=import-outside-toplevel

                        rgb_image = stitched[:, :, ::-1] if len(stitched.shape) == 3 else stitched
                        Image.fromarray(rgb_image).save(cap_path, format="JPEG", quality=95)
                        cap_wrote = True
                    except Exception:
                        pass
            if cap_wrote:
                ocr_source_path = cap_path
            ocr_text, ocr_keywords, ocr_hash = _run_authoritative_ocr(ocr_source_path)
            if cap_wrote:
                stitched_cap_path = cap_path

    result = ArchiveScanOCRAuthority(
        page_key=page_key,
        group_paths=tuple(group_paths),
        signature=group_signature,
        ocr_text=ocr_text,
        ocr_keywords=ocr_keywords,
        ocr_hash=ocr_hash,
        stitched_image_path=stitched_cap_path,
    )
    cache[page_key] = result
    return result


def _page_scan_filenames(image_path: Path) -> list[str]:
    """Return sorted list of scan TIF basenames associated with image_path's page.

    For scan TIFs (_S##): returns all sibling TIFs sharing the same page key.
    For derived or stitched/base page images: finds archive TIF scans for the same page.
    Returns [] if no scans are found.
    """
    if _scan_name_match(image_path):
        return [p.name for p in sorted(_scan_group_paths(image_path))]
    archive_dir = find_archive_dir_for_image(image_path)
    if archive_dir is None or not archive_dir.is_dir():
        return []
    _, _, _, page_str = parse_album_filename(image_path.name)
    if not page_str.isdigit() or int(page_str) == 0:
        return []
    page_int = int(page_str)
    scans: list[Path] = sorted(
        p
        for p in archive_dir.iterdir()
        for sm in (SCAN_TIFF_RE.match(p.name),)
        if sm and int(sm.group("page")) == page_int
    )
    return [p.name for p in scans]


def _build_dc_source(album_title: str, image_path: Path, scan_filenames: list[str]) -> str:
    """Build a human-readable dc:source string followed by source scan filenames.

    e.g. "Mainland China 1986 Book 11 Page 02 Scan(s) S01 S02; China_1986_B02_P17_S01.tif; ..."
    """
    _, _, _, _page_str = parse_album_filename(image_path.name)
    page_number = int(_page_str) if _page_str.isdigit() else 0
    source_filenames = _dedupe([str(fn or "").strip() for fn in scan_filenames if str(fn or "").strip()])
    if not source_filenames and (scan_match := _scan_name_match(image_path)):
        source_filenames = [image_path.name]
        scan_nums = [int(scan_match.group("scan"))]
    else:
        scan_nums = sorted(int(sm.group("scan")) for fn in source_filenames if (sm := _scan_name_match(fn)))
    parts: list[str] = [p for p in [str(album_title or "").strip()] if p]
    if page_number > 0:
        parts.append(f"Page {page_number:02d}")
    if scan_nums:
        parts.append("Scan(s) " + " ".join(f"S{n:02d}" for n in scan_nums))
    label = " ".join(parts)
    return "; ".join(p for p in [label] + source_filenames if p)


def _dc_source_needs_refresh(image_path: Path, sidecar_state: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar_state, dict):
        return False
    source_text = str(sidecar_state.get("source_text") or "").strip()
    album_title = str(sidecar_state.get("album_title") or "").strip()
    if not album_title and " Page " in source_text:
        album_title = source_text.split(" Page ", 1)[0].strip()
    expected_source = _build_dc_source(album_title, image_path, _page_scan_filenames(image_path))
    return source_text != expected_source


def _aggregate_best_rows(results: list[ImageAnalysis], section: str, key_name: str) -> list[dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for result in results:
        for row in list(result.payload.get(section) or []):
            name = str(row.get(key_name) or "").strip()
            if not name:
                continue
            score = float(row.get("score") or 0.0)
            current = best_rows.get(name)
            if current is None or score > float(current.get("score") or 0.0):
                best_rows[name] = dict(row)
    out = list(best_rows.values())
    out.sort(
        key=lambda row: (
            -float(row.get("score") or 0.0),
            str(row.get(key_name) or "").casefold(),
        )
    )
    return out


def _layout_payload(layout: PreparedImageLayout) -> dict[str, Any]:
    return {
        "kind": str(layout.kind),
        "page_like": bool(layout.page_like),
        "split_mode": str(layout.split_mode),
        "content_bounds": layout.content_bounds.as_dict(),
        "split_applied": bool(layout.split_applied),
        "fallback_used": bool(layout.fallback_used),
    }


def _bounds_offset(bounds: Any) -> tuple[int, int]:
    if hasattr(bounds, "x") and hasattr(bounds, "y"):
        return int(getattr(bounds, "x")), int(getattr(bounds, "y"))
    if hasattr(bounds, "as_dict"):
        try:
            payload = dict(bounds.as_dict())
        except Exception:
            payload = {}
        return int(payload.get("x", 0) or 0), int(payload.get("y", 0) or 0)
    if isinstance(bounds, dict):
        return int(bounds.get("x", 0) or 0), int(bounds.get("y", 0) or 0)
    return 0, 0


def _build_flat_payload(layout: PreparedImageLayout, analysis: ImageAnalysis) -> dict[str, Any]:
    payload = dict(analysis.payload)
    payload["layout"] = _layout_payload(layout)
    payload["subphotos"] = []
    return payload


def _build_flat_page_description(*, analysis: ImageAnalysis) -> str:
    return analysis.description
