"""Deterministic audit for crop-caption / page-region mismatches.

Scans photoalbums crop-view sidecars (``*_D##-##_V.xmp``) and their parent
page sidecars and emits CSV rows for suspicious cases, without rerunning any
model. Use it to build the small "suspicious set" worth a visual contact-sheet
review or a targeted Docling-only refresh.

Usage:
    python -m photoalbums.scripts.audit_crop_caption_mismatches \
        [--photos-root R] [--album SUBSTR] [--page NN] [--out audit.csv]

CSV columns: page, crop, crop_caption, page_region_caption, reason, confidence

Reasons (deterministic flags):
  crop_has_page_captions_summary  (high)   crop description is a "Page Captions:" summary
  crop_inherited_page_text        (high)   crop copied the page description instead of its region caption
  crop_missing_caption            (medium) crop description empty but region resolves to real text
  crop_caption_mismatch           (low)    crop description != resolved region caption
  metadata_photos_exceed_regions  (medium) page caption.photos entries outnumber photo regions
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib.ai_photo_crops import resolve_region_caption
from photoalbums.lib.caption_layout_migration import (
    _crop_region_index,
    _resolve_parent_page_sidecar,
)
from photoalbums.lib.xmp_sidecar import (
    _metadata_photos_by_number,
    read_ai_sidecar_state,
    read_region_list,
)
from photoalbums.naming import DERIVED_VIEW_RE, is_photos_dir, parse_album_filename

CSV_COLUMNS = ["page", "crop", "crop_caption", "page_region_caption", "reason", "confidence"]


def _iter_crop_sidecars(photos_root: Path, album_filter: str, page_filter: str):
    for sidecar_path in sorted(photos_root.rglob("*.xmp")):
        if not is_photos_dir(sidecar_path.parent):
            continue
        if not DERIVED_VIEW_RE.search(sidecar_path.stem):
            continue
        if album_filter and album_filter not in sidecar_path.parent.name.casefold():
            continue
        if page_filter:
            _, _, _, page = parse_album_filename(sidecar_path.name)
            page_token = f"{int(page):02d}" if str(page).isdigit() else str(page)
            if page_token != page_filter:
                continue
        yield sidecar_path


class _PageCache:
    """Lazily reads and memoises per-page sidecar state."""

    def __init__(self) -> None:
        self._regions: dict[Path, list[dict]] = {}
        self._state: dict[Path, dict] = {}

    def regions(self, page_sidecar: Path) -> list[dict]:
        if page_sidecar not in self._regions:
            # Caption/Name fields are unaffected by the dummy 1x1 dims.
            try:
                self._regions[page_sidecar] = read_region_list(page_sidecar, 1, 1)
            except Exception:
                self._regions[page_sidecar] = []
        return self._regions[page_sidecar]

    def state(self, page_sidecar: Path) -> dict:
        if page_sidecar not in self._state:
            state = read_ai_sidecar_state(page_sidecar)
            self._state[page_sidecar] = state if isinstance(state, dict) else {}
        return self._state[page_sidecar]


def _distinct_region_captions(regions: list[dict]) -> int:
    seen: set[str] = set()
    for region in regions:
        text = str(region.get("caption") or region.get("caption_hint") or "").strip()
        if text:
            seen.add(text.casefold())
    return len(seen)


def _crop_rows(crop_sidecar: Path, cache: _PageCache) -> list[dict]:
    """Return audit rows (0 or 1) for a single crop sidecar."""
    crop_state = read_ai_sidecar_state(crop_sidecar)
    if not isinstance(crop_state, dict):
        return [_row(crop_sidecar.name, crop_sidecar.name, "", "", "crop_sidecar_unreadable", "high")]
    crop_caption = str(crop_state.get("description") or "").strip()

    page_sidecar = _resolve_parent_page_sidecar(crop_sidecar)
    region_index = _crop_region_index(crop_sidecar)
    if page_sidecar is None or region_index is None:
        # No resolvable parent region (e.g. an archive-derived crop) - nothing to compare.
        return []

    page_name = page_sidecar.name
    regions = cache.regions(page_sidecar)
    if region_index >= len(regions):
        return [_row(page_name, crop_sidecar.name, crop_caption, "", "region_index_out_of_range", "high")]

    region = regions[region_index]
    region_caption = str(region.get("caption") or "").strip()
    region_hint = str(region.get("caption_hint") or "").strip()
    page_description = str(cache.state(page_sidecar).get("description") or "").strip()
    expected = resolve_region_caption(region_caption, region_hint, page_description).strip()

    def row(reason: str, confidence: str) -> dict:
        return _row(page_name, crop_sidecar.name, crop_caption, region_caption or expected, reason, confidence)

    # Priority order: most specific / highest-confidence flags first.
    if crop_caption.startswith("Page Captions:"):
        return [row("crop_has_page_captions_summary", "high")]

    page_is_multi = page_description.startswith("Page Captions:") or _distinct_region_captions(regions) >= 2
    if (
        page_description
        and crop_caption == page_description
        and page_is_multi
        and expected
        and crop_caption != expected
    ):
        return [row("crop_inherited_page_text", "high")]

    if not crop_caption and expected:
        return [row("crop_missing_caption", "medium")]

    if expected and crop_caption != expected:
        return [row("crop_caption_mismatch", "low")]

    return []


def _page_row(page_sidecar: Path, cache: _PageCache) -> dict | None:
    detections = cache.state(page_sidecar).get("detections")
    if not isinstance(detections, dict):
        return None
    metadata_photos = _metadata_photos_by_number(detections)
    photo_regions = [r for r in cache.regions(page_sidecar) if int(r.get("photo_number") or 0) > 0]
    if len(metadata_photos) > len(photo_regions):
        detail = f"caption.photos={len(metadata_photos)} photo_regions={len(photo_regions)}"
        return _row(page_sidecar.name, "", "", detail, "metadata_photos_exceed_regions", "medium")
    return None


def _row(page: str, crop: str, crop_caption: str, page_region_caption: str, reason: str, confidence: str) -> dict:
    return {
        "page": page,
        "crop": crop,
        "crop_caption": crop_caption,
        "page_region_caption": page_region_caption,
        "reason": reason,
        "confidence": confidence,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR), help="Photo Albums root directory.")
    parser.add_argument("--album", default="", help="Optional substring filter against the album directory name.")
    parser.add_argument("--page", default="", help="Optional page-number filter (e.g. 07).")
    parser.add_argument("--out", default="", help="Write CSV here instead of stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")

    album_filter = str(args.album or "").casefold()
    page_raw = str(args.page or "").strip()
    page_filter = f"{int(page_raw):02d}" if page_raw.isdigit() else page_raw

    cache = _PageCache()
    rows: list[dict] = []
    seen_pages: list[Path] = []
    seen_page_set: set[Path] = set()

    for crop_sidecar in _iter_crop_sidecars(photos_root, album_filter, page_filter):
        page_sidecar = _resolve_parent_page_sidecar(crop_sidecar)
        if page_sidecar is not None and page_sidecar not in seen_page_set:
            seen_page_set.add(page_sidecar)
            seen_pages.append(page_sidecar)
        try:
            rows.extend(_crop_rows(crop_sidecar, cache))
        except Exception as exc:
            rows.append(_row("", crop_sidecar.name, "", "", f"audit_error: {exc}", "high"))

    for page_sidecar in seen_pages:
        try:
            page_row = _page_row(page_sidecar, cache)
        except Exception as exc:
            rows.append(_row(page_sidecar.name, "", "", "", f"audit_error: {exc}", "high"))
            continue
        if page_row is not None:
            rows.append(page_row)

    out_handle = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(out_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out_handle.close()

    by_reason: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    summary = ", ".join(f"{reason}={count}" for reason, count in sorted(by_reason.items()))
    print(
        f"audited pages={len(seen_pages)} flags={len(rows)} [{summary}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
