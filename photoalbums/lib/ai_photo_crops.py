"""Crop detected photo regions from CTM-corrected page view JPEGs.

Reads MWG-RS mwg-rs:RegionList from a page view sidecar, converts normalised
centre-point coordinates to pixel rectangles, crops each region from the page
_V.jpg, and writes _D{index:02d}-00_V.jpg files under the album's _Photos/
sibling directory.

Entry point: crop_page_regions(view_path, photos_dir, *, force=False)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path

from ..naming import DERIVED_NAME_RE as ALBUM_DERIVED_NAME_RE
from ..naming import archive_dir_for_album_dir, is_pages_dir

log = logging.getLogger(__name__)


@dataclass
class CropPageStats:
    ignored_empty_regions: int = 0
    skipped_existing_outputs: bool = False
    reran_missing_outputs: bool = False


# ---------------------------------------------------------------------------
# Caption resolution
# ---------------------------------------------------------------------------


def resolve_region_caption(
    region_name: str,
    region_caption_hint: str,
    page_dc_description: str,
) -> str:
    """Return the best available caption for a region using priority order.

    Priority:
    1. region_name
    2. region_caption_hint
    3. page_dc_description
    4. "" (empty)
    """
    for candidate in (region_name, region_caption_hint, page_dc_description):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def mwgrs_normalised_to_pixel_rect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    *,
    warn_on_significant_clamp: bool = True,
) -> tuple[int, int, int, int]:
    """Convert MWG-RS centre-point normalised coords to a pixel rectangle.

    Returns (left, top, right, bottom) clamped to image bounds.
    Logs a warning if any dimension was clamped by more than 5% of the image.
    """
    left_f = (cx - w / 2.0) * img_w
    top_f = (cy - h / 2.0) * img_h
    right_f = (cx + w / 2.0) * img_w
    bottom_f = (cy + h / 2.0) * img_h

    left = max(0, int(round(left_f)))
    top = max(0, int(round(top_f)))
    right = min(img_w, int(round(right_f)))
    bottom = min(img_h, int(round(bottom_f)))

    threshold_w = img_w * 0.05
    threshold_h = img_h * 0.05
    if warn_on_significant_clamp and (
        left - left_f > threshold_w
        or top - top_f > threshold_h
        or right_f - right > threshold_w
        or bottom_f - bottom > threshold_h
    ):
        log.warning(
            "Region coords clamped significantly: raw=(%.2f, %.2f, %.2f, %.2f) clamped=(%d, %d, %d, %d) img=(%d, %d)",
            left_f,
            top_f,
            right_f,
            bottom_f,
            left,
            top,
            right,
            bottom,
            img_w,
            img_h,
        )

    return left, top, right, bottom


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

_DERIVED_NAME_RE = re.compile(r"_D\d{2}-\d{2}_V\b")


def crop_page_prefix(view_path: str | Path) -> str:
    stem = Path(view_path).stem  # e.g. Egypt_1975_B00_P26_V
    return stem[:-2] if stem.endswith("_V") else stem


def highest_archive_derived_number(view_path: str | Path) -> int:
    """Return the highest D## already assigned in the sibling _Archive directory for this page."""
    view_path = Path(view_path)
    if not is_pages_dir(view_path.parent):
        return 0
    archive_dir = archive_dir_for_album_dir(view_path.parent)
    if not archive_dir.is_dir():
        return 0

    page_prefix = crop_page_prefix(view_path)
    highest = 0
    for candidate in archive_dir.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() == ".xmp":
            continue
        stem = candidate.stem
        if not stem.startswith(f"{page_prefix}_D"):
            continue
        match = ALBUM_DERIVED_NAME_RE.search(stem)
        if match is None:
            continue
        highest = max(highest, int(match.group("derived")))
    return highest


def _crop_output_stem(view_path: str | Path, derived_number: int) -> str:
    return f"{crop_page_prefix(view_path)}_D{derived_number:02d}-00_V"


def crop_output_path(
    view_path: str | Path,
    region_index: int,
    photos_dir: str | Path,
    *,
    archive_max_derived: int | None = None,
) -> Path:
    """Build the canonical _D{index:02d}-00_V.jpg path under photos_dir.

    region_index is 1-based (matching MWG-RS photo_1, photo_2, ...).
    Crop numbering starts after the highest archive-derived number already used
    for the same page in the sibling _Archive directory.
    """
    if archive_max_derived is None:
        archive_max_derived = highest_archive_derived_number(view_path)
    filename = f"{_crop_output_stem(view_path, archive_max_derived + region_index)}.jpg"
    return Path(photos_dir) / filename


def _expected_crop_output_paths(view_path: str | Path, photos_dir: str | Path, region_count: int) -> list[Path]:
    archive_max_derived = highest_archive_derived_number(view_path)
    return [
        crop_output_path(view_path, index, photos_dir, archive_max_derived=archive_max_derived)
        for index in range(1, region_count + 1)
    ]


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------


def _read_subjects_from_xmp(sidecar_path: Path) -> list[str]:
    """Read dc:subject bag from an XMP sidecar. Returns [] on any error."""
    import xml.etree.ElementTree as ET

    _DC_NS = "http://purl.org/dc/elements/1.1/"
    _RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    if not sidecar_path.is_file():
        return []
    try:
        tree = ET.parse(str(sidecar_path))
    except ET.ParseError:
        return []
    root = tree.getroot()
    if root is None:
        return []
    # Walk any rdf:Description
    subjects: list[str] = []
    for subj_el in root.iter(f"{{{_DC_NS}}}subject"):
        for li in subj_el.iter(f"{{{_RDF_NS}}}li"):
            text = str(li.text or "").strip()
            if text:
                subjects.append(text)
    return subjects


def _verify_crop_sidecar_metadata(
    crop_xmp: Path,
    *,
    expected_album_title: str,
    expected_source_text: str,
) -> None:
    from .xmp_sidecar import read_ai_sidecar_state

    state = read_ai_sidecar_state(crop_xmp)
    if not isinstance(state, dict):
        raise RuntimeError(f"Crop sidecar verification failed due to unreadable XMP: {crop_xmp}")
    actual_source_text = str(state.get("source_text") or "").strip()
    if actual_source_text != expected_source_text:
        raise RuntimeError(
            f"Crop sidecar dc:source verification failed for {crop_xmp}: "
            f"expected {expected_source_text!r}, got {actual_source_text!r}"
        )
    if expected_album_title:
        actual_album_title = str(state.get("album_title") or "").strip()
        if actual_album_title != expected_album_title:
            raise RuntimeError(
                f"Crop sidecar imago:AlbumTitle verification failed for {crop_xmp}: "
                f"expected {expected_album_title!r}, got {actual_album_title!r}"
            )


def _write_crop_sidecar(
    crop_path: Path,
    view_path: Path,
    caption: str,
    view_state: dict,
    locations_shown: list[dict],
    person_names: list[str],
) -> None:
    """Write or update the XMP sidecar for a crop JPEG.

    Writes DocumentID, DerivedFrom, Pantry, dc:description (if caption),
    dc:source, location/date/subject metadata from view_state, and
    PersonInImage. Preserves unrelated existing sidecar fields.
    """
    from .xmpmm_provenance import assign_document_id, write_derived_from, write_pantry_entry
    from .ai_index_scan import _build_dc_source, _page_scan_filenames
    from .xmp_sidecar import write_xmp_sidecar

    crop_xmp = crop_path.with_suffix(".xmp")
    view_xmp = view_path.with_suffix(".xmp")

    # Step 1: ensure crop sidecar exists and has a DocumentID
    assign_document_id(crop_xmp)

    # Step 2: get the page view's DocumentID for provenance links
    from .xmpmm_provenance import read_document_id

    view_doc_id = read_document_id(view_xmp)

    # Step 3: write provenance
    if view_doc_id:
        source_rel = Path(os.path.relpath(view_path, crop_xmp.parent)).as_posix()
        write_derived_from(crop_xmp, view_doc_id, source_path=source_rel)
        write_pantry_entry(crop_xmp, view_doc_id, source_path=source_rel)

    # Step 4: write metadata via write_xmp_sidecar (handles merge)
    # Merge view subjects with any subjects already on the crop sidecar
    view_subjects = _read_subjects_from_xmp(view_xmp)
    crop_subjects = _read_subjects_from_xmp(crop_xmp)
    seen: set[str] = set()
    subjects: list[str] = []
    for s in crop_subjects + view_subjects:
        if s not in seen:
            seen.add(s)
            subjects.append(s)

    page_description = str(view_state.get("description") or "").strip()
    parent_ocr_text = str(view_state.get("parent_ocr_text") or view_state.get("ocr_text") or "").strip()
    if not caption:
        caption = page_description
    from .ai_sidecar_state import _dc_source_scan_names, _effective_sidecar_album_title, _sidecar_location_payload
    from .ai_render_settings import find_archive_dir_for_image
    from .xmp_sidecar import read_ai_sidecar_state

    crop_album_title = _effective_sidecar_album_title(view_path, view_state)
    archive_source_text = str(view_state.get("source_text") or "").strip()
    archive_scan_names = _page_scan_filenames(view_path) or _dc_source_scan_names(archive_source_text)
    if not crop_album_title and archive_scan_names:
        archive_dir = find_archive_dir_for_image(view_path)
        if archive_dir is not None and archive_dir.is_dir():
            archive_state = read_ai_sidecar_state((archive_dir / archive_scan_names[0]).with_suffix(".xmp"))
            if isinstance(archive_state, dict):
                crop_album_title = _effective_sidecar_album_title(view_path, archive_state)
    if archive_scan_names:
        archive_source_text = _build_dc_source(crop_album_title, view_path, archive_scan_names)

    # Compute effective location: page view state first, then archive scan fallback.
    # Crops are created before the page view refresh may have written GPS/location,
    # so we walk up to the archive scan if the page view has no GPS.
    effective_loc = _sidecar_location_payload(view_state)
    if not str(effective_loc.get("gps_latitude") or "").strip():
        archive_dir = find_archive_dir_for_image(view_path)
        if archive_dir is not None and archive_dir.is_dir():
            for scan_name in _dc_source_scan_names(archive_source_text)[:1]:
                archive_state = read_ai_sidecar_state((archive_dir / scan_name).with_suffix(".xmp"))
                if isinstance(archive_state, dict):
                    archive_loc = _sidecar_location_payload(archive_state)
                    if str(archive_loc.get("gps_latitude") or "").strip():
                        effective_loc = archive_loc
                        break

    write_xmp_sidecar(
        crop_xmp,
        creator_tool="imago-crop-regions",
        person_names=list(person_names),
        subjects=subjects,
        description=caption,
        album_title=crop_album_title,
        source_text=archive_source_text,
        gps_latitude=str(effective_loc.get("gps_latitude") or "").strip(),
        gps_longitude=str(effective_loc.get("gps_longitude") or "").strip(),
        location_city=str(effective_loc.get("city") or view_state.get("location_city") or "").strip(),
        location_state=str(effective_loc.get("state") or view_state.get("location_state") or "").strip(),
        location_country=str(effective_loc.get("country") or view_state.get("location_country") or "").strip(),
        location_sublocation=str(effective_loc.get("sublocation") or view_state.get("location_sublocation") or "").strip(),
        create_date=str(view_state.get("create_date") or "").strip(),
        dc_date=list(view_state.get("dc_date_values") or []),
        locations_shown=locations_shown,
        parent_ocr_text=parent_ocr_text,
        ocr_text="",
    )
    _verify_crop_sidecar_metadata(
        crop_xmp,
        expected_album_title=crop_album_title,
        expected_source_text=archive_source_text,
    )


# ---------------------------------------------------------------------------
# Main crop function
# ---------------------------------------------------------------------------


def crop_page_regions(
    view_path: str | Path,
    photos_dir: str | Path,
    *,
    force: bool = False,
    skip_restoration: bool = False,
    force_restoration: bool = False,
    stats: CropPageStats | None = None,
) -> int:
    """Crop each detected region from a page view JPEG and write to photos_dir.

    Reads mwg-rs:RegionList from the view XMP sidecar. For each region:
    - Resolves caption via resolve_region_caption()
    - Converts normalised coords to pixel rect
    - Crops from the page _V.jpg using Pillow
    - Writes _D{index:02d}-00_V.jpg under photos_dir
    - Writes/updates XMP sidecar with provenance and metadata

    Skips silently if no regions exist. Skips existing crops unless force=True.
    Tracks completion in pipeline.crop_regions on the page view sidecar.

    Returns the count of crops written.
    """
    from PIL import Image
    from .xmp_sidecar import (
        clear_pipeline_steps,
        read_ai_sidecar_state,
        read_locations_shown,
        read_pipeline_step,
        read_region_list,
        write_region_list,
        write_pipeline_step,
    )

    view_path = Path(view_path)
    photos_dir = Path(photos_dir)
    view_xmp = view_path.with_suffix(".xmp")

    if _DERIVED_NAME_RE.search(view_path.name):
        log.info("Skipping derived view crop source: %s", view_path.name)
        return 0

    if not view_path.is_file():
        log.warning("View JPEG not found: %s", view_path)
        return 0

    # Read image dimensions
    try:
        with Image.open(view_path) as img:
            img_w, img_h = img.size
    except Exception as exc:
        log.error("Failed to open %s: %s", view_path, exc)
        return 0

    view_regions_state = read_pipeline_step(view_xmp, "view_regions") or {}
    if str(view_regions_state.get("result") or "").strip() == "no_regions":
        write_region_list(view_xmp, [], img_w, img_h)
        return 0

    # Read regions
    regions = read_region_list(view_xmp, img_w, img_h)
    if not regions:
        return 0

    # Pipeline state check
    if not force and read_pipeline_step(view_xmp, "crop_regions") is not None:
        if _has_complete_crop_outputs(view_path, photos_dir, len(regions)):
            if not force_restoration:
                if stats is not None:
                    stats.skipped_existing_outputs = True
                return 0
        else:
            if stats is not None:
                stats.reran_missing_outputs = True
            print(f"  [crop-regions] Re-running {view_path.name} (pipeline state present but crop outputs are missing)")

    # Force: clear pipeline state and orphaned crops
    if force:
        clear_pipeline_steps(view_xmp, ["crop_regions"])
        _remove_orphaned_crops(view_path, photos_dir, len(regions))

    # Read page-level metadata
    view_state: dict = read_ai_sidecar_state(view_xmp) or {}
    locations_shown: list[dict] = read_locations_shown(view_xmp)
    page_description = str(view_state.get("description") or "").strip()

    photos_dir.mkdir(parents=True, exist_ok=True)

    crops_written = 0
    failed = False
    archive_max_derived = highest_archive_derived_number(view_path)

    try:
        with Image.open(view_path) as page_img:
            for region in regions:
                region_index = region["index"] + 1  # 1-based
                output_path = crop_output_path(
                    view_path,
                    region_index,
                    photos_dir,
                    archive_max_derived=archive_max_derived,
                )

                if output_path.exists() and not force and not force_restoration:
                    continue

                caption = resolve_region_caption(
                    region.get("caption") or "",
                    region.get("caption_hint") or "",
                    page_description,
                )
                person_names = list(region.get("person_names") or [])

                try:
                    cx = region["cx"]
                    cy = region["cy"]
                    nw = region["nw"]
                    nh = region["nh"]
                    left, top, right, bottom = mwgrs_normalised_to_pixel_rect(
                        cx,
                        cy,
                        nw,
                        nh,
                        img_w,
                        img_h,
                        warn_on_significant_clamp=False,
                    )
                    if right <= left or bottom <= top:
                        if stats is not None:
                            stats.ignored_empty_regions += 1
                        log.warning(
                            "Ignoring empty crop region for %s after clamping: rect=(%d, %d, %d, %d) img=(%d, %d)",
                            view_path.name,
                            left,
                            top,
                            right,
                            bottom,
                            img_w,
                            img_h,
                        )
                        continue
                    crop_img = page_img.crop((left, top, right, bottom))
                    restoration_result = "skipped"
                    restoration_model = None
                    if not skip_restoration:
                        from .photo_restoration import (
                            REAL_RESTORER_MODEL_NAME,
                            RESTORE_RESULT_RESTORED,
                            restore_photo_with_result,
                        )

                        crop_img, restoration_result = restore_photo_with_result(crop_img)
                        if restoration_result == RESTORE_RESULT_RESTORED:
                            restoration_model = REAL_RESTORER_MODEL_NAME
                    crop_img.save(str(output_path), format="JPEG", quality=95)

                    _write_crop_sidecar(
                        output_path,
                        view_path,
                        caption,
                        view_state,
                        locations_shown,
                        person_names,
                    )
                    write_pipeline_step(
                        output_path.with_suffix(".xmp"),
                        "photo_restoration",
                        model=restoration_model,
                        extra={"result": restoration_result},
                    )
                    crops_written += 1
                except Exception as exc:
                    log.error("Failed to write crop %s: %s", output_path.name, exc)
                    failed = True
    except Exception as exc:
        log.error("Failed to read page image %s: %s", view_path, exc)
        return 0

    if not failed:
        write_pipeline_step(view_xmp, "crop_regions")

    return crops_written


def _remove_orphaned_crops(view_path: Path, photos_dir: Path, current_region_count: int) -> None:
    """Remove _D##-00_V.jpg crops (and sidecars) not produced by the current region set."""
    if not photos_dir.is_dir():
        return
    page_prefix = crop_page_prefix(view_path)
    archive_max_derived = highest_archive_derived_number(view_path)

    # Build the set of paths that the current run will produce
    expected_stems = {
        _crop_output_stem(view_path, archive_max_derived + index) for index in range(1, current_region_count + 1)
    }

    orphan_pattern = re.compile(rf"^{re.escape(page_prefix)}_D(\d+)-00_V$")
    for f in list(photos_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".xmp"}:
            continue
        candidate_stem = f.stem
        if orphan_pattern.match(candidate_stem) and candidate_stem not in expected_stems:
            f.unlink(missing_ok=True)
            log.debug("Removed orphaned crop file: %s", f.name)


def _has_complete_crop_outputs(view_path: Path, photos_dir: Path, region_count: int) -> bool:
    expected_outputs = _expected_crop_output_paths(view_path, photos_dir, region_count)
    if not expected_outputs:
        return True
    return all(output_path.is_file() and output_path.with_suffix(".xmp").is_file() for output_path in expected_outputs)
