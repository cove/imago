"""Crop detected photo regions from CTM-corrected page view JPEGs.

Reads MWG-RS mwg-rs:RegionList from a page view sidecar, converts normalised
centre-point coordinates to pixel rectangles, crops each region from the page
_V.jpg, and writes _D{index:02d}-00_V.jpg files under the album's _Photos/
sibling directory.

Entry point: crop_page_regions(view_path, photos_dir, *, force=False)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption resolution
# ---------------------------------------------------------------------------


def resolve_region_caption(
    region_dc_description: str,
    region_caption_hint: str,
    page_dc_description: str,
) -> str:
    """Return the best available caption for a region using priority order.

    Priority:
    1. region_dc_description
    2. region_caption_hint
    3. page_dc_description
    4. "" (empty)
    """
    for candidate in (region_dc_description, region_caption_hint, page_dc_description):
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
    if (
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


def crop_output_path(view_path: str | Path, region_index: int, photos_dir: str | Path) -> Path:
    """Build the _D{index:02d}-00_V.jpg path under photos_dir.

    region_index is 1-based (matching MWG-RS photo_1, photo_2, ...).
    Example: Egypt_1975_B00_P26_V.jpg + index 2 -> Egypt_1975_B00_P26_D02-00_V.jpg
    """
    stem = Path(view_path).stem  # e.g. Egypt_1975_B00_P26_V
    # Strip trailing _V marker to get the page prefix
    page_prefix = stem[:-2] if stem.endswith("_V") else stem
    filename = f"{page_prefix}_D{region_index:02d}-00_V.jpg"
    return Path(photos_dir) / filename


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
    from .xmp_sidecar import write_xmp_sidecar, read_ai_sidecar_state

    crop_xmp = crop_path.with_suffix(".xmp")
    view_xmp = view_path.with_suffix(".xmp")

    # Step 1: ensure crop sidecar exists and has a DocumentID
    assign_document_id(crop_xmp)

    # Step 2: get the page view's DocumentID for provenance links
    from .xmpmm_provenance import read_document_id

    view_doc_id = read_document_id(view_xmp)

    # Step 3: write provenance
    if view_doc_id:
        source_rel = view_path.name
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

    write_xmp_sidecar(
        crop_xmp,
        creator_tool="imago-crop-regions",
        person_names=list(person_names),
        subjects=subjects,
        description=caption,
        source_text=view_path.name,
        gps_latitude=str(view_state.get("gps_latitude") or "").strip(),
        gps_longitude=str(view_state.get("gps_longitude") or "").strip(),
        location_city=str(view_state.get("location_city") or "").strip(),
        location_state=str(view_state.get("location_state") or "").strip(),
        location_country=str(view_state.get("location_country") or "").strip(),
        location_sublocation=str(view_state.get("location_sublocation") or "").strip(),
        create_date=str(view_state.get("create_date") or "").strip(),
        dc_date=list(view_state.get("dc_date_values") or []),
        locations_shown=locations_shown,
        ocr_text="",
    )


# ---------------------------------------------------------------------------
# Main crop function
# ---------------------------------------------------------------------------


def crop_page_regions(
    view_path: str | Path,
    photos_dir: str | Path,
    *,
    force: bool = False,
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
    from .xmp_sidecar import read_region_list, read_ai_sidecar_state, read_locations_shown
    from .xmpmm_provenance import write_pipeline_step, clear_pipeline_steps, read_pipeline_step

    view_path = Path(view_path)
    photos_dir = Path(photos_dir)
    view_xmp = view_path.with_suffix(".xmp")

    # Pipeline state check
    if not force and read_pipeline_step(view_xmp, "crop_regions") is not None:
        print(f"  [crop-regions] Skipping {view_path.name} (pipeline state present; use --force to rerun)")
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

    # Read regions
    regions = read_region_list(view_xmp, img_w, img_h)
    if not regions:
        return 0

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

    try:
        with Image.open(view_path) as page_img:
            for region in regions:
                region_index = region["index"] + 1  # 1-based
                output_path = crop_output_path(view_path, region_index, photos_dir)

                if output_path.exists() and not force:
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
                    left, top, right, bottom = mwgrs_normalised_to_pixel_rect(cx, cy, nw, nh, img_w, img_h)
                    crop_img = page_img.crop((left, top, right, bottom))
                    crop_img.save(str(output_path), format="JPEG", quality=95)

                    _write_crop_sidecar(
                        output_path,
                        view_path,
                        caption,
                        view_state,
                        locations_shown,
                        person_names,
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
    stem = view_path.stem  # e.g. Egypt_1975_B00_P26_V
    page_prefix = stem[:-2] if stem.endswith("_V") else stem

    # Build the set of paths that the current run will produce
    expected_stems = {f"{page_prefix}_D{i:02d}-00_V" for i in range(1, current_region_count + 1)}

    orphan_pattern = re.compile(rf"^{re.escape(page_prefix)}_D(\d{{2}})-00_V$")
    for f in list(photos_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".xmp"}:
            continue
        candidate_stem = f.stem
        if f.suffix.lower() in {".xmp"}:
            # strip .xmp to get stem
            candidate_stem = Path(f.stem).stem if f.stem.endswith("_V") else f.stem
        if orphan_pattern.match(candidate_stem) and candidate_stem not in expected_stems:
            f.unlink(missing_ok=True)
            log.debug("Removed orphaned crop file: %s", f.name)
