"""Crop detected photo regions from page view JPEGs.

Reads MWG-RS mwg-rs:RegionList from a page view sidecar, converts normalised
centre-point coordinates to pixel rectangles, crops each region from the page
_V.jpg, and writes _D{index:02d}-00_V.jpg files under the album's _Photos/
sibling directory.

Entry point: crop_page_regions(view_path, photos_dir, *, force=False)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ..naming import DERIVED_NAME_RE as ALBUM_DERIVED_NAME_RE
from ..naming import archive_dir_for_album_dir, is_pages_dir
from .metadata_resolver import (
    build_location_filter_set as _resolver_build_location_filter_set,
)
from .metadata_resolver import (
    filter_location_names_from_people as _resolver_filter_location_names_from_people,
)
from .metadata_resolver import (
    location_payload_from_caption as _resolver_location_payload_from_caption,
)
from .metadata_resolver import (
    location_shown_from_payload as _resolver_location_shown_from_payload,
)
from .metadata_resolver import (
    match_caption_to_location_shown as _resolver_match_caption_to_location_shown,
)
from .metadata_resolver import (
    materialize_location_payload as _resolver_materialize_location_payload,
)
from .metadata_resolver import (
    normalize_location_payload as _resolver_normalize_location_payload,
)
from .metadata_resolver import (
    resolve_crop_location as _resolver_resolve_crop_location,
)
from .metadata_resolver import (
    resolve_crop_locations_shown as _resolver_resolve_crop_locations_shown,
)
from .metadata_resolver import (
    resolve_person_in_image as _resolver_resolve_person_in_image,
)

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
    region_text = str(region_name or "").strip()
    hint_text = str(region_caption_hint or "").strip()
    page_text = str(page_dc_description or "").strip()
    region_is_placeholder = bool(re.fullmatch(r"photo_\d+", region_text, flags=re.IGNORECASE))

    if region_text and not region_is_placeholder:
        return region_text
    if hint_text:
        return hint_text
    if page_text:
        if region_is_placeholder and not page_text.startswith("Page caption:"):
            return f"Page caption: {page_text}"
        return page_text
    return ""


def _format_numbered_caption(caption: str, index: int) -> str:
    text = str(caption or "").strip()
    if not text:
        return ""
    if text[-1:] not in {".", "!", "?"}:
        text = f"{text}."
    return f"{index}. {text}"


def _resolve_page_description_from_regions(regions: list[dict], fallback_description: str) -> str:
    unique_captions: list[str] = []
    seen: set[str] = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        caption = str(region.get("caption_hint") or region.get("caption") or "").strip()
        if not caption:
            continue
        if re.fullmatch(r"photo_\d+", caption, flags=re.IGNORECASE):
            continue
        folded = caption.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique_captions.append(caption)
    if len(unique_captions) >= 2:
        numbered = ", ".join(
            _format_numbered_caption(caption, index) for index, caption in enumerate(unique_captions, start=1)
        )
        return f"Page Captions: {numbered}"
    if len(unique_captions) == 1:
        return unique_captions[0]
    return str(fallback_description or "").strip()


def _normalize_region_location_payload(payload: dict[str, Any] | None) -> dict[str, str]:
    return _resolver_normalize_location_payload(payload)


def _build_location_filter_set(
    locations_shown: list[dict],
    location_payload: dict[str, Any] | None = None,
) -> set[str]:
    return _resolver_build_location_filter_set(
        locations_shown=locations_shown,
        location_payload=location_payload,
    )


def _filter_location_names_from_people(
    person_names: list[str] | tuple[str, ...] | None,
    location_filter_set: set[str] | None,
) -> list[str]:
    return _resolver_filter_location_names_from_people(person_names, location_filter_set)


def _match_caption_to_location_shown(
    caption: str,
    locations_shown: list[dict],
) -> dict[str, str] | None:
    return _resolver_match_caption_to_location_shown(caption, locations_shown)


def _materialize_region_location_payload(
    payload: dict[str, Any] | None,
    *,
    geocoder,
) -> dict[str, Any]:
    return _resolver_materialize_location_payload(payload, geocoder=geocoder)


def _update_page_description(
    view_xmp: Path,
    view_state: dict[str, Any],
    *,
    description: str,
    locations_shown: list[dict],
) -> None:
    del view_state, locations_shown
    import xml.etree.ElementTree as ET

    from .xmp_sidecar import DC_NS, _get_or_create_rdf_desc, _set_alt_text

    tree = ET.parse(view_xmp)
    desc = _get_or_create_rdf_desc(tree)
    existing = desc.find(f"{{{DC_NS}}}description")
    if existing is not None:
        desc.remove(existing)
    _set_alt_text(desc, f"{{{DC_NS}}}description", description)
    ET.indent(tree, space="  ")
    tree.write(str(view_xmp), encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def mwgrs_normalised_to_pixel_rect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    img_w: int,
    img_h: int,
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


def _merged_crop_subjects(view_xmp: Path, crop_xmp: Path) -> list[str]:
    seen: set[str] = set()
    subjects: list[str] = []
    for subject in _read_subjects_from_xmp(crop_xmp) + _read_subjects_from_xmp(view_xmp):
        if subject not in seen:
            seen.add(subject)
            subjects.append(subject)
    return subjects


def _crop_source_metadata(view_path: Path, view_state: dict) -> tuple[str, str]:
    from .ai_index_scan import _build_dc_source, _page_scan_filenames
    from .ai_render_settings import find_archive_dir_for_image
    from .ai_sidecar_state import _dc_source_scan_names, _effective_sidecar_album_title
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
    return crop_album_title, archive_source_text


def _page_location_for_crop(view_path: Path, view_state: dict, archive_source_text: str) -> dict:
    from .ai_render_settings import find_archive_dir_for_image
    from .ai_sidecar_state import _dc_source_scan_names, _sidecar_location_payload
    from .xmp_sidecar import read_ai_sidecar_state

    page_location = _sidecar_location_payload(view_state)
    if page_location and str(page_location.get("gps_latitude") or "").strip():
        return page_location

    archive_dir = find_archive_dir_for_image(view_path)
    if archive_dir is None or not archive_dir.is_dir():
        return page_location

    for scan_name in _dc_source_scan_names(archive_source_text)[:1]:
        archive_state = read_ai_sidecar_state((archive_dir / scan_name).with_suffix(".xmp"))
        if isinstance(archive_state, dict):
            archive_loc = _sidecar_location_payload(archive_state)
            if str(archive_loc.get("gps_latitude") or "").strip():
                return archive_loc
    return page_location


def _resolved_crop_location_metadata(
    *,
    caption: str,
    locations_shown: list[dict],
    page_location: dict,
    region_location_payload: dict[str, Any] | None,
    region_location_override: dict[str, Any] | None,
    geocoder,
) -> tuple[dict, list[dict]]:
    effective_loc = _materialize_region_location_payload(
        _resolver_resolve_crop_location(
            region_location_override=region_location_override,
            region_location_assigned=region_location_payload,
            caption=caption,
            locations_shown=locations_shown,
            page_location=page_location,
        ),
        geocoder=geocoder,
    )
    crop_locations_shown = _resolver_resolve_crop_locations_shown(
        region_location_override=region_location_override,
        region_location_assigned=region_location_payload,
        caption=caption,
        locations_shown=locations_shown,
    )
    if effective_loc and not crop_locations_shown:
        crop_location_shown = _resolver_location_shown_from_payload(effective_loc)
        crop_locations_shown = [crop_location_shown] if crop_location_shown else []
    if (
        effective_loc
        and crop_locations_shown
        and not all(str(location.get("gps_latitude") or "").strip() for location in crop_locations_shown)
    ):
        crop_location_shown = _resolver_location_shown_from_payload(effective_loc)
        crop_locations_shown = [crop_location_shown] if crop_location_shown else crop_locations_shown
    return effective_loc, crop_locations_shown


def _write_crop_provenance(crop_xmp: Path, view_path: Path, view_xmp: Path) -> None:
    from .xmpmm_provenance import assign_document_id, read_document_id, write_derived_from, write_pantry_entry

    assign_document_id(crop_xmp)
    view_doc_id = read_document_id(view_xmp)
    if view_doc_id:
        source_rel = Path(os.path.relpath(view_path, crop_xmp.parent)).as_posix()
        write_derived_from(crop_xmp, view_doc_id, source_path=source_rel)
        write_pantry_entry(crop_xmp, view_doc_id, source_path=source_rel)


class _CropXmpParams(TypedDict):
    caption: str
    view_state: dict
    view_xmp: Path
    album_title: str
    archive_source: str
    effective_loc: dict
    locations_shown: list[dict]
    person_names: list[str]
    parent_ocr: str


def _write_crop_xmp_metadata(
    crop_xmp: Path,
    *,
    params: _CropXmpParams,
) -> None:
    from .xmp_sidecar import write_xmp_sidecar

    subjects = _merged_crop_subjects(params["view_xmp"], crop_xmp)
    write_xmp_sidecar(
        crop_xmp,
        person_names=params["person_names"],
        subjects=subjects,
        description=params["caption"],
        album_title=params["album_title"],
        source_text=params["archive_source"],
        gps_latitude=str(params["effective_loc"].get("gps_latitude") or "").strip(),
        gps_longitude=str(params["effective_loc"].get("gps_longitude") or "").strip(),
        location_address=str(params["effective_loc"].get("address") or "").strip(),
        location_city=str(params["effective_loc"].get("city") or "").strip(),
        location_state=str(params["effective_loc"].get("state") or "").strip(),
        location_country=str(params["effective_loc"].get("country") or "").strip(),
        location_sublocation=str(params["effective_loc"].get("sublocation") or "").strip(),
        create_date=str(params["view_state"].get("create_date") or "").strip(),
        dc_date=list(params["view_state"].get("dc_date_values") or []),
        locations_shown=params["locations_shown"],
        parent_ocr_text=params["parent_ocr"],
        ocr_text="",
        detections_payload={"location": params["effective_loc"]} if params["effective_loc"] else None,
    )


def _write_crop_sidecar(
    *,
    crop_path: Path,
    view_path: Path,
    caption: str,
    view_state: dict,
    locations_shown: list[dict],
    person_names: list[str] | None = None,
    region_location_payload: dict[str, Any] | None = None,
    region_location_override: dict[str, Any] | None = None,
    geocoder=None,
) -> None:
    """Write or update the XMP sidecar for a crop JPEG.

    Writes DocumentID, DerivedFrom, Pantry, dc:description (if caption),
    dc:source, location/date/subject metadata from view_state, and
    PersonInImage. Preserves unrelated existing sidecar fields.
    """
    person_names = person_names or []
    crop_xmp = crop_path.with_suffix(".xmp")
    view_xmp = view_path.with_suffix(".xmp")

    _write_crop_provenance(crop_xmp, view_path, view_xmp)

    page_description = str(view_state.get("description") or "").strip()
    parent_ocr_text = str(view_state.get("parent_ocr_text") or "").strip()
    if not caption:
        caption = page_description

    crop_album_title, archive_source_text = _crop_source_metadata(view_path, view_state)

    page_location = _page_location_for_crop(view_path, view_state, archive_source_text)
    effective_loc, crop_locations_shown = _resolved_crop_location_metadata(
        caption=caption,
        locations_shown=locations_shown,
        page_location=page_location,
        region_location_payload=region_location_payload,
        region_location_override=region_location_override,
        geocoder=geocoder,
    )
    resolved_person_names = _resolver_resolve_person_in_image(
        person_names,
        locations_shown=locations_shown,
        location_payload=effective_loc,
    )

    _write_crop_xmp_metadata(
        crop_xmp,
        params={
            "caption": caption,
            "view_state": view_state,
            "view_xmp": view_xmp,
            "album_title": crop_album_title,
            "archive_source": archive_source_text,
            "effective_loc": effective_loc,
            "locations_shown": crop_locations_shown,
            "person_names": resolved_person_names,
            "parent_ocr": parent_ocr_text,
        },
    )
    _verify_crop_sidecar_metadata(
        crop_xmp,
        expected_album_title=crop_album_title,
        expected_source_text=archive_source_text,
    )


# ---------------------------------------------------------------------------
# Main crop function
# ---------------------------------------------------------------------------


def _region_location_needs_geocoder(region: dict, page_description: str) -> bool:
    payload = dict(region.get("location_override") or {}) or dict(region.get("location_payload") or {})
    normalized = _normalize_region_location_payload(payload)
    if normalized.get("address") and not normalized.get("gps_latitude"):
        return True
    caption_location = _resolver_location_payload_from_caption(
        resolve_region_caption(
            region.get("caption") or "",
            region.get("caption_hint") or "",
            page_description,
        )
    )
    return bool(caption_location)


def _crop_run_needs_geocoder(regions: list, locations_shown: list[dict], page_description: str) -> bool:
    if any(_region_location_needs_geocoder(region, page_description) for region in regions if isinstance(region, dict)):
        return True
    return any(
        (str(location.get("address") or "").strip() or str(location.get("name") or "").strip())
        and not str(location.get("gps_latitude") or "").strip()
        for location in locations_shown
        if isinstance(location, dict)
    )


def _crop_region_rect(
    *,
    region: dict,
    view_path: Path,
    img_w: int,
    img_h: int,
    stats: CropPageStats | None,
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = mwgrs_normalised_to_pixel_rect(
        region["cx"],
        region["cy"],
        region["nw"],
        region["nh"],
        img_w=img_w,
        img_h=img_h,
        warn_on_significant_clamp=False,
    )
    if right > left and bottom > top:
        return left, top, right, bottom
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
    return None


def _restore_crop_image(crop_img, skip_restoration: bool):
    restoration_result = "skipped"
    restoration_model = None
    if skip_restoration:
        return crop_img, restoration_result, restoration_model

    from .photo_restoration import (
        REAL_RESTORER_MODEL_NAME,
        RESTORE_RESULT_RESTORED,
        restore_photo_with_result,
    )

    crop_img, restoration_result = restore_photo_with_result(crop_img)
    if restoration_result == RESTORE_RESULT_RESTORED:
        restoration_model = REAL_RESTORER_MODEL_NAME
    return crop_img, restoration_result, restoration_model


def _write_page_crop_region(
    *,
    page_img,
    region: dict,
    view_path: Path,
    photos_dir: Path,
    view_state: dict,
    locations_shown: list[dict],
    page_description: str,
    img_w: int,
    img_h: int,
    archive_max_derived: int,
    force: bool,
    force_restoration: bool,
    skip_restoration: bool,
    stats: CropPageStats | None,
    geocoder,
) -> bool:
    from .xmp_sidecar import write_pipeline_step

    region_index = region["index"] + 1
    output_path = crop_output_path(
        view_path,
        region_index,
        photos_dir,
        archive_max_derived=archive_max_derived,
    )
    if output_path.exists() and not force and not force_restoration:
        return False

    rect = _crop_region_rect(region=region, view_path=view_path, img_w=img_w, img_h=img_h, stats=stats)
    if rect is None:
        return False

    caption = resolve_region_caption(
        region.get("caption") or "",
        region.get("caption_hint") or "",
        page_description,
    )
    crop_img = page_img.crop(rect)
    crop_img, restoration_result, restoration_model = _restore_crop_image(crop_img, skip_restoration)
    crop_img.save(str(output_path), format="JPEG", quality=95)

    _write_crop_sidecar(
        crop_path=output_path,
        view_path=view_path,
        caption=caption,
        view_state=view_state,
        locations_shown=locations_shown,
        person_names=list(region.get("person_names") or []),
        region_location_payload=dict(region.get("location_payload") or {}),
        region_location_override=dict(region.get("location_override") or {}),
        geocoder=geocoder,
    )
    write_pipeline_step(
        output_path.with_suffix(".xmp"),
        "photo_restoration",
        model=restoration_model,
        extra={"result": restoration_result},
    )
    return True


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
        write_pipeline_step,
        write_region_list,
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
    if _should_skip_existing_crop_regions(
        view_path,
        view_xmp,
        photos_dir,
        region_count=len(regions),
        force=force,
        force_restoration=force_restoration,
        stats=stats,
        read_pipeline_step=read_pipeline_step,
    ):
        return 0

    # Force: clear pipeline state and orphaned crops
    if force:
        clear_pipeline_steps(view_xmp, ["crop_regions"])
        _remove_orphaned_crops(view_path, photos_dir, len(regions))

    # Read page-level metadata
    view_state, locations_shown, page_description = _crop_page_metadata(
        view_xmp,
        regions=regions,
        read_ai_sidecar_state=read_ai_sidecar_state,
        read_locations_shown=read_locations_shown,
    )

    photos_dir.mkdir(parents=True, exist_ok=True)

    archive_max_derived = highest_archive_derived_number(view_path)
    geocoder = None
    if _crop_run_needs_geocoder(regions, locations_shown, page_description):
        from .ai_geocode import NominatimGeocoder

        geocoder = NominatimGeocoder()

    crops_written, failed = _write_page_crop_regions(
        Image,
        view_path=view_path,
        photos_dir=photos_dir,
        regions=regions,
        view_state=view_state,
        locations_shown=locations_shown,
        page_description=page_description,
        img_w=img_w,
        img_h=img_h,
        archive_max_derived=archive_max_derived,
        force=force,
        force_restoration=force_restoration,
        skip_restoration=skip_restoration,
        stats=stats,
        geocoder=geocoder,
    )

    if not failed:
        write_pipeline_step(view_xmp, "crop_regions")

    return crops_written


def _should_skip_existing_crop_regions(
    view_path: Path,
    view_xmp: Path,
    photos_dir: Path,
    *,
    region_count: int,
    force: bool,
    force_restoration: bool,
    stats: CropPageStats | None,
    read_pipeline_step,
) -> bool:
    if force or read_pipeline_step(view_xmp, "crop_regions") is None:
        return False
    if _has_complete_crop_outputs(view_path, photos_dir, region_count):
        if not force_restoration:
            if stats is not None:
                stats.skipped_existing_outputs = True
            return True
        return False
    if stats is not None:
        stats.reran_missing_outputs = True
    print(f"  [crop-regions] Re-running {view_path.name} (pipeline state present but crop outputs are missing)")
    return False


def _crop_page_metadata(
    view_xmp: Path,
    *,
    regions: list[dict],
    read_ai_sidecar_state,
    read_locations_shown,
) -> tuple[dict, list[dict], str]:
    view_state: dict = read_ai_sidecar_state(view_xmp) or {}
    locations_shown: list[dict] = read_locations_shown(view_xmp)
    page_description = str(view_state.get("description") or "").strip()
    resolved_page_description = _resolve_page_description_from_regions(regions, page_description)
    if resolved_page_description == page_description:
        return view_state, locations_shown, page_description
    _update_page_description(
        view_xmp,
        view_state,
        description=resolved_page_description,
        locations_shown=locations_shown,
    )
    view_state = dict(view_state)
    view_state["description"] = resolved_page_description
    return view_state, locations_shown, resolved_page_description


def _write_page_crop_regions(
    Image,
    *,
    view_path: Path,
    photos_dir: Path,
    regions: list[dict],
    view_state: dict,
    locations_shown: list[dict],
    page_description: str,
    img_w: int,
    img_h: int,
    archive_max_derived: int,
    force: bool,
    force_restoration: bool,
    skip_restoration: bool,
    stats: CropPageStats | None,
    geocoder,
) -> tuple[int, bool]:
    try:
        with Image.open(view_path) as page_img:
            return _write_open_page_crop_regions(
                page_img=page_img,
                view_path=view_path,
                photos_dir=photos_dir,
                regions=regions,
                view_state=view_state,
                locations_shown=locations_shown,
                page_description=page_description,
                img_w=img_w,
                img_h=img_h,
                archive_max_derived=archive_max_derived,
                force=force,
                force_restoration=force_restoration,
                skip_restoration=skip_restoration,
                stats=stats,
                geocoder=geocoder,
            )
    except Exception as exc:
        log.error("Failed to read page image %s: %s", view_path, exc)
        return 0, True


def _write_open_page_crop_regions(
    *,
    page_img,
    view_path: Path,
    photos_dir: Path,
    regions: list[dict],
    view_state: dict,
    locations_shown: list[dict],
    page_description: str,
    img_w: int,
    img_h: int,
    archive_max_derived: int,
    force: bool,
    force_restoration: bool,
    skip_restoration: bool,
    stats: CropPageStats | None,
    geocoder,
) -> tuple[int, bool]:
    crops_written = 0
    failed = False
    for region in regions:
        try:
            if _write_page_crop_region(
                page_img=page_img,
                region=region,
                view_path=view_path,
                photos_dir=photos_dir,
                view_state=view_state,
                locations_shown=locations_shown,
                page_description=page_description,
                img_w=img_w,
                img_h=img_h,
                archive_max_derived=archive_max_derived,
                force=force,
                force_restoration=force_restoration,
                skip_restoration=skip_restoration,
                stats=stats,
                geocoder=geocoder,
            ):
                crops_written += 1
        except Exception as exc:
            output_path = crop_output_path(
                view_path,
                region["index"] + 1,
                photos_dir,
                archive_max_derived=archive_max_derived,
            )
            log.error("Failed to write crop %s: %s", output_path.name, exc)
            failed = True
    return crops_written, failed


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
