from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .xmp_sidecar import (
    DC_NS,
    IMAGO_NS,
    IPTC_EXT_NS,
    RDF_NS,
    read_ai_sidecar_state,
    read_person_in_image,
)

_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_LI = f"{{{RDF_NS}}}li"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_named_file_path(
    photos_root: str | Path,
    value: str,
    *,
    label: str,
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate

    photos_root_path = Path(photos_root)
    if not photos_root_path.is_dir():
        raise ValueError(f"photos_root is not a directory: {photos_root_path}")

    target_name = candidate.name.casefold()
    matches = sorted(
        path.resolve() for path in photos_root_path.rglob("*") if path.is_file() and path.name.casefold() == target_name
    )
    if not matches:
        raise ValueError(f"{label} filename '{value}' was not found under photos_root '{photos_root_path}'.")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches[:10])
        if len(matches) > 10:
            joined += f", ... ({len(matches)} matches total)"
        raise ValueError(f"{label} filename '{value}' is ambiguous under photos_root '{photos_root_path}': {joined}")
    return matches[0]


def resolve_ai_xmp_review_path(
    photos_root: str | Path,
    file_name: str,
) -> tuple[Path, Path | None]:
    """Resolve an XMP sidecar from a single image or XMP filename/path."""
    file_value = _clean_text(file_name)
    if not file_value:
        raise ValueError("Provide a file_name.")

    candidate = _resolve_named_file_path(photos_root, file_value, label="File")
    if candidate.suffix.casefold() == ".xmp":
        if not candidate.is_file():
            raise ValueError(f"XMP sidecar was not found: {candidate}")
        return candidate.resolve(), None

    photo_path = candidate
    if not photo_path.is_file():
        raise ValueError(f"Photo was not found: {photo_path}")
    sidecar_path = photo_path.with_suffix(".xmp")
    if not sidecar_path.is_file():
        raise ValueError(f"No XMP sidecar was found for photo '{photo_path}'. Expected '{sidecar_path}'.")
    return sidecar_path.resolve(), photo_path.resolve()


def _get_rdf_desc(tree: ET.ElementTree) -> ET.Element | None:
    root = tree.getroot()
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        return None
    return rdf.find(_RDF_DESC)


def _read_alt_text(parent: ET.Element, tag: str) -> str:
    field = parent.find(tag)
    if field is None:
        return ""
    alt = field.find(_RDF_ALT)
    if alt is None:
        return ""
    fallback = ""
    for item in alt.findall(_RDF_LI):
        text = _clean_text(item.text)
        if not text:
            continue
        if item.get(_XML_LANG) == "x-default":
            return text
        if not fallback:
            fallback = text
    return fallback


def _read_bag_values(parent: ET.Element, tag: str) -> list[str]:
    field = parent.find(tag)
    if field is None:
        return []
    bag = field.find(_RDF_BAG)
    if bag is None:
        return []
    values: list[str] = []
    for item in bag.findall(_RDF_LI):
        text = _clean_text(item.text)
        if text:
            values.append(text)
    return values


def _read_json_text(value: object) -> object | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _read_iptc_subphotos(desc: ET.Element) -> list[dict[str, object]]:
    field = desc.find(f"{{{IPTC_EXT_NS}}}ImageRegion")
    if field is None:
        return []
    bag = field.find(_RDF_BAG)
    if bag is None:
        return []
    rows: list[dict[str, object]] = []
    next_index = 1
    for item in bag.findall(_RDF_LI):
        region_id = _clean_text(item.findtext(f"{{{IPTC_EXT_NS}}}rId", default=""))
        if not region_id.casefold().startswith("photo"):
            continue
        match = region_id.casefold().removeprefix("photo-").strip()
        index = _coerce_int(match, default=(0 if region_id == "photo" else next_index))
        if index <= 0:
            index = next_index
        boundary = item.find(f"{{{IPTC_EXT_NS}}}RegionBoundary")
        rel_bounds = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        if boundary is not None:
            try:
                rel_bounds = {
                    "x": float(boundary.findtext(f"{{{IPTC_EXT_NS}}}rbX", default="0") or 0.0),
                    "y": float(boundary.findtext(f"{{{IPTC_EXT_NS}}}rbY", default="0") or 0.0),
                    "width": float(boundary.findtext(f"{{{IPTC_EXT_NS}}}rbW", default="0") or 0.0),
                    "height": float(boundary.findtext(f"{{{IPTC_EXT_NS}}}rbH", default="0") or 0.0),
                }
            except (TypeError, ValueError):
                rel_bounds = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        rows.append(
            {
                "index": index,
                "bounds": {
                    "x": rel_bounds["x"],
                    "y": rel_bounds["y"],
                    "width": rel_bounds["width"],
                    "height": rel_bounds["height"],
                },
                "description": _read_alt_text(item, f"{{{DC_NS}}}description"),
                "scene_text": _clean_text(item.findtext(f"{{{IMAGO_NS}}}SceneText", default="")),
                "people": _read_bag_values(item, f"{{{IMAGO_NS}}}People"),
                "subjects": _read_bag_values(item, f"{{{IMAGO_NS}}}Subjects"),
                "detections": _read_json_text(item.findtext(f"{{{IMAGO_NS}}}Detections", default="")),
            }
        )
        next_index += 1
    return rows


def _read_subphotos(desc: ET.Element) -> list[dict[str, object]]:
    return _read_iptc_subphotos(desc)


def load_ai_xmp_review(
    sidecar_path: str | Path,
    *,
    include_raw_xml: bool = False,
) -> dict[str, object]:
    """Load a photoalbums AI XMP sidecar into a review-friendly structure."""
    path = Path(sidecar_path)
    if not path.is_file():
        raise ValueError(f"XMP sidecar was not found: {path}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"XMP sidecar is not valid XML: {path}") from exc

    desc = _get_rdf_desc(tree)
    if desc is None:
        raise ValueError(f"XMP sidecar is missing rdf:Description: {path}")

    state = read_ai_sidecar_state(path)
    if state is None:
        raise ValueError(f"XMP sidecar could not be parsed for review: {path}")

    detections = state.get("detections")
    detection_people = detections.get("people") if isinstance(detections, dict) else None
    detection_objects = detections.get("objects") if isinstance(detections, dict) else None
    detection_ocr = detections.get("ocr") if isinstance(detections, dict) else None

    person_names = read_person_in_image(path)
    subjects = _read_bag_values(desc, f"{{{DC_NS}}}subject")
    subphotos = _read_subphotos(desc)

    result: dict[str, object] = {
        "sidecar_path": str(path.resolve()),
        "creator_tool": _clean_text(state.get("creator_tool")),
        "create_date": _clean_text(state.get("create_date")),
        "title": _clean_text(state.get("title")),
        "person_names": person_names,
        "subjects": subjects,
        "description": _clean_text(state.get("description")),
        "album_title": _clean_text(state.get("album_title")),
        "gps_latitude": _clean_text(state.get("gps_latitude")),
        "gps_longitude": _clean_text(state.get("gps_longitude")),
        "source_text": _clean_text(desc.findtext(f"{{{DC_NS}}}source", default="")),
        "ocr_text": _clean_text(state.get("ocr_text")),
        "author_text": _clean_text(state.get("author_text")),
        "scene_text": _clean_text(state.get("scene_text")),
        "title_source": _clean_text(state.get("title_source")),
        "ocr_authority_source": _clean_text(state.get("ocr_authority_source")),
        "stitch_key": _clean_text(state.get("stitch_key")),
        "ocr_ran": state.get("ocr_ran"),
        "people_detected": state.get("people_detected"),
        "people_identified": state.get("people_identified"),
        "processing_history": state.get("processing_history"),
        "subphotos": subphotos,
        "detections": detections,
        "summary": {
            "people_in_image_count": len(person_names),
            "subject_count": len(subjects),
            "detected_people_count": len(detection_people) if isinstance(detection_people, list) else 0,
            "detected_object_count": len(detection_objects) if isinstance(detection_objects, list) else 0,
            "ocr_char_count": _coerce_int(detection_ocr.get("chars") if isinstance(detection_ocr, dict) else 0),
            "subphoto_count": len(subphotos),
        },
    }
    if include_raw_xml:
        result["raw_xml"] = path.read_text(encoding="utf-8")
    return result
