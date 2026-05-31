from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Namespaces copied from photoalbums/lib/xmp_sidecar.py (stdlib only, no cross-project import)
X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
DC_NS = "http://purl.org/dc/elements/1.1/"
MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"
EXIFTOOL_STAREA_NS = "http://ns.adobe.com/xmp/sType/Area#"
STDIM_NS = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
MP_NS = "http://ns.microsoft.com/photo/1.2/"
MPRI_NS = "http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
MPREG_NS = "http://ns.microsoft.com/photo/1.2/t/Region#"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("mwg-rs", MWGRS_NS)
ET.register_namespace("stArea", STAREA_NS)
ET.register_namespace("stDim", STDIM_NS)
ET.register_namespace("MP", MP_NS)
ET.register_namespace("MPRI", MPRI_NS)
ET.register_namespace("MPReg", MPREG_NS)

_PERSON_TAG = f"{{{IPTC_EXT_NS}}}PersonInImage"
_DC_DESC_TAG = f"{{{DC_NS}}}description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_LI = f"{{{RDF_NS}}}li"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_PARSE_TYPE = f"{{{RDF_NS}}}parseType"

_IPTC_IMAGE_REGION_TAG = f"{{{IPTC_EXT_NS}}}ImageRegion"
_IPTC_REGION_BOUNDARY_TAG = f"{{{IPTC_EXT_NS}}}RegionBoundary"
_IPTC_RCTYPE_TAG = f"{{{IPTC_EXT_NS}}}RCtype"
_IPTC_RID_TAG = f"{{{IPTC_EXT_NS}}}rId"
_IPTC_NAME_TAG = f"{{{IPTC_EXT_NS}}}Name"
_IPTC_RB_SHAPE_TAG = f"{{{IPTC_EXT_NS}}}rbShape"
_IPTC_RB_UNIT_TAG = f"{{{IPTC_EXT_NS}}}rbUnit"
_IPTC_RB_X_TAG = f"{{{IPTC_EXT_NS}}}rbX"
_IPTC_RB_Y_TAG = f"{{{IPTC_EXT_NS}}}rbY"
_IPTC_RB_W_TAG = f"{{{IPTC_EXT_NS}}}rbW"
_IPTC_RB_H_TAG = f"{{{IPTC_EXT_NS}}}rbH"
_MWGRS_REGION_INFO_TAG = f"{{{MWGRS_NS}}}RegionInfo"
_MWGRS_REGIONS_TAG = f"{{{MWGRS_NS}}}Regions"
_MWGRS_APPLIED_TO_DIMENSIONS_TAG = f"{{{MWGRS_NS}}}AppliedToDimensions"
_MWGRS_REGION_LIST_TAG = f"{{{MWGRS_NS}}}RegionList"
_MWGRS_AREA_TAG = f"{{{MWGRS_NS}}}Area"
_MWGRS_TYPE_TAG = f"{{{MWGRS_NS}}}Type"
_MWGRS_NAME_TAG = f"{{{MWGRS_NS}}}Name"
_MWGRS_TYPE_ATTR = f"{{{MWGRS_NS}}}Type"
_MWGRS_NAME_ATTR = f"{{{MWGRS_NS}}}Name"
_STAREA_X_ATTR = f"{{{STAREA_NS}}}x"
_STAREA_Y_ATTR = f"{{{STAREA_NS}}}y"
_STAREA_W_ATTR = f"{{{STAREA_NS}}}w"
_STAREA_H_ATTR = f"{{{STAREA_NS}}}h"
_STAREA_UNIT_ATTR = f"{{{STAREA_NS}}}unit"
_EXIFTOOL_STAREA_X_TAG = f"{{{EXIFTOOL_STAREA_NS}}}x"
_EXIFTOOL_STAREA_Y_TAG = f"{{{EXIFTOOL_STAREA_NS}}}y"
_EXIFTOOL_STAREA_W_TAG = f"{{{EXIFTOOL_STAREA_NS}}}w"
_EXIFTOOL_STAREA_H_TAG = f"{{{EXIFTOOL_STAREA_NS}}}h"
_STDIM_W_ATTR = f"{{{STDIM_NS}}}w"
_STDIM_H_ATTR = f"{{{STDIM_NS}}}h"
_STDIM_UNIT_ATTR = f"{{{STDIM_NS}}}unit"
_MP_REGION_INFO_TAG = f"{{{MP_NS}}}RegionInfo"
_MPRI_REGIONS_TAG = f"{{{MPRI_NS}}}Regions"
_MPREG_PERSON_DISPLAY_NAME_ATTR = f"{{{MPREG_NS}}}PersonDisplayName"
_MPREG_RECTANGLE_ATTR = f"{{{MPREG_NS}}}Rectangle"

_MWGRS_STANDARD_REGION_CHILDREN = {
    _MWGRS_NAME_TAG,
    _MWGRS_TYPE_TAG,
    _MWGRS_AREA_TAG,
}


def _get_rdf_desc(tree: ET.ElementTree) -> ET.Element | None:  # type: ignore[type-arg]
    """Return the rdf:Description element from an XMP tree, or None."""
    root = tree.getroot()
    if root is None:
        return None
    rdf_rdf = root.find(_RDF_ROOT)
    if rdf_rdf is None:
        return None
    return rdf_rdf.find(_RDF_DESC)


def _dedupe(names: list[str]) -> list[str]:
    first_seen: dict[str, str] = {}
    for raw in names:
        clean = str(raw or "").strip()
        if clean:
            first_seen.setdefault(clean.casefold(), clean)
    return list(first_seen.values())


def _normalize_xmp_text(value: str, *, multiline: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace(r"\n", "\n" if multiline else " ")


def read_xmp_description(sidecar_path: Path | str) -> str:
    """Read dc:description alt-text from an XMP sidecar. Returns '' on any error."""
    try:
        path = Path(sidecar_path)
        if not path.is_file():
            return ""
        tree = ET.parse(str(path))
        desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
        if desc is None:
            return ""
        desc_elem = desc.find(_DC_DESC_TAG)
        if desc_elem is None:
            return ""
        alt = desc_elem.find(_RDF_ALT)
        if alt is not None:
            for li in alt.findall(_RDF_LI):
                text = (li.text or "").strip()
                if text:
                    return text
        return (desc_elem.text or "").strip()
    except Exception:
        return ""


def read_person_in_image(sidecar_path: Path) -> list[str]:
    """
    Parse an existing .xmp sidecar and return the Iptc4xmpExt:PersonInImage
    bag as a list of name strings. Returns [] if the file doesn't exist,
    is malformed, or has no people.
    """
    try:
        path = Path(sidecar_path)
        if not path.is_file():
            return []
        tree = ET.parse(str(path))
        desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
        if desc is None:
            return []
        person_elem = desc.find(_PERSON_TAG)
        if person_elem is None:
            return []
        bag = person_elem.find(_RDF_BAG)
        if bag is None:
            return []
        raw_names = [(li.text or "").strip() for li in bag.findall(_RDF_LI)]
        return _dedupe([n for n in raw_names if n])
    except Exception:
        return []


def merge_persons_xmp(
    sidecar_path: Path,
    person_names: list[str],
    *,
    description: str | None = None,
) -> Path:
    """
    Write PersonInImage names to a .xmp sidecar.

    - If the sidecar already exists, only the Iptc4xmpExt:PersonInImage bag is
      updated; all other fields (dc:description, dc:subject, imago:*, etc.) are
      preserved untouched, unless `description` is provided.
    - If the sidecar does not exist, a minimal XMP file is created containing
      only PersonInImage.

    Returns the path that was written.
    """
    sidecar_path = Path(sidecar_path)
    names = _dedupe(person_names)

    if sidecar_path.exists():
        try:
            tree = ET.parse(str(sidecar_path))
        except ET.ParseError as exc:
            # Don't overwrite an existing-but-unparseable sidecar with a persons-only
            # skeleton; that would drop dc:description, imago:*, regions, etc. Surface
            # the error so the corrupt file is preserved for inspection.
            raise RuntimeError(
                f"Refusing to write person names: existing sidecar {sidecar_path} is "
                f"not valid XML ({exc}). Repair or remove it before re-running."
            ) from exc
        _merge_into_tree(tree, names, description=description)  # type: ignore[arg-type]
        tree.write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)
        return sidecar_path

    _write_minimal(sidecar_path, names, description=description)
    return sidecar_path


def _update_dc_description(
    desc: ET.Element,  # type: ignore[type-arg]
    desc_elem: ET.Element | None,  # type: ignore[type-arg]
    clean: str,
) -> None:
    if clean:
        if desc_elem is None:
            desc_elem = ET.SubElement(desc, _DC_DESC_TAG)
        alt = desc_elem.find(_RDF_ALT)
        if alt is None:
            desc_elem.clear()
            alt = ET.SubElement(desc_elem, _RDF_ALT)
        else:
            for li in list(alt.findall(_RDF_LI)):
                alt.remove(li)
        li = ET.SubElement(alt, _RDF_LI)
        li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
        li.text = clean
    elif desc_elem is not None:
        desc.remove(desc_elem)


def _merge_into_tree(
    tree: ET.ElementTree,  # type: ignore[type-arg]
    names: list[str],
    *,
    description: str | None = None,
) -> None:
    """Update or create the PersonInImage bag inside an existing XMP tree."""
    root = tree.getroot()
    assert root is not None

    # Find or create the rdf:RDF element
    rdf_rdf = root.find(_RDF_ROOT)
    if rdf_rdf is None:
        rdf_rdf = ET.SubElement(root, _RDF_ROOT)  # type: ignore[arg-type]

    # Find or create rdf:Description inside rdf:RDF
    desc = rdf_rdf.find(_RDF_DESC)
    if desc is None:
        desc = ET.SubElement(rdf_rdf, _RDF_DESC)

    # Find or create the PersonInImage element
    person_elem = desc.find(_PERSON_TAG)
    if person_elem is None:
        person_elem = ET.SubElement(desc, _PERSON_TAG)

    # Replace the bag contents entirely
    bag = person_elem.find(_RDF_BAG)
    if bag is not None:
        person_elem.remove(bag)
    bag = ET.SubElement(person_elem, _RDF_BAG)
    for name in names:
        li = ET.SubElement(bag, _RDF_LI)
        li.text = name

    # Optionally update the dc:description alt-text
    if description is not None:
        clean = _normalize_xmp_text(description, multiline=True)
        desc_elem = desc.find(_DC_DESC_TAG)
        _update_dc_description(desc, desc_elem, clean)


def _write_minimal(
    sidecar_path: Path,
    names: list[str],
    *,
    description: str | None = None,
) -> None:
    """Create a minimal XMP sidecar with only PersonInImage."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf_rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    rdf_rdf.set("xmlns:rdf", RDF_NS)
    desc = ET.SubElement(rdf_rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")

    person_elem = ET.SubElement(desc, _PERSON_TAG)
    bag = ET.SubElement(person_elem, _RDF_BAG)
    for name in names:
        li = ET.SubElement(bag, _RDF_LI)
        li.text = name

    if description:
        clean = _normalize_xmp_text(description, multiline=True)
        if clean:
            desc_elem = ET.SubElement(desc, _DC_DESC_TAG)
            alt = ET.SubElement(desc_elem, _RDF_ALT)
            li = ET.SubElement(alt, _RDF_LI)
            li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
            li.text = clean

    tree = ET.ElementTree(xmpmeta)
    tree.write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# IPTC ImageRegion face bounding boxes (written by immich-sync)
# ---------------------------------------------------------------------------


def _region_is_face(li: ET.Element) -> bool:  # type: ignore[type-arg]
    rctype = str(li.findtext(_IPTC_RCTYPE_TAG, default="") or "").strip().lower()
    if rctype.startswith("face-"):
        return True
    rid = str(li.findtext(_IPTC_RID_TAG, default="") or "").strip()
    return rid.startswith("face-")


def _add_region_name_alt(parent: ET.Element, name: str) -> None:  # type: ignore[type-arg]
    field = ET.SubElement(parent, _IPTC_NAME_TAG)
    alt = ET.SubElement(field, _RDF_ALT)
    li = ET.SubElement(alt, _RDF_LI)
    li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    li.text = name


def _remove_face_regions(
    ir_elem: ET.Element,  # type: ignore[type-arg]
    desc: ET.Element,  # type: ignore[type-arg]
) -> ET.Element | None:  # type: ignore[type-arg]
    bag = ir_elem.find(_RDF_BAG)
    if bag is not None:
        for li in list(bag.findall(_RDF_LI)):
            if _region_is_face(li):
                bag.remove(li)
        if not list(bag):
            ir_elem.remove(bag)
    if not list(ir_elem):
        desc.remove(ir_elem)
        return None
    return ir_elem


def _valid_face_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in regions
        if str(r.get("name") or "").strip()
        and float(r.get("rw") or 0) > 0
        and float(r.get("rh") or 0) > 0
    ]


def _build_face_region_li(
    bag: ET.Element,  # type: ignore[type-arg]
    n: int,
    region: dict[str, Any],
) -> None:
    name = str(region.get("name") or "").strip()
    li = ET.SubElement(bag, _RDF_LI)
    li.set(_RDF_PARSE_TYPE, "Resource")
    boundary = ET.SubElement(li, _IPTC_REGION_BOUNDARY_TAG)
    boundary.set(_RDF_PARSE_TYPE, "Resource")
    ET.SubElement(boundary, _IPTC_RB_SHAPE_TAG).text = "rectangle"
    ET.SubElement(boundary, _IPTC_RB_UNIT_TAG).text = "relative"
    ET.SubElement(boundary, _IPTC_RB_X_TAG).text = f"{float(region.get('rx', 0)):.6f}"
    ET.SubElement(boundary, _IPTC_RB_Y_TAG).text = f"{float(region.get('ry', 0)):.6f}"
    ET.SubElement(boundary, _IPTC_RB_W_TAG).text = f"{float(region.get('rw', 0)):.6f}"
    ET.SubElement(boundary, _IPTC_RB_H_TAG).text = f"{float(region.get('rh', 0)):.6f}"
    ET.SubElement(li, _IPTC_RCTYPE_TAG).text = "face-identified"
    ET.SubElement(li, _IPTC_RID_TAG).text = f"face-{n}"
    _add_region_name_alt(li, name)


def _set_face_regions_in_desc(
    desc: ET.Element,  # type: ignore[type-arg]
    regions: list[dict[str, Any]],
) -> None:
    ir_elem = desc.find(_IPTC_IMAGE_REGION_TAG)
    if ir_elem is not None:
        ir_elem = _remove_face_regions(ir_elem, desc)

    valid = _valid_face_regions(regions)
    if not valid:
        return

    if ir_elem is None:
        ir_elem = ET.SubElement(desc, _IPTC_IMAGE_REGION_TAG)
    bag = ir_elem.find(_RDF_BAG)
    if bag is None:
        bag = ET.SubElement(ir_elem, _RDF_BAG)

    for n, region in enumerate(valid, 1):
        _build_face_region_li(bag, n, region)


def _region_image_dimensions(regions: list[dict[str, Any]]) -> tuple[int, int]:
    for region in regions:
        image_width = int(region.get("image_width") or 0)
        image_height = int(region.get("image_height") or 0)
        if image_width > 0 and image_height > 0:
            return image_width, image_height
    return 0, 0


def _parse_float(value: object) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return 0.0


def _region_info_dimensions(region_info: ET.Element | None) -> tuple[int, int]:
    if region_info is None:
        return 0, 0
    applied = region_info.find(_MWGRS_APPLIED_TO_DIMENSIONS_TAG)
    if applied is None:
        return 0, 0
    width = int(_parse_float(applied.get(_STAREA_W_ATTR) or applied.findtext(f"{{{STDIM_NS}}}w")))
    height = int(_parse_float(applied.get(_STAREA_H_ATTR) or applied.findtext(f"{{{STDIM_NS}}}h")))
    return width, height


def _photo_region_entry_from_compact(li: ET.Element) -> dict[str, Any] | None:
    if str(li.get(_MWGRS_TYPE_ATTR) or "").strip() != "Photo":
        return None
    x = _parse_float(li.get(_STAREA_X_ATTR))
    y = _parse_float(li.get(_STAREA_Y_ATTR))
    w = _parse_float(li.get(_STAREA_W_ATTR))
    h = _parse_float(li.get(_STAREA_H_ATTR))
    if w <= 0 or h <= 0:
        return None
    return {
        "Name": str(li.get(_MWGRS_NAME_ATTR) or "").strip(),
        "Type": "Photo",
        "Area": {"X": x, "Y": y, "W": w, "H": h, "Unit": "normalized"},
        "_extras": _region_extras(li),
    }


def _photo_region_entry_from_exiftool(li: ET.Element) -> dict[str, Any] | None:
    region_type = str(li.findtext(_MWGRS_TYPE_TAG) or "").strip()
    if region_type not in {"", "Photo"}:
        return None
    area = li.find(_MWGRS_AREA_TAG)
    if area is None:
        return None
    x = _parse_float(area.findtext(_EXIFTOOL_STAREA_X_TAG))
    y = _parse_float(area.findtext(_EXIFTOOL_STAREA_Y_TAG))
    w = _parse_float(area.findtext(_EXIFTOOL_STAREA_W_TAG))
    h = _parse_float(area.findtext(_EXIFTOOL_STAREA_H_TAG))
    if w <= 0 or h <= 0:
        return None
    return {
        "Name": str(li.findtext(_MWGRS_NAME_TAG) or "").strip(),
        "Type": "Photo",
        "Area": {"X": x, "Y": y, "W": w, "H": h, "Unit": "normalized"},
        "_extras": _region_extras(li),
    }


def _region_extras(li: ET.Element) -> dict[str, Any]:
    attrs = {
        key: value
        for key, value in li.attrib.items()
        if key not in {_RDF_PARSE_TYPE, _MWGRS_TYPE_ATTR, _MWGRS_NAME_ATTR, _STAREA_X_ATTR, _STAREA_Y_ATTR, _STAREA_W_ATTR, _STAREA_H_ATTR, _STAREA_UNIT_ATTR}
    }
    children = [child for child in list(li) if child.tag not in _MWGRS_STANDARD_REGION_CHILDREN]
    return {"attrs": attrs, "children": children}


def _existing_photo_region_entries(sidecar_path: Path) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    if not sidecar_path.exists():
        return [], (0, 0)
    try:
        tree = ET.parse(str(sidecar_path))
    except ET.ParseError:
        return [], (0, 0)
    root = tree.getroot()
    rdf_rdf = root.find(_RDF_ROOT)
    if rdf_rdf is None:
        return [], (0, 0)
    entries: list[dict[str, Any]] = []
    dimensions = (0, 0)
    for desc in rdf_rdf.findall(_RDF_DESC):
        found_entries, found_dimensions = _photo_entries_from_desc(desc, canonical=True)
        if not found_entries:
            found_entries, found_dimensions = _photo_entries_from_desc(desc, canonical=False)
        entries.extend(found_entries)
        if dimensions == (0, 0):
            dimensions = found_dimensions
    return entries, dimensions


def _photo_entries_from_desc(desc: ET.Element, *, canonical: bool) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    region_info = desc.find(_MWGRS_REGIONS_TAG if canonical else _MWGRS_REGION_INFO_TAG)
    if region_info is None:
        return [], (0, 0)
    region_list = region_info.find(_MWGRS_REGION_LIST_TAG)
    bag = region_list.find(_RDF_BAG) if region_list is not None else None
    if bag is None:
        return [], _region_info_dimensions(region_info)
    entries = [_photo_entry_from_li(li, canonical=canonical) for li in bag.findall(_RDF_LI)]
    return [entry for entry in entries if entry is not None], _region_info_dimensions(region_info)


def _photo_entry_from_li(li: ET.Element, *, canonical: bool) -> dict[str, Any] | None:
    if canonical:
        return _photo_region_entry_from_exiftool(li)
    return _photo_region_entry_from_compact(li)


def _entry_for_exiftool(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _restore_photo_region_extras(target_regions: list[ET.Element], photo_entries: list[dict[str, Any]]) -> None:
    for li, entry in zip(target_regions, photo_entries, strict=False):
        if li.find(_MWGRS_TYPE_TAG) is None:
            ET.SubElement(li, _MWGRS_TYPE_TAG).text = "Photo"
        extras = dict(entry.get("_extras") or {})
        for key, value in dict(extras.get("attrs") or {}).items():
            li.set(str(key), str(value))
        for child in list(extras.get("children") or []):
            li.append(child)


def _is_canonical_photo_li(li: ET.Element) -> bool:
    region_type = str(li.findtext(_MWGRS_TYPE_TAG) or "").strip()
    return region_type in {"", "Photo"} and li.find(_MWGRS_AREA_TAG) is not None


_MINIMAL_XMP = (
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
    "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
    "<rdf:Description rdf:about=''/>"
    "</rdf:RDF>"
    "</x:xmpmeta>"
)


def write_face_regions_exiftool(
    sidecar_path: Path,
    regions: list[dict[str, Any]],
) -> None:
    """Write MWG-RS and MP face regions via ExifTool (canonical format).

    Replaces XMP-mwg-rs:RegionInfo (mwg-rs:Regions in XMP) and XMP-MP:RegionInfoMP
    entirely. Pass an empty list to clear existing regions.

    regions: list of {"name", "rx", "ry", "rw", "rh", "image_width", "image_height"}
      where coordinates are normalized (0.0-1.0), origin at top-left.

    ExifTool is run on a fresh temp XMP file (not the main sidecar) to avoid
    ExifTool restructuring the sidecar's RDF into multiple rdf:Description blocks.
    The face-region elements are then injected into the main sidecar's primary
    description via Python's ElementTree.
    """
    import json
    import subprocess
    import tempfile

    valid = _valid_face_regions(regions)
    photo_entries, photo_dimensions = _existing_photo_region_entries(sidecar_path)
    image_width, image_height = _region_image_dimensions(valid)
    if image_width <= 0 or image_height <= 0:
        image_width, image_height = photo_dimensions

    entry: dict[str, Any] = {}

    if (photo_entries or valid) and image_width > 0 and image_height > 0:
        region_list = [_entry_for_exiftool(photo_entry) for photo_entry in photo_entries]
        mp_regions = []
        for r in valid:
            rx = float(r.get("rx") or 0)
            ry = float(r.get("ry") or 0)
            rw = float(r.get("rw") or 0)
            rh = float(r.get("rh") or 0)
            name = str(r.get("name") or "").strip()
            region_list.append({
                "Name": name,
                "Type": "Face",
                "Area": {
                    "X": round(rx + rw / 2.0, 6),
                    "Y": round(ry + rh / 2.0, 6),
                    "W": round(rw, 6),
                    "H": round(rh, 6),
                    "Unit": "normalized",
                },
            })
            mp_regions.append({
                "PersonDisplayName": name,
                "Rectangle": f"{rx:.6f}, {ry:.6f}, {rw:.6f}, {rh:.6f}",
            })
        entry["XMP-mwg-rs:RegionInfo"] = {
            "AppliedToDimensions": {"W": image_width, "H": image_height, "Unit": "pixel"},
            "RegionList": region_list,
        }
        entry["XMP-MP:RegionInfoMP"] = {"Regions": mp_regions}
    else:
        entry["XMP-mwg-rs:RegionInfo"] = ""
        entry["XMP-MP:RegionInfoMP"] = ""

    # Run ExifTool on a minimal temp XMP so ExifTool doesn't restructure the main
    # sidecar into multiple rdf:Description blocks (which breaks read_ai_sidecar_state).
    with tempfile.NamedTemporaryFile(suffix=".xmp", mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(_MINIMAL_XMP)
        temp_xmp = Path(tf.name)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json_path = Path(f.name)
        json.dump([entry], f, ensure_ascii=False)

    try:
        result = subprocess.run(
            ["exiftool", "-struct", "-overwrite_original", f"-json={json_path}", str(temp_xmp)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ExifTool failed writing face regions for {sidecar_path}: {result.stderr.strip()}"
            )
        _inject_face_region_elements(sidecar_path, temp_xmp, photo_entries=photo_entries)
    finally:
        json_path.unlink(missing_ok=True)
        temp_xmp.unlink(missing_ok=True)


def _extract_exiftool_face_elements(
    exiftool_xmp: Path,
) -> tuple[ET.Element | None, ET.Element | None]:
    """Return (mwg-rs:Regions, MP:RegionInfo) from an ExifTool-written XMP, or (None, None)."""
    et_rdf = ET.parse(str(exiftool_xmp)).getroot().find(f"{{{RDF_NS}}}RDF")
    if et_rdf is None:
        return None, None
    mwgrs_el: ET.Element | None = None
    mp_el: ET.Element | None = None
    for desc in et_rdf.findall(f"{{{RDF_NS}}}Description"):
        if mwgrs_el is None:
            mwgrs_el = desc.find(f"{{{MWGRS_NS}}}Regions")
        if mp_el is None:
            mp_el = desc.find(f"{{{MP_NS}}}RegionInfo")
    return mwgrs_el, mp_el


def _inject_face_region_elements(
    sidecar_path: Path,
    exiftool_xmp: Path,
    *,
    photo_entries: list[dict[str, Any]] | None = None,
) -> None:
    """Inject mwg-rs:Regions and MP:RegionInfo from ExifTool-written XMP into the
    primary rdf:Description of the main sidecar (preserves all other properties)."""
    mwgrs_el, mp_el = _extract_exiftool_face_elements(exiftool_xmp)

    if not sidecar_path.exists():
        return
    main_tree = ET.parse(str(sidecar_path))
    main_rdf = main_tree.getroot().find(f"{{{RDF_NS}}}RDF")
    if main_rdf is None:
        return
    main_desc = main_rdf.find(f"{{{RDF_NS}}}Description")
    if main_desc is None:
        return

    # Remove existing region elements from all descriptions; the injected
    # mwg-rs:Regions block already includes preserved Photo entries.
    for tag in (f"{{{MWGRS_NS}}}Regions", f"{{{MWGRS_NS}}}RegionInfo", f"{{{MP_NS}}}RegionInfo"):
        for d in main_rdf.findall(f"{{{RDF_NS}}}Description"):
            old = d.find(tag)
            if old is not None:
                d.remove(old)

    # Only inject populated (non-empty) elements
    if mwgrs_el is not None and len(mwgrs_el) > 0:
        main_desc.append(mwgrs_el)
        if photo_entries:
            photo_lis = [
                li
                for li in mwgrs_el.findall(f".//{{{RDF_NS}}}li")
                if _is_canonical_photo_li(li)
            ]
            _restore_photo_region_extras(photo_lis, photo_entries)
    if mp_el is not None and len(mp_el) > 0:
        main_desc.append(mp_el)

    ET.indent(main_tree, space="  ")
    main_tree.write(str(sidecar_path), encoding="utf-8", xml_declaration=True)


def merge_face_regions_xmp(
    sidecar_path: Path,
    regions: list[dict[str, Any]],
) -> Path:
    """Write face bounding box regions to an XMP sidecar.

    Writes IPTC4xmpExt:ImageRegion via XML and MWG-RS / MP regions via
    ExifTool (canonical format). Replaces existing face regions; preserves
    all other XMP fields and non-face IPTC region types. Creates a minimal
    sidecar if none exists.

    regions: list of {"name", "rx", "ry", "rw", "rh", "image_width", "image_height"}
      where coordinates are normalized (0.0-1.0), origin at top-left.

    Returns the path written.
    """
    sidecar_path = Path(sidecar_path)
    tree: ET.ElementTree | None = None  # type: ignore[type-arg]

    if sidecar_path.exists():
        try:
            tree = ET.parse(str(sidecar_path))
        except ET.ParseError as exc:
            # Never silently rebuild an existing-but-unparseable sidecar from scratch:
            # the minimal skeleton below would carry only face regions and drop every
            # other field (dc:description, exif GPS/date, imago:*). Surface the error
            # so the corrupt file is preserved for inspection instead of overwritten.
            raise RuntimeError(
                f"Refusing to write face regions: existing sidecar {sidecar_path} is "
                f"not valid XML ({exc}). Repair or remove it before re-running."
            ) from exc

    if tree is not None:
        root = tree.getroot()
        if root is not None:
            rdf_rdf = root.find(_RDF_ROOT)
            if rdf_rdf is None:
                rdf_rdf = ET.SubElement(root, _RDF_ROOT)
            desc = rdf_rdf.find(_RDF_DESC)
            if desc is None:
                desc = ET.SubElement(rdf_rdf, _RDF_DESC)
            _set_face_regions_in_desc(desc, regions)
            tree.write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)
            write_face_regions_exiftool(sidecar_path, regions)
            return sidecar_path

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf_rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    rdf_rdf.set("xmlns:rdf", RDF_NS)
    desc = ET.SubElement(rdf_rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")
    _set_face_regions_in_desc(desc, regions)
    ET.ElementTree(xmpmeta).write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)
    write_face_regions_exiftool(sidecar_path, regions)
    return sidecar_path
