from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ._caption_album import dedupe as _dedupe

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
EXIF_NS = "http://ns.adobe.com/exif/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
IMAGO_NS = "https://imago.local/ns/1.0/"
MWG_RS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
ST_AREA_NS = "http://ns.adobe.com/xmp/schemata/area/"
ST_DIM_NS = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
PHOTOSHOP_NS = "http://ns.adobe.com/photoshop/1.0/"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("exif", EXIF_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("imago", IMAGO_NS)
ET.register_namespace("mwg-rs", MWG_RS_NS)
ET.register_namespace("stArea", ST_AREA_NS)
ET.register_namespace("stDim", ST_DIM_NS)
ET.register_namespace("photoshop", PHOTOSHOP_NS)

_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_SEQ = f"{{{RDF_NS}}}Seq"
_RDF_LI = f"{{{RDF_NS}}}li"


def _add_bag(parent: ET.Element, tag: str, values: list[str]) -> None:
    if not values:
        return
    field = ET.SubElement(parent, tag)
    bag = ET.SubElement(field, _RDF_BAG)
    for value in values:
        item = ET.SubElement(bag, _RDF_LI)
        item.text = value


def _add_alt_text(parent: ET.Element, tag: str, value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    field = ET.SubElement(parent, tag)
    alt = ET.SubElement(field, _RDF_ALT)
    item = ET.SubElement(alt, _RDF_LI)
    item.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    item.text = text


def _add_simple_text(parent: ET.Element, tag: str, value: str | int | float) -> None:
    text = str(value)
    field = ET.SubElement(parent, tag)
    field.text = text


def _format_xmp_gps_coordinate(value: str | float | int, *, axis: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    decimal = float(text)
    if axis == "lat":
        hemisphere = "N" if decimal >= 0 else "S"
    else:
        hemisphere = "E" if decimal >= 0 else "W"
    absolute = abs(decimal)
    degrees = int(absolute)
    minutes = (absolute - float(degrees)) * 60.0
    minute_text = f"{minutes:.5f}".rstrip("0").rstrip(".")
    if not minute_text:
        minute_text = "0"
    return f"{degrees},{minute_text}{hemisphere}"


def _set_gps_fields(parent: ET.Element, gps_latitude: str, gps_longitude: str) -> None:
    lat_text = str(gps_latitude or "").strip()
    lon_text = str(gps_longitude or "").strip()
    if lat_text and lon_text:
        _set_simple_text(
            parent,
            f"{{{EXIF_NS}}}GPSLatitude",
            _format_xmp_gps_coordinate(lat_text, axis="lat"),
        )
        _set_simple_text(
            parent,
            f"{{{EXIF_NS}}}GPSLongitude",
            _format_xmp_gps_coordinate(lon_text, axis="lon"),
        )
        _set_simple_text(parent, f"{{{EXIF_NS}}}GPSMapDatum", "WGS-84")
        _set_simple_text(parent, f"{{{EXIF_NS}}}GPSVersionID", "2.3.0.0")
        return
    for tag in (
        f"{{{EXIF_NS}}}GPSLatitude",
        f"{{{EXIF_NS}}}GPSLongitude",
        f"{{{EXIF_NS}}}GPSMapDatum",
        f"{{{EXIF_NS}}}GPSVersionID",
    ):
        existing = parent.find(tag)
        if existing is not None:
            parent.remove(existing)


def _add_face_regions(
    parent: ET.Element,
    people: list[dict],
    image_width: int,
    image_height: int,
    subphotos: list[dict] | None = None,
) -> None:
    if image_width <= 0 or image_height <= 0:
        return
    region_entries: list[tuple[str, str, str, float, float, float, float]] = []
    for person in people:
        name = str(person.get("name") or "").strip()
        bbox = list(person.get("bbox") or [])
        if not name or len(bbox) < 4:
            continue
        x, y, w, h = [int(v) for v in bbox[:4]]
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / image_width
        cy = (y + h / 2) / image_height
        nw = w / image_width
        nh = h / image_height
        region_entries.append((name, "Face", "", cx, cy, nw, nh))
    for row in subphotos or []:
        bounds = dict(row.get("bounds") or {})
        bx = int(bounds.get("x", 0))
        by = int(bounds.get("y", 0))
        bw = int(bounds.get("width", 0))
        bh = int(bounds.get("height", 0))
        if bw <= 0 or bh <= 0:
            continue
        cx = (bx + bw / 2) / image_width
        cy = (by + bh / 2) / image_height
        nw = bw / image_width
        nh = bh / image_height
        idx = int(row.get("index", 0))
        name = f"Photo {idx}" if idx > 0 else "Photo"
        desc = str(row.get("author_text") or row.get("description") or "").strip()
        region_entries.append((name, "Photo", desc, cx, cy, nw, nh))
    if not region_entries:
        return
    ri = ET.SubElement(parent, f"{{{MWG_RS_NS}}}RegionInfo")
    ri.set(f"{{{RDF_NS}}}parseType", "Resource")
    dims = ET.SubElement(ri, f"{{{MWG_RS_NS}}}AppliedToDimensions")
    dims.set(f"{{{RDF_NS}}}parseType", "Resource")
    ET.SubElement(dims, f"{{{ST_DIM_NS}}}w").text = str(image_width)
    ET.SubElement(dims, f"{{{ST_DIM_NS}}}h").text = str(image_height)
    ET.SubElement(dims, f"{{{ST_DIM_NS}}}unit").text = "pixel"
    region_list = ET.SubElement(ri, f"{{{MWG_RS_NS}}}RegionList")
    bag = ET.SubElement(region_list, _RDF_BAG)
    for name, rtype, desc, cx, cy, nw, nh in region_entries:
        li = ET.SubElement(bag, _RDF_LI)
        li.set(f"{{{RDF_NS}}}parseType", "Resource")
        ET.SubElement(li, f"{{{MWG_RS_NS}}}Name").text = name
        ET.SubElement(li, f"{{{MWG_RS_NS}}}Type").text = rtype
        if desc:
            ET.SubElement(li, f"{{{MWG_RS_NS}}}Description").text = desc
        area = ET.SubElement(li, f"{{{MWG_RS_NS}}}Area")
        area.set(f"{{{RDF_NS}}}parseType", "Resource")
        ET.SubElement(area, f"{{{ST_AREA_NS}}}x").text = f"{cx:.6f}"
        ET.SubElement(area, f"{{{ST_AREA_NS}}}y").text = f"{cy:.6f}"
        ET.SubElement(area, f"{{{ST_AREA_NS}}}w").text = f"{nw:.6f}"
        ET.SubElement(area, f"{{{ST_AREA_NS}}}h").text = f"{nh:.6f}"
        ET.SubElement(area, f"{{{ST_AREA_NS}}}unit").text = "normalized"


def _set_face_regions(
    parent: ET.Element,
    people: list[dict],
    image_width: int,
    image_height: int,
    subphotos: list[dict] | None = None,
) -> None:
    existing = parent.find(f"{{{MWG_RS_NS}}}RegionInfo")
    if existing is not None:
        parent.remove(existing)
    _add_face_regions(parent, people, image_width, image_height, subphotos=subphotos)


def _add_iptc_image_regions(
    parent: ET.Element,
    subphotos: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    """Write Iptc4xmpExt:ImageRegion entries for photo subregions (IPTC standard)."""
    if not subphotos or image_width <= 0 or image_height <= 0:
        return
    field = ET.SubElement(parent, f"{{{IPTC_EXT_NS}}}ImageRegion")
    bag = ET.SubElement(field, _RDF_BAG)
    for row in subphotos:
        bounds = dict(row.get("bounds") or {})
        bx = int(bounds.get("x", 0))
        by = int(bounds.get("y", 0))
        bw = int(bounds.get("width", 0))
        bh = int(bounds.get("height", 0))
        if bw <= 0 or bh <= 0:
            continue
        rx = bx / image_width
        ry = by / image_height
        rw = bw / image_width
        rh = bh / image_height
        idx = int(row.get("index", 0))
        li = ET.SubElement(bag, _RDF_LI)
        li.set(f"{{{RDF_NS}}}parseType", "Resource")
        boundary = ET.SubElement(li, f"{{{IPTC_EXT_NS}}}RegionBoundary")
        boundary.set(f"{{{RDF_NS}}}parseType", "Resource")
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbShape").text = "rectangle"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbUnit").text = "relative"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbX").text = f"{rx:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbY").text = f"{ry:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbW").text = f"{rw:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbH").text = f"{rh:.6f}"
        ET.SubElement(li, f"{{{IPTC_EXT_NS}}}rId").text = f"photo-{idx}" if idx > 0 else "photo"
        author_text = str(row.get("author_text") or row.get("description") or "").strip()
        scene_text = str(row.get("scene_text") or "").strip()
        annotation_scope = str(row.get("annotation_scope") or "").strip()
        if author_text:
            _add_alt_text(li, f"{{{DC_NS}}}description", author_text)
        _add_simple_text(li, f"{{{IMAGO_NS}}}OCRText", str(row.get("ocr_text") or "").strip())
        _add_simple_text(li, f"{{{IMAGO_NS}}}AuthorText", author_text)
        _add_simple_text(li, f"{{{IMAGO_NS}}}SceneText", scene_text)
        _add_simple_text(li, f"{{{IMAGO_NS}}}AnnotationScope", annotation_scope)
        _add_bag(li, f"{{{IMAGO_NS}}}People", _dedupe(list(row.get("people") or [])))
        _add_bag(li, f"{{{IMAGO_NS}}}Subjects", _dedupe(list(row.get("subjects") or [])))
        detections = row.get("detections")
        if isinstance(detections, dict):
            _add_simple_text(
                li,
                f"{{{IMAGO_NS}}}Detections",
                json.dumps(detections, ensure_ascii=False, sort_keys=True),
            )


def _set_iptc_image_regions(
    parent: ET.Element,
    subphotos: list[dict] | None,
    image_width: int,
    image_height: int,
) -> None:
    existing = parent.find(f"{{{IPTC_EXT_NS}}}ImageRegion")
    if existing is not None:
        parent.remove(existing)
    if subphotos:
        _add_iptc_image_regions(parent, subphotos, image_width, image_height)


def _add_subphotos(parent: ET.Element, subphotos: list[dict]) -> None:
    if not subphotos:
        return
    field = ET.SubElement(parent, f"{{{IMAGO_NS}}}SubPhotos")
    seq = ET.SubElement(field, _RDF_SEQ)
    for row in subphotos:
        item = ET.SubElement(seq, _RDF_LI)
        _add_simple_text(item, f"{{{IMAGO_NS}}}Index", int(row.get("index", 0) or 0))
        bounds = dict(row.get("bounds") or {})
        _add_simple_text(item, f"{{{IMAGO_NS}}}X", int(bounds.get("x", 0) or 0))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Y", int(bounds.get("y", 0) or 0))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Width", int(bounds.get("width", 0) or 0))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Height", int(bounds.get("height", 0) or 0))
        author_text = str(row.get("author_text") or row.get("description") or "").strip()
        scene_text = str(row.get("scene_text") or "").strip()
        annotation_scope = str(row.get("annotation_scope") or "").strip()
        _add_alt_text(item, f"{{{IMAGO_NS}}}Description", author_text)
        _add_simple_text(item, f"{{{IMAGO_NS}}}OCRText", str(row.get("ocr_text") or "").strip())
        _add_simple_text(item, f"{{{IMAGO_NS}}}AuthorText", author_text)
        _add_simple_text(item, f"{{{IMAGO_NS}}}SceneText", scene_text)
        _add_simple_text(item, f"{{{IMAGO_NS}}}AnnotationScope", annotation_scope)
        _add_bag(item, f"{{{IMAGO_NS}}}People", _dedupe(list(row.get("people") or [])))
        _add_bag(item, f"{{{IMAGO_NS}}}Subjects", _dedupe(list(row.get("subjects") or [])))
        detections = row.get("detections")
        if isinstance(detections, dict):
            _add_simple_text(
                item,
                f"{{{IMAGO_NS}}}Detections",
                json.dumps(detections, ensure_ascii=False, sort_keys=True),
            )


def _set_subphotos(parent: ET.Element, subphotos: list[dict] | None) -> None:
    existing = parent.find(f"{{{IMAGO_NS}}}SubPhotos")
    if existing is not None:
        parent.remove(existing)
    if subphotos:
        _add_subphotos(parent, subphotos)


def build_xmp_tree(
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    source_text: str,
    ocr_text: str,
    author_text: str = "",
    scene_text: str = "",
    annotation_scope: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    image_width: int = 0,
    image_height: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
) -> ET.ElementTree:
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    desc = ET.SubElement(rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")

    _add_bag(desc, f"{{{DC_NS}}}subject", _dedupe(subjects))
    _add_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", _dedupe(person_names))
    _add_alt_text(desc, f"{{{DC_NS}}}title", title)
    _add_alt_text(desc, f"{{{DC_NS}}}description", description)
    if str(album_title or "").strip():
        _add_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    if str(gps_latitude or "").strip() and str(gps_longitude or "").strip():
        _add_simple_text(
            desc,
            f"{{{EXIF_NS}}}GPSLatitude",
            _format_xmp_gps_coordinate(gps_latitude, axis="lat"),
        )
        _add_simple_text(
            desc,
            f"{{{EXIF_NS}}}GPSLongitude",
            _format_xmp_gps_coordinate(gps_longitude, axis="lon"),
        )
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSMapDatum", "WGS-84")
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSVersionID", "2.3.0.0")
    if str(location_city or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city).strip())
    if str(location_state or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state).strip())
    if str(location_country or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country).strip())
    _add_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())

    creator = ET.SubElement(desc, f"{{{XMP_NS}}}CreatorTool")
    creator.text = str(creator_tool or "").strip() or "https://github.com/cove/imago"

    clean_ocr = str(ocr_text or "").strip()
    if clean_ocr:
        ocr = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRText")
        ocr.text = clean_ocr
    clean_author_text = str(author_text or "").strip()
    if clean_author_text:
        author = ET.SubElement(desc, f"{{{IMAGO_NS}}}AuthorText")
        author.text = clean_author_text
    clean_scene_text = str(scene_text or "").strip()
    if clean_scene_text:
        scene = ET.SubElement(desc, f"{{{IMAGO_NS}}}SceneText")
        scene.text = clean_scene_text
    clean_annotation_scope = str(annotation_scope or "").strip()
    if clean_annotation_scope:
        scope = ET.SubElement(desc, f"{{{IMAGO_NS}}}AnnotationScope")
        scope.text = clean_annotation_scope
    clean_title_source = str(title_source or "").strip()
    if clean_title_source:
        title_src = ET.SubElement(desc, f"{{{IMAGO_NS}}}TitleSource")
        title_src.text = clean_title_source
    clean_ocr_authority_source = str(ocr_authority_source or "").strip()
    if clean_ocr_authority_source:
        ocr_source = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRAuthoritySource")
        ocr_source.text = clean_ocr_authority_source

    if detections_payload:
        payload = ET.SubElement(desc, f"{{{IMAGO_NS}}}Detections")
        payload.text = json.dumps(detections_payload, ensure_ascii=False, sort_keys=True)
    _add_face_regions(
        desc,
        list((detections_payload or {}).get("people") or []),
        image_width,
        image_height,
        subphotos=list(subphotos) if subphotos else None,
    )
    _add_iptc_image_regions(desc, list(subphotos) if subphotos else [], image_width, image_height)
    if subphotos and (image_width <= 0 or image_height <= 0):
        _add_subphotos(desc, list(subphotos))
    clean_stitch_key = str(stitch_key or "").strip()
    if clean_stitch_key:
        sk = ET.SubElement(desc, f"{{{IMAGO_NS}}}StitchKey")
        sk.text = clean_stitch_key

    _add_simple_text(desc, f"{{{IMAGO_NS}}}OcrRan", str(ocr_ran).lower())
    _add_simple_text(desc, f"{{{IMAGO_NS}}}PeopleDetected", str(people_detected).lower())
    _add_simple_text(desc, f"{{{IMAGO_NS}}}PeopleIdentified", str(people_identified).lower())

    tree = ET.ElementTree(xmpmeta)
    ET.indent(tree, space="  ")
    return tree


def _get_or_create_rdf_desc(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    assert root is not None
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        rdf = ET.SubElement(root, _RDF_ROOT)  # type: ignore[arg-type]
    desc = rdf.find(_RDF_DESC)
    if desc is None:
        desc = ET.SubElement(rdf, _RDF_DESC)
        desc.set(f"{{{RDF_NS}}}about", "")
    elif f"{{{RDF_NS}}}about" not in desc.attrib:
        desc.set(f"{{{RDF_NS}}}about", "")
    return desc


def _replace_field(parent: ET.Element, tag: str, builder) -> None:
    existing = parent.find(tag)
    if existing is not None:
        parent.remove(existing)
    field = ET.SubElement(parent, tag)
    builder(field)


def _set_bag(parent: ET.Element, tag: str, values: list[str]) -> None:
    clean = _dedupe(values)
    existing = parent.find(tag)
    if not clean:
        if existing is not None:
            parent.remove(existing)
        return

    def _builder(field: ET.Element) -> None:
        bag = ET.SubElement(field, _RDF_BAG)
        for value in clean:
            item = ET.SubElement(bag, _RDF_LI)
            item.text = value

    _replace_field(parent, tag, _builder)


def _set_alt_text(parent: ET.Element, tag: str, value: str) -> None:
    text = str(value or "").strip()
    existing = parent.find(tag)
    if not text:
        if existing is not None:
            parent.remove(existing)
        return

    def _builder(field: ET.Element) -> None:
        alt = ET.SubElement(field, _RDF_ALT)
        item = ET.SubElement(alt, _RDF_LI)
        item.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
        item.text = text

    _replace_field(parent, tag, _builder)


def _set_simple_text(parent: ET.Element, tag: str, value: str | int | float, *, allow_empty: bool = False) -> None:
    text = str(value or "").strip() if isinstance(value, str) else str(value)
    existing = parent.find(tag)
    if not text and not allow_empty:
        if existing is not None:
            parent.remove(existing)
        return
    if existing is None:
        existing = ET.SubElement(parent, tag)
    existing.text = text


def _get_rdf_desc(tree: ET.ElementTree) -> ET.Element | None:
    root = tree.getroot()
    assert root is not None
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        return None
    return rdf.find(_RDF_DESC)


def _get_alt_text(parent: ET.Element, tag: str) -> str:
    field = parent.find(tag)
    if field is None:
        return ""
    alt = field.find(_RDF_ALT)
    if alt is None:
        return ""
    for item in alt.findall(_RDF_LI):
        text = str(item.text or "").strip()
        if text:
            return text
    return ""


def _read_xmp_bool(desc: ET.Element, tag: str) -> bool | None:
    """Return True/False if the tag is present with a boolean value, else None if absent."""
    raw = desc.findtext(tag)
    if raw is None:
        return None
    return str(raw or "").strip().lower() == "true"


def read_person_in_image(sidecar_path: str | Path) -> list[str]:
    """Return Iptc4xmpExt:PersonInImage names from an XMP sidecar. Returns [] on any error."""
    _PERSON_TAG = f"{{{IPTC_EXT_NS}}}PersonInImage"
    try:
        path = Path(sidecar_path)
        if not path.is_file():
            return []
        tree = ET.parse(path)
        desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
        if desc is None:
            return []
        names: list[str] = []
        person_elem = desc.find(_PERSON_TAG)
        if person_elem is None:
            return []
        bag = person_elem.find(_RDF_BAG)
        if bag is None:
            return []
        for li in bag.findall(_RDF_LI):
            text = (li.text or "").strip()
            if text:
                names.append(text)
        return _dedupe(names)
    except Exception:
        return []


def read_ai_sidecar_state(sidecar_path: str | Path) -> dict[str, object] | None:
    path = Path(sidecar_path)
    if not path.is_file():
        return None
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
    if desc is None:
        return None
    detections_text = str(desc.findtext(f"{{{IMAGO_NS}}}Detections", default="") or "").strip()
    detections_payload: dict[str, object] | None = None
    if detections_text:
        try:
            parsed = json.loads(detections_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            detections_payload = parsed
    return {
        "creator_tool": str(desc.findtext(f"{{{XMP_NS}}}CreatorTool", default="") or "").strip(),
        "title": _get_alt_text(desc, f"{{{DC_NS}}}title"),
        "description": _get_alt_text(desc, f"{{{DC_NS}}}description"),
        "album_title": str(desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="") or "").strip(),
        "gps_latitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or "").strip(),
        "gps_longitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or "").strip(),
        "ocr_text": str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "").strip(),
        "author_text": str(desc.findtext(f"{{{IMAGO_NS}}}AuthorText", default="") or "").strip(),
        "scene_text": str(desc.findtext(f"{{{IMAGO_NS}}}SceneText", default="") or "").strip(),
        "annotation_scope": str(desc.findtext(f"{{{IMAGO_NS}}}AnnotationScope", default="") or "").strip(),
        "title_source": str(desc.findtext(f"{{{IMAGO_NS}}}TitleSource", default="") or "").strip(),
        "ocr_authority_source": str(desc.findtext(f"{{{IMAGO_NS}}}OCRAuthoritySource", default="") or "").strip(),
        "stitch_key": str(desc.findtext(f"{{{IMAGO_NS}}}StitchKey", default="") or "").strip(),
        "detections": detections_payload,
        "ocr_ran": _read_xmp_bool(desc, f"{{{IMAGO_NS}}}OcrRan"),
        "people_detected": _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleDetected"),
        "people_identified": _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleIdentified"),
    }


def sidecar_has_expected_ai_fields(
    sidecar_path: str | Path,
    *,
    creator_tool: str,
    enable_people: bool,
    enable_objects: bool,
    ocr_engine: str,
    caption_engine: str,
) -> bool:
    state = read_ai_sidecar_state(sidecar_path)
    if not isinstance(state, dict):
        return False
    if str(state.get("creator_tool") or "").strip() != str(creator_tool or "").strip():
        return False
    detections = state.get("detections")
    if not isinstance(detections, dict):
        return False
    if bool(enable_people) and not isinstance(detections.get("people"), list):
        return False
    if bool(enable_people) and isinstance(detections.get("people"), list) and detections["people"]:
        if not any(
            isinstance(p, dict) and isinstance(p.get("bbox"), list) and len(p["bbox"]) >= 4
            for p in detections["people"]
        ):
            return False
    if bool(enable_objects) and not isinstance(detections.get("objects"), list):
        return False
    if str(ocr_engine or "").strip().lower() != "none" and not isinstance(detections.get("ocr"), dict):
        return False
    caption_name = str(caption_engine or "").strip().lower()
    if caption_name != "none" and not isinstance(detections.get("caption"), dict):
        return False
    description = str(state.get("description") or "").strip()
    if description:
        try:
            from .ai_caption import (
                _looks_like_reasoning_or_prompt_echo,
            )  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - defensive import fallback
            _looks_like_reasoning_or_prompt_echo = None
        if _looks_like_reasoning_or_prompt_echo is not None and _looks_like_reasoning_or_prompt_echo(description):
            return False
    ocr_text = str(state.get("ocr_text") or "").strip()
    if ocr_text:
        try:
            from .ai_ocr import (
                _looks_like_ocr_reasoning,
            )  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - defensive import fallback
            _looks_like_ocr_reasoning = None
        if _looks_like_ocr_reasoning is not None and _looks_like_ocr_reasoning(ocr_text):
            return False
    for field_name in ("author_text", "scene_text"):
        field_value = str(state.get(field_name) or "").strip()
        if not field_value:
            continue
        try:
            from .ai_caption import (
                _looks_like_reasoning_or_prompt_echo,
            )  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - defensive import fallback
            _looks_like_reasoning_or_prompt_echo = None
        if _looks_like_reasoning_or_prompt_echo is not None and _looks_like_reasoning_or_prompt_echo(field_value):
            return False
    return True


def _merge_xmp_tree(
    tree: ET.ElementTree,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    source_text: str,
    ocr_text: str,
    author_text: str = "",
    scene_text: str = "",
    annotation_scope: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    image_width: int = 0,
    image_height: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
) -> ET.ElementTree:
    desc = _get_or_create_rdf_desc(tree)
    _set_bag(desc, f"{{{DC_NS}}}subject", subjects)
    _set_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", person_names)
    _set_alt_text(desc, f"{{{DC_NS}}}title", title)
    _set_alt_text(desc, f"{{{DC_NS}}}description", description)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    _set_gps_fields(desc, gps_latitude, gps_longitude)
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())
    _set_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())
    _set_simple_text(
        desc,
        f"{{{XMP_NS}}}CreatorTool",
        str(creator_tool or "").strip() or "https://github.com/cove/imago",
    )
    clean_ocr = str(ocr_text or "").strip()
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRText", clean_ocr)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AuthorText", str(author_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}SceneText", str(scene_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AnnotationScope", str(annotation_scope or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}TitleSource", str(title_source or "").strip())
    _set_simple_text(
        desc,
        f"{{{IMAGO_NS}}}OCRAuthoritySource",
        str(ocr_authority_source or "").strip(),
    )
    if detections_payload:
        _set_simple_text(
            desc,
            f"{{{IMAGO_NS}}}Detections",
            json.dumps(detections_payload, ensure_ascii=False, sort_keys=True),
        )
    else:
        _set_simple_text(desc, f"{{{IMAGO_NS}}}Detections", "")
    _set_face_regions(
        desc,
        list((detections_payload or {}).get("people") or []),
        image_width,
        image_height,
        subphotos=list(subphotos) if subphotos else None,
    )
    _set_iptc_image_regions(desc, list(subphotos) if subphotos else None, image_width, image_height)
    _set_subphotos(desc, list(subphotos) if subphotos and (image_width <= 0 or image_height <= 0) else None)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}StitchKey", str(stitch_key or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OcrRan", str(ocr_ran).lower(), allow_empty=True)
    _set_simple_text(
        desc,
        f"{{{IMAGO_NS}}}PeopleDetected",
        str(people_detected).lower(),
        allow_empty=True,
    )
    _set_simple_text(
        desc,
        f"{{{IMAGO_NS}}}PeopleIdentified",
        str(people_identified).lower(),
        allow_empty=True,
    )
    ET.indent(tree, space="  ")
    return tree


def write_xmp_sidecar(
    sidecar_path: str | Path,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    ocr_text: str,
    author_text: str = "",
    scene_text: str = "",
    annotation_scope: str = "",
    album_title: str = "",
    gps_latitude: str = "",
    gps_longitude: str = "",
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    source_text: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    image_width: int = 0,
    image_height: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
) -> Path:
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree: ET.ElementTree | None = None
    if path.exists():
        try:
            tree = ET.parse(path)  # type: ignore[assignment]
        except ET.ParseError:
            tree = None
    if tree is None:
        tree = build_xmp_tree(
            creator_tool=creator_tool,
            person_names=person_names,
            subjects=subjects,
            title=title,
            title_source=title_source,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            source_text=source_text,
            ocr_text=ocr_text,
            author_text=author_text,
            scene_text=scene_text,
            annotation_scope=annotation_scope,
            detections_payload=detections_payload,
            subphotos=subphotos,
            stitch_key=stitch_key,
            ocr_authority_source=ocr_authority_source,
            image_width=image_width,
            image_height=image_height,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
        )
    else:
        tree = _merge_xmp_tree(
            tree,
            creator_tool=creator_tool,
            person_names=person_names,
            subjects=subjects,
            title=title,
            title_source=title_source,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            source_text=source_text,
            ocr_text=ocr_text,
            author_text=author_text,
            scene_text=scene_text,
            annotation_scope=annotation_scope,
            detections_payload=detections_payload,
            subphotos=subphotos,
            stitch_key=stitch_key,
            ocr_authority_source=ocr_authority_source,
            image_width=image_width,
            image_height=image_height,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
