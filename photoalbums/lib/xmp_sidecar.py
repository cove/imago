from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
EXIF_NS = "http://ns.adobe.com/exif/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
IMAGO_NS = "https://imago.local/ns/1.0/"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("exif", EXIF_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("imago", IMAGO_NS)

_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_SEQ = f"{{{RDF_NS}}}Seq"
_RDF_LI = f"{{{RDF_NS}}}li"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
        _set_simple_text(parent, f"{{{EXIF_NS}}}GPSLatitude", _format_xmp_gps_coordinate(lat_text, axis="lat"))
        _set_simple_text(parent, f"{{{EXIF_NS}}}GPSLongitude", _format_xmp_gps_coordinate(lon_text, axis="lon"))
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


def _add_subphotos(parent: ET.Element, subphotos: list[dict]) -> None:
    if not subphotos:
        return
    field = ET.SubElement(parent, f"{{{IMAGO_NS}}}SubPhotos")
    seq = ET.SubElement(field, _RDF_SEQ)
    for row in subphotos:
        item = ET.SubElement(seq, _RDF_LI)
        item.set(f"{{{RDF_NS}}}parseType", "Resource")
        _add_simple_text(item, f"{{{IMAGO_NS}}}Index", int(row.get("index", 0)))
        bounds = dict(row.get("bounds") or {})
        _add_simple_text(item, f"{{{IMAGO_NS}}}X", int(bounds.get("x", 0)))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Y", int(bounds.get("y", 0)))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Width", int(bounds.get("width", 0)))
        _add_simple_text(item, f"{{{IMAGO_NS}}}Height", int(bounds.get("height", 0)))
        _add_alt_text(item, f"{{{IMAGO_NS}}}Description", str(row.get("description") or ""))
        ocr_text = str(row.get("ocr_text") or "").strip()
        if ocr_text:
            _add_simple_text(item, f"{{{IMAGO_NS}}}OCRText", ocr_text)
        _add_bag(item, f"{{{IMAGO_NS}}}People", _dedupe(list(row.get("people") or [])))
        _add_bag(item, f"{{{IMAGO_NS}}}Subjects", _dedupe(list(row.get("subjects") or [])))
        detections = row.get("detections")
        if detections:
            _add_simple_text(
                item,
                f"{{{IMAGO_NS}}}Detections",
                json.dumps(detections, ensure_ascii=False, sort_keys=True),
            )


def build_xmp_tree(
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    source_text: str,
    ocr_text: str,
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
) -> ET.ElementTree:
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    desc = ET.SubElement(rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")

    _add_bag(desc, f"{{{DC_NS}}}subject", _dedupe(subjects))
    _add_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", _dedupe(person_names))
    _add_alt_text(desc, f"{{{DC_NS}}}description", description)
    if str(album_title or "").strip():
        _add_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    if str(gps_latitude or "").strip() and str(gps_longitude or "").strip():
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSLatitude", _format_xmp_gps_coordinate(gps_latitude, axis="lat"))
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSLongitude", _format_xmp_gps_coordinate(gps_longitude, axis="lon"))
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSMapDatum", "WGS-84")
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSVersionID", "2.3.0.0")
    _add_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())

    creator = ET.SubElement(desc, f"{{{XMP_NS}}}CreatorTool")
    creator.text = str(creator_tool or "").strip() or "imago-photoalbums-ai-index"

    clean_ocr = str(ocr_text or "").strip()
    if clean_ocr:
        ocr = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRText")
        ocr.text = clean_ocr

    if detections_payload:
        payload = ET.SubElement(desc, f"{{{IMAGO_NS}}}Detections")
        payload.text = json.dumps(detections_payload, ensure_ascii=False, sort_keys=True)
    if subphotos:
        _add_subphotos(desc, list(subphotos))

    tree = ET.ElementTree(xmpmeta)
    ET.indent(tree, space="  ")
    return tree


def _get_or_create_rdf_desc(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        rdf = ET.SubElement(root, _RDF_ROOT)
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


def _set_subphotos(parent: ET.Element, subphotos: list[dict] | None) -> None:
    existing = parent.find(f"{{{IMAGO_NS}}}SubPhotos")
    if existing is not None:
        parent.remove(existing)
    if not subphotos:
        return
    _add_subphotos(parent, list(subphotos))


def _get_rdf_desc(tree: ET.ElementTree) -> ET.Element | None:
    root = tree.getroot()
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


def read_ai_sidecar_state(sidecar_path: str | Path) -> dict[str, object] | None:
    path = Path(sidecar_path)
    if not path.is_file():
        return None
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    desc = _get_rdf_desc(tree)
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
        "description": _get_alt_text(desc, f"{{{DC_NS}}}description"),
        "album_title": str(desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="") or "").strip(),
        "gps_latitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or "").strip(),
        "gps_longitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or "").strip(),
        "ocr_text": str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "").strip(),
        "detections": detections_payload,
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
    expected_creator = str(creator_tool or "").strip()
    if expected_creator and str(state.get("creator_tool") or "").strip() != expected_creator:
        return False
    detections = state.get("detections")
    if not isinstance(detections, dict):
        return False
    if bool(enable_people) and not isinstance(detections.get("people"), list):
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
            from .ai_caption import _looks_like_reasoning_or_prompt_echo  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - defensive import fallback
            _looks_like_reasoning_or_prompt_echo = None
        if _looks_like_reasoning_or_prompt_echo is not None and _looks_like_reasoning_or_prompt_echo(description):
            return False
    ocr_text = str(state.get("ocr_text") or "").strip()
    if ocr_text:
        try:
            from .ai_ocr import _looks_like_ocr_reasoning  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - defensive import fallback
            _looks_like_ocr_reasoning = None
        if _looks_like_ocr_reasoning is not None and _looks_like_ocr_reasoning(ocr_text):
            return False
    if caption_name != "none":
        caption = detections.get("caption")
        ocr = detections.get("ocr")
        has_signal = False
        if isinstance(detections.get("people"), list) and detections.get("people"):
            has_signal = True
        elif isinstance(detections.get("objects"), list) and detections.get("objects"):
            has_signal = True
        elif isinstance(ocr, dict) and int(ocr.get("chars") or 0) > 0:
            has_signal = True
        elif isinstance(caption, dict) and str(caption.get("effective_engine") or "").strip() == "page-summary":
            has_signal = True
        if has_signal and not description:
            return False
    return True


def _merge_xmp_tree(
    tree: ET.ElementTree,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    source_text: str,
    ocr_text: str,
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
) -> ET.ElementTree:
    desc = _get_or_create_rdf_desc(tree)
    _set_bag(desc, f"{{{DC_NS}}}subject", subjects)
    _set_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", person_names)
    _set_alt_text(desc, f"{{{DC_NS}}}description", description)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    _set_gps_fields(desc, gps_latitude, gps_longitude)
    _set_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())
    _set_simple_text(desc, f"{{{XMP_NS}}}CreatorTool", str(creator_tool or "").strip() or "imago-photoalbums-ai-index")
    clean_ocr = str(ocr_text or "").strip()
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRText", clean_ocr)
    if detections_payload:
        _set_simple_text(
            desc,
            f"{{{IMAGO_NS}}}Detections",
            json.dumps(detections_payload, ensure_ascii=False, sort_keys=True),
        )
    else:
        _set_simple_text(desc, f"{{{IMAGO_NS}}}Detections", "")
    _set_subphotos(desc, subphotos)
    ET.indent(tree, space="  ")
    return tree


def write_xmp_sidecar(
    sidecar_path: str | Path,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    description: str,
    ocr_text: str,
    album_title: str = "",
    gps_latitude: str = "",
    gps_longitude: str = "",
    source_text: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
) -> Path:
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree: ET.ElementTree | None = None
    if path.exists():
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            tree = None
    if tree is None:
        tree = build_xmp_tree(
            creator_tool=creator_tool,
            person_names=person_names,
            subjects=subjects,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            source_text=source_text,
            ocr_text=ocr_text,
            detections_payload=detections_payload,
            subphotos=subphotos,
        )
    else:
        tree = _merge_xmp_tree(
            tree,
            creator_tool=creator_tool,
            person_names=person_names,
            subjects=subjects,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            source_text=source_text,
            ocr_text=ocr_text,
            detections_payload=detections_payload,
            subphotos=subphotos,
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
