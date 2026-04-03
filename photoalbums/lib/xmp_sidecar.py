from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ._caption_text import dedupe as _dedupe
from ..naming import SCAN_NAME_RE, parse_album_filename

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
XMPMM_NS = "http://ns.adobe.com/xap/1.0/mm/"
EXIF_NS = "http://ns.adobe.com/exif/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
IMAGO_NS = "https://imago.local/ns/1.0/"
ST_EVT_NS = "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#"
PHOTOSHOP_NS = "http://ns.adobe.com/photoshop/1.0/"
XMPDM_NS = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("xmpMM", XMPMM_NS)
ET.register_namespace("exif", EXIF_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("imago", IMAGO_NS)
ET.register_namespace("stEvt", ST_EVT_NS)
ET.register_namespace("photoshop", PHOTOSHOP_NS)
ET.register_namespace("xmpDM", XMPDM_NS)


_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_SEQ = f"{{{RDF_NS}}}Seq"
_RDF_LI = f"{{{RDF_NS}}}li"
_RDF_PARSE_TYPE = f"{{{RDF_NS}}}parseType"


def _xmp_datetime_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_xmp_datetime(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for candidate in (text, text.replace("Z", "+00:00"), text.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = None
        if parsed is not None:
            normalized = parsed.replace(microsecond=0).isoformat()
            return normalized.replace("+00:00", "Z")
    for fmt in (
        "%Y:%m:%d %H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f%z",
        "%Y:%m:%d %H:%M:%S.%f",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        normalized = parsed.replace(microsecond=0).isoformat()
        return normalized.replace("+00:00", "Z")
    return text


def _normalize_xmp_text(value: str, *, multiline: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace(r"\n", "\n" if multiline else " ")


def _normalize_partial_dc_date(value: str) -> str:
    text = str(value or "").strip()
    if not text or any(ch.isalpha() for ch in text):
        return ""
    parts: list[str] = []
    current: list[str] = []
    saw_separator = False
    for ch in text:
        if ch.isdigit():
            current.append(ch)
            continue
        if ch in "-/.: ":
            if not current:
                return ""
            parts.append("".join(current))
            current = []
            saw_separator = True
            continue
        return ""
    if current:
        parts.append("".join(current))
    elif saw_separator:
        return ""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0] if len(parts[0]) == 4 and parts[0].isdigit() else ""
    if len(parts) not in {2, 3}:
        return ""
    year_text, month_text = parts[0], parts[1]
    if len(year_text) != 4 or not year_text.isdigit() or not month_text.isdigit():
        return ""
    month = int(month_text)
    if month == 0:
        if len(parts) == 2:
            return year_text
        day_text = parts[2]
        return year_text if day_text.isdigit() and int(day_text) == 0 else ""
    if month < 1 or month > 12:
        return ""
    normalized_month = f"{month:02d}"
    if len(parts) == 2:
        return f"{year_text}-{normalized_month}"
    day_text = parts[2]
    if not day_text.isdigit():
        return ""
    day = int(day_text)
    if day == 0:
        return f"{year_text}-{normalized_month}"
    try:
        datetime.strptime(f"{year_text}-{normalized_month}-{day:02d}", "%Y-%m-%d")
    except ValueError:
        return ""
    return f"{year_text}-{normalized_month}-{day:02d}"


def _normalize_dc_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_partial = _normalize_partial_dc_date(text)
    if normalized_partial:
        return normalized_partial
    normalized = _normalize_xmp_datetime(text)
    if not normalized:
        return ""
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return normalized


def _normalize_dc_dates(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(item or "") for item in value]
    else:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = _normalize_dc_date(candidate)
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


def _normalize_exif_date_time_original(value: str) -> str:
    return _normalize_xmp_datetime(str(value or "").strip())


def _resolve_date_time_original(*, dc_date: str | list[str] | tuple[str, ...], date_time_original: str = "") -> str:
    clean_dc_dates = _normalize_dc_dates(dc_date)
    clean_dc_date = clean_dc_dates[0] if clean_dc_dates else ""
    if len(clean_dc_date) == 4:
        return f"{clean_dc_date}-07-01T12:00:00"
    if len(clean_dc_date) == 7:
        return f"{clean_dc_date}-15T12:00:00"
    if len(clean_dc_date) == 10:
        return f"{clean_dc_date}T12:00:00"
    if clean_dc_date:
        return _normalize_exif_date_time_original(clean_dc_date)
    return _normalize_exif_date_time_original(date_time_original)


def _serialize_history_parameters(parameters: dict[str, object]) -> str:
    return json.dumps(parameters, ensure_ascii=False, sort_keys=True)


def _build_processing_history(
    *,
    creator_tool: str,
    history_when: str,
    stitch_key: str,
    ocr_ran: bool,
    people_detected: bool,
    people_identified: bool,
    ocr_authority_source: str,
) -> list[dict[str, object]]:
    when_text = _normalize_xmp_datetime(history_when) or _xmp_datetime_now()
    agent_text = str(creator_tool or "").strip() or "https://github.com/cove/imago"
    ocr_parameters: dict[str, object] = {
        "stage": "ocr",
        "ocr_ran": bool(ocr_ran),
    }
    clean_ocr_authority = str(ocr_authority_source or "").strip()
    if clean_ocr_authority:
        ocr_parameters["ocr_authority_source"] = clean_ocr_authority
    history: list[dict[str, object]] = [
        {
            "action": "analyzed",
            "when": when_text,
            "software_agent": agent_text,
            "parameters": {
                "stage": "people",
                "people_detected": bool(people_detected),
                "people_identified": bool(people_identified),
            },
        },
        {
            "action": "processed",
            "when": when_text,
            "software_agent": agent_text,
            "parameters": ocr_parameters,
        },
    ]
    clean_stitch_key = str(stitch_key or "").strip()
    if clean_stitch_key:
        history.append(
            {
                "action": "stitched",
                "when": when_text,
                "software_agent": agent_text,
                "parameters": {
                    "stage": "stitch",
                    "stitch_key": clean_stitch_key,
                },
            }
        )
    return history


def _add_processing_history(parent: ET.Element, history: list[dict[str, object]]) -> None:
    if not history:
        return
    field = ET.SubElement(parent, f"{{{XMPMM_NS}}}History")
    seq = ET.SubElement(field, _RDF_SEQ)
    for event in history:
        action = str(event.get("action") or "").strip()
        when_text = _normalize_xmp_datetime(str(event.get("when") or "").strip())
        software_agent = str(event.get("software_agent") or "").strip()
        parameters = event.get("parameters")
        if not action:
            continue
        li = ET.SubElement(seq, _RDF_LI)
        li.set(_RDF_PARSE_TYPE, "Resource")
        ET.SubElement(li, f"{{{ST_EVT_NS}}}action").text = action
        if software_agent:
            ET.SubElement(li, f"{{{ST_EVT_NS}}}softwareAgent").text = software_agent
        if when_text:
            ET.SubElement(li, f"{{{ST_EVT_NS}}}when").text = when_text
        if isinstance(parameters, dict) and parameters:
            ET.SubElement(li, f"{{{ST_EVT_NS}}}parameters").text = _serialize_history_parameters(parameters)


def _set_processing_history(parent: ET.Element, history: list[dict[str, object]]) -> None:
    existing = parent.find(f"{{{XMPMM_NS}}}History")
    if existing is not None:
        parent.remove(existing)
    if history:
        _add_processing_history(parent, history)


def _read_processing_history(desc: ET.Element) -> list[dict[str, object]]:
    field = desc.find(f"{{{XMPMM_NS}}}History")
    if field is None:
        return []
    seq = field.find(_RDF_SEQ)
    if seq is None:
        return []
    history: list[dict[str, object]] = []
    for item in seq.findall(_RDF_LI):
        action = str(item.findtext(f"{{{ST_EVT_NS}}}action", default="") or "").strip()
        when_text = _normalize_xmp_datetime(str(item.findtext(f"{{{ST_EVT_NS}}}when", default="") or "").strip())
        software_agent = str(item.findtext(f"{{{ST_EVT_NS}}}softwareAgent", default="") or "").strip()
        parameters_text = str(item.findtext(f"{{{ST_EVT_NS}}}parameters", default="") or "").strip()
        parameters: dict[str, object] | str = {}
        if parameters_text:
            try:
                parsed = json.loads(parameters_text)
            except json.JSONDecodeError:
                parameters = parameters_text
            else:
                parameters = parsed if isinstance(parsed, dict) else parameters_text
        history.append(
            {
                "action": action,
                "when": when_text,
                "software_agent": software_agent,
                "parameters": parameters,
            }
        )
    return history


def _derive_processing_state(history: list[dict[str, object]]) -> dict[str, object]:
    state: dict[str, object] = {}
    for event in history:
        parameters = event.get("parameters")
        if not isinstance(parameters, dict):
            continue
        stage = str(parameters.get("stage") or "").strip().lower()
        if stage == "stitch":
            stitch_key = str(parameters.get("stitch_key") or "").strip()
            if stitch_key:
                state["stitch_key"] = stitch_key
            continue
        if stage == "ocr":
            if isinstance(parameters.get("ocr_ran"), bool):
                state["ocr_ran"] = bool(parameters["ocr_ran"])
            ocr_authority_source = str(parameters.get("ocr_authority_source") or "").strip()
            if ocr_authority_source:
                state["ocr_authority_source"] = ocr_authority_source
            continue
        if stage == "people":
            if isinstance(parameters.get("people_detected"), bool):
                state["people_detected"] = bool(parameters["people_detected"])
            if isinstance(parameters.get("people_identified"), bool):
                state["people_identified"] = bool(parameters["people_identified"])
    return state


def _add_bag(parent: ET.Element, tag: str, values: list[str]) -> None:
    if not values:
        return
    field = ET.SubElement(parent, tag)
    bag = ET.SubElement(field, _RDF_BAG)
    for value in values:
        item = ET.SubElement(bag, _RDF_LI)
        item.text = value


def _add_locations_shown_bag(parent: ET.Element, locations: list[dict]) -> None:
    """Add Iptc4xmpExt:LocationShown as a bag of LocationDetails structures."""
    if not locations:
        return
    field = ET.SubElement(parent, f"{{{IPTC_EXT_NS}}}LocationShown")
    bag = ET.SubElement(field, _RDF_BAG)
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        li = ET.SubElement(bag, _RDF_LI)
        li.set(f"{{{RDF_NS}}}parseType", "Resource")
        _add_alt_text(li, f"{{{IPTC_EXT_NS}}}LocationName", str(loc.get("name") or "").strip())
        if str(loc.get("world_region") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}WorldRegion").text = _normalize_xmp_text(loc.get("world_region"))
        if str(loc.get("country_code") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}CountryCode").text = _normalize_xmp_text(loc.get("country_code"))
        if str(loc.get("country_name") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}CountryName").text = _normalize_xmp_text(loc.get("country_name"))
        if str(loc.get("province_or_state") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}ProvinceState").text = _normalize_xmp_text(
                loc.get("province_or_state")
            )
        if str(loc.get("city") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}City").text = _normalize_xmp_text(loc.get("city"))
        if str(loc.get("sublocation") or "").strip():
            ET.SubElement(li, f"{{{IPTC_EXT_NS}}}Sublocation").text = _normalize_xmp_text(loc.get("sublocation"))
        if str(loc.get("gps_latitude") or "").strip():
            ET.SubElement(li, f"{{{EXIF_NS}}}GPSLatitude").text = _format_xmp_gps_coordinate(
                str(loc.get("gps_latitude") or "").strip(),
                axis="lat",
            )
        if str(loc.get("gps_longitude") or "").strip():
            ET.SubElement(li, f"{{{EXIF_NS}}}GPSLongitude").text = _format_xmp_gps_coordinate(
                str(loc.get("gps_longitude") or "").strip(),
                axis="lon",
            )


def _set_locations_shown_bag(parent: ET.Element, locations: list[dict]) -> None:
    existing = parent.find(f"{{{IPTC_EXT_NS}}}LocationShown")
    if existing is not None:
        parent.remove(existing)
    _add_locations_shown_bag(parent, locations)


def _add_alt_text(parent: ET.Element, tag: str, value: str) -> None:
    text = _normalize_xmp_text(value, multiline=True)
    if not text:
        return
    field = ET.SubElement(parent, tag)
    alt = ET.SubElement(field, _RDF_ALT)
    item = ET.SubElement(alt, _RDF_LI)
    item.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    item.text = text


def _add_seq_text(parent: ET.Element, tag: str, value: str | list[str] | tuple[str, ...]) -> None:
    if isinstance(value, str):
        values = [_normalize_xmp_text(value)]
    else:
        values = [_normalize_xmp_text(item) for item in value]
    values = [text for text in values if text]
    if not values:
        return
    field = ET.SubElement(parent, tag)
    seq = ET.SubElement(field, _RDF_SEQ)
    for text in values:
        item = ET.SubElement(seq, _RDF_LI)
        item.text = text


def _number_lines(text: str) -> str:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return text.strip()
    return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))


def _description_alt_entries(
    *,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> list[tuple[str, str]]:
    clean_description = _normalize_xmp_text(description, multiline=True)
    clean_ocr = _normalize_xmp_text(ocr_text, multiline=True)
    clean_author = _normalize_xmp_text(author_text, multiline=True)
    clean_scene = _normalize_xmp_text(scene_text, multiline=True)
    if clean_author:
        default_text = _number_lines(clean_author)
    elif clean_ocr:
        default_text = clean_ocr
    else:
        default_text = clean_description
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lang, value in (
        ("x-default", default_text),
        ("x-caption", clean_description if clean_description != default_text else ""),
        ("x-author", clean_author if clean_author != default_text else ""),
        ("x-scene", clean_scene if clean_scene != default_text else ""),
    ):
        if not value or value in seen:
            continue
        entries.append((lang, value))
        seen.add(value)
    return entries


def _add_description_with_text_layers(
    parent: ET.Element,
    tag: str,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> None:
    """Write dc:description with full visible text in x-default and classified layers preserved."""
    entries = _description_alt_entries(
        description=description,
        ocr_text=ocr_text,
        author_text=author_text,
        scene_text=scene_text,
    )
    if not entries:
        return
    field = ET.SubElement(parent, tag)
    alt = ET.SubElement(field, _RDF_ALT)
    for lang, value in entries:
        item = ET.SubElement(alt, _RDF_LI)
        item.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
        item.text = value


def _add_simple_text(parent: ET.Element, tag: str, value: str | int | float) -> None:
    text = _normalize_xmp_text(value)
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


def _get_or_create_iptc_image_region_bag(parent: ET.Element) -> ET.Element:
    field = parent.find(f"{{{IPTC_EXT_NS}}}ImageRegion")
    if field is None:
        field = ET.SubElement(parent, f"{{{IPTC_EXT_NS}}}ImageRegion")
    bag = field.find(_RDF_BAG)
    if bag is None:
        bag = ET.SubElement(field, _RDF_BAG)
    return bag


def _add_iptc_face_regions(
    parent: ET.Element,
    people: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    if image_width <= 0 or image_height <= 0:
        return
    bag = _get_or_create_iptc_image_region_bag(parent)
    face_n = 0
    for person in people:
        name = str(person.get("name") or "").strip()
        bbox = list(person.get("bbox") or [])
        if not name or len(bbox) < 4:
            continue
        x, y, w, h = [int(v) for v in bbox[:4]]
        if w <= 0 or h <= 0:
            continue
        face_n += 1
        rx = x / image_width
        ry = y / image_height
        rw = w / image_width
        rh = h / image_height
        li = ET.SubElement(bag, _RDF_LI)
        li.set(_RDF_PARSE_TYPE, "Resource")
        boundary = ET.SubElement(li, f"{{{IPTC_EXT_NS}}}RegionBoundary")
        boundary.set(_RDF_PARSE_TYPE, "Resource")
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbShape").text = "rectangle"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbUnit").text = "relative"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbX").text = f"{rx:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbY").text = f"{ry:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbW").text = f"{rw:.6f}"
        ET.SubElement(boundary, f"{{{IPTC_EXT_NS}}}rbH").text = f"{rh:.6f}"
        ET.SubElement(li, f"{{{IPTC_EXT_NS}}}rId").text = f"face-{face_n}"
        _add_alt_text(li, f"{{{IPTC_EXT_NS}}}Name", name)


def _set_iptc_face_regions(
    parent: ET.Element,
    people: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    existing = parent.find(f"{{{IPTC_EXT_NS}}}ImageRegion")
    if existing is not None:
        parent.remove(existing)
    _add_iptc_face_regions(parent, people, image_width, image_height)


def _add_iptc_image_regions(
    parent: ET.Element,
    subphotos: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    """Write Iptc4xmpExt:ImageRegion entries for photo subregions (IPTC standard)."""
    if not subphotos or image_width <= 0 or image_height <= 0:
        return
    bag = _get_or_create_iptc_image_region_bag(parent)
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
        if author_text:
            _add_alt_text(li, f"{{{DC_NS}}}description", author_text)
        _add_simple_text(li, f"{{{IMAGO_NS}}}SceneText", scene_text)
        _add_bag(li, f"{{{IMAGO_NS}}}People", _dedupe(list(row.get("people") or [])))
        _add_bag(li, f"{{{IMAGO_NS}}}Subjects", _dedupe(list(row.get("subjects") or [])))
        detections = row.get("detections")
        if isinstance(detections, dict):
            _add_simple_text(
                li,
                f"{{{IMAGO_NS}}}Detections",
                json.dumps(detections, ensure_ascii=False, sort_keys=True),
            )


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
    location_sublocation: str = "",
    source_text: str,
    ocr_text: str,
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    create_date: str = "",
    dc_date: str | list[str] | tuple[str, ...] = "",
    date_time_original: str = "",
    history_when: str = "",
    image_width: int = 0,
    image_height: int = 0,
    page_number: int = 0,
    scan_number: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
    locations_shown: list[dict] | None = None,
) -> ET.ElementTree:
    del subphotos, image_width, image_height, scan_number
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    desc = ET.SubElement(rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")

    _add_bag(desc, f"{{{DC_NS}}}subject", _dedupe(subjects))
    _add_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", _dedupe(person_names))
    _add_alt_text(desc, f"{{{DC_NS}}}title", title)
    _add_description_with_text_layers(
        desc,
        f"{{{DC_NS}}}description",
        description,
        ocr_text,
        author_text,
        scene_text,
    )
    clean_dc_dates = _normalize_dc_dates(dc_date)
    if clean_dc_dates:
        _add_seq_text(desc, f"{{{DC_NS}}}date", clean_dc_dates)
    resolved_date_time_original = _resolve_date_time_original(
        dc_date=clean_dc_dates,
        date_time_original=date_time_original,
    )
    if resolved_date_time_original:
        _add_simple_text(desc, f"{{{EXIF_NS}}}DateTimeOriginal", resolved_date_time_original)
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
    if str(location_sublocation or "").strip():
        _add_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation).strip())
    if page_number > 0:
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}PageNumber", str(page_number))
    _add_simple_text(desc, f"{{{DC_NS}}}source", _normalize_xmp_text(source_text))

    creator = ET.SubElement(desc, f"{{{XMP_NS}}}CreatorTool")
    creator.text = _normalize_xmp_text(creator_tool) or "https://github.com/cove/imago"

    clean_ocr = _normalize_xmp_text(ocr_text, multiline=True)
    if clean_ocr:
        ocr = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRText")
        ocr.text = clean_ocr
    clean_ocr_lang = _normalize_xmp_text(ocr_lang)
    if clean_ocr_lang:
        ocr_lang_el = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRLang")
        ocr_lang_el.text = clean_ocr_lang
    clean_author_text = _normalize_xmp_text(author_text, multiline=True)
    if clean_author_text:
        author = ET.SubElement(desc, f"{{{IMAGO_NS}}}AuthorText")
        author.text = clean_author_text
    clean_scene_text = _normalize_xmp_text(scene_text, multiline=True)
    if clean_scene_text:
        scene = ET.SubElement(desc, f"{{{IMAGO_NS}}}SceneText")
        scene.text = clean_scene_text
    clean_title_source = _normalize_xmp_text(title_source)
    if clean_title_source:
        title_src = ET.SubElement(desc, f"{{{IMAGO_NS}}}TitleSource")
        title_src.text = clean_title_source
    clean_ocr_authority_source = _normalize_xmp_text(ocr_authority_source)
    if clean_ocr_authority_source:
        ocr_source = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRAuthoritySource")
        ocr_source.text = clean_ocr_authority_source

    if detections_payload:
        payload = ET.SubElement(desc, f"{{{IMAGO_NS}}}Detections")
        payload.text = json.dumps(detections_payload, ensure_ascii=False, sort_keys=True)
    _set_locations_shown_bag(desc, list(locations_shown) if locations_shown else [])
    _add_processing_history(
        desc,
        _build_processing_history(
            creator_tool=creator_tool,
            history_when=history_when,
            stitch_key=stitch_key,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
            ocr_authority_source=ocr_authority_source,
        ),
    )

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


def _remove_field(parent: ET.Element, tag: str) -> None:
    existing = parent.find(tag)
    if existing is not None:
        parent.remove(existing)


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


def _set_seq_text(parent: ET.Element, tag: str, value: str | list[str] | tuple[str, ...]) -> None:
    if isinstance(value, str):
        values = [str(value or "").strip()]
    else:
        values = [str(item or "").strip() for item in value]
    values = [text for text in values if text]
    existing = parent.find(tag)
    if not values:
        if existing is not None:
            parent.remove(existing)
        return

    def _builder(field: ET.Element) -> None:
        seq = ET.SubElement(field, _RDF_SEQ)
        for text in values:
            item = ET.SubElement(seq, _RDF_LI)
            item.text = text

    _replace_field(parent, tag, _builder)


def _set_description_with_text_layers(
    parent: ET.Element,
    tag: str,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> None:
    """Set dc:description with full visible text in x-default and classified layers preserved."""
    entries = _description_alt_entries(
        description=description,
        ocr_text=ocr_text,
        author_text=author_text,
        scene_text=scene_text,
    )
    existing = parent.find(tag)
    if not entries:
        if existing is not None:
            parent.remove(existing)
        return

    def _builder(field: ET.Element) -> None:
        alt = ET.SubElement(field, _RDF_ALT)
        for lang, value in entries:
            item = ET.SubElement(alt, _RDF_LI)
            item.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
            item.text = value

    _replace_field(parent, tag, _builder)


def _set_simple_text(parent: ET.Element, tag: str, value: str | int | float, *, allow_empty: bool = False) -> None:
    text = _normalize_xmp_text(value) if isinstance(value, str) else str(value)
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


def _get_alt_text(parent: ET.Element, tag: str, *, prefer_lang: str = "", fallback_to_any: bool = True) -> str:
    field = parent.find(tag)
    if field is None:
        return ""
    alt = field.find(_RDF_ALT)
    if alt is None:
        return ""
    lang_attr = "{http://www.w3.org/XML/1998/namespace}lang"
    preferred = str(prefer_lang or "").strip()
    if preferred:
        for item in alt.findall(_RDF_LI):
            text = str(item.text or "").strip()
            if item.get(lang_attr) == preferred and text:
                return text
        if not fallback_to_any:
            return ""
    for item in alt.findall(_RDF_LI):
        text = str(item.text or "").strip()
        if text:
            return text
    return ""


def _get_seq_values(parent: ET.Element, tag: str) -> list[str]:
    field = parent.find(tag)
    if field is None:
        return []
    seq = field.find(_RDF_SEQ)
    if seq is not None:
        values: list[str] = []
        for item in seq.findall(_RDF_LI):
            text = str(item.text or "").strip()
            if text:
                values.append(text)
        return values
    text = str(field.text or "").strip()
    return [text] if text else []


def _get_seq_text(parent: ET.Element, tag: str) -> str:
    values = _get_seq_values(parent, tag)
    return values[0] if values else ""


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


def read_locations_shown(sidecar_path: str | Path) -> list[dict[str, str]]:
    """Return Iptc4xmpExt:LocationShown rows from an XMP sidecar. Returns [] on any error."""
    _LOCATION_TAG = f"{{{IPTC_EXT_NS}}}LocationShown"
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel

        path = Path(sidecar_path)
        if not path.is_file():
            return []
        tree = ET.parse(path)
        desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
        if desc is None:
            return []
        field = desc.find(_LOCATION_TAG)
        if field is None:
            return []
        bag = field.find(_RDF_BAG)
        if bag is None:
            return []
        rows: list[dict[str, str]] = []
        for li in bag.findall(_RDF_LI):
            row = {
                "name": _get_alt_text(li, f"{{{IPTC_EXT_NS}}}LocationName", prefer_lang="x-default"),
                "world_region": str(li.findtext(f"{{{IPTC_EXT_NS}}}WorldRegion", default="") or "").strip(),
                "country_code": str(li.findtext(f"{{{IPTC_EXT_NS}}}CountryCode", default="") or "").strip(),
                "country_name": str(li.findtext(f"{{{IPTC_EXT_NS}}}CountryName", default="") or "").strip(),
                "province_or_state": str(li.findtext(f"{{{IPTC_EXT_NS}}}ProvinceState", default="") or "").strip(),
                "city": str(li.findtext(f"{{{IPTC_EXT_NS}}}City", default="") or "").strip(),
                "sublocation": str(li.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or "").strip(),
                "gps_latitude": _xmp_gps_to_decimal(
                    li.findtext(f"{{{EXIF_NS}}}GPSLatitude", default=""),
                    axis="lat",
                ),
                "gps_longitude": _xmp_gps_to_decimal(
                    li.findtext(f"{{{EXIF_NS}}}GPSLongitude", default=""),
                    axis="lon",
                ),
            }
            if any(row.values()):
                rows.append(row)
        return rows
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
    processing_meta: dict[str, object] = dict((detections_payload or {}).get("processing") or {})
    processing_history = _read_processing_history(desc)
    processing_state = _derive_processing_state(processing_history)
    dc_date_values = _normalize_dc_dates(_get_seq_values(desc, f"{{{DC_NS}}}date"))
    return {
        "creator_tool": str(desc.findtext(f"{{{XMP_NS}}}CreatorTool", default="") or "").strip(),
        "create_date": _normalize_xmp_datetime(str(desc.findtext(f"{{{XMP_NS}}}CreateDate", default="") or "").strip()),
        "dc_date": dc_date_values[0] if dc_date_values else "",
        "dc_date_values": dc_date_values,
        "date_time_original": _normalize_exif_date_time_original(
            str(desc.findtext(f"{{{EXIF_NS}}}DateTimeOriginal", default="") or "").strip()
        ),
        "title": _get_alt_text(desc, f"{{{DC_NS}}}title", prefer_lang="x-default"),
        "description": _get_alt_text(
            desc,
            f"{{{DC_NS}}}description",
            prefer_lang="x-default",
            fallback_to_any=False,
        ),
        "album_title": str(
            desc.findtext(f"{{{XMPDM_NS}}}album", default="")
            or desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="")
            or ""
        ).strip(),
        "source_text": str(desc.findtext(f"{{{DC_NS}}}source", default="") or "").strip(),
        "gps_latitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or "").strip(),
        "gps_longitude": str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or "").strip(),
        "location_city": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or "").strip(),
        "location_state": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or "").strip(),
        "location_country": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or "").strip(),
        "location_sublocation": str(desc.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or "").strip(),
        "ocr_text": str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "").strip(),
        "ocr_lang": str(desc.findtext(f"{{{IMAGO_NS}}}OCRLang", default="") or "").strip(),
        "author_text": str(desc.findtext(f"{{{IMAGO_NS}}}AuthorText", default="") or "").strip(),
        "scene_text": str(desc.findtext(f"{{{IMAGO_NS}}}SceneText", default="") or "").strip(),
        "title_source": str(desc.findtext(f"{{{IMAGO_NS}}}TitleSource", default="") or "").strip(),
        "ocr_authority_source": str(
            processing_state.get("ocr_authority_source")
            or desc.findtext(f"{{{IMAGO_NS}}}OCRAuthoritySource", default="")
            or ""
        ).strip(),
        "stitch_key": str(
            processing_state.get("stitch_key") or desc.findtext(f"{{{IMAGO_NS}}}StitchKey", default="") or ""
        ).strip(),
        "processing_history": processing_history,
        "detections": detections_payload,
        "ocr_ran": (
            processing_state["ocr_ran"]
            if "ocr_ran" in processing_state
            else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}OcrRan")
        ),
        "people_detected": (
            processing_state["people_detected"]
            if "people_detected" in processing_state
            else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleDetected")
        ),
        "people_identified": (
            processing_state["people_identified"]
            if "people_identified" in processing_state
            else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleIdentified")
        ),
        "processor_signature": str(processing_meta.get("processor_signature") or "").strip(),
        "settings_signature": str(processing_meta.get("settings_signature") or "").strip(),
        "cast_store_signature": str(processing_meta.get("cast_store_signature") or "").strip(),
        "size": int(processing_meta.get("size") or -1),
        "mtime_ns": int(processing_meta.get("mtime_ns") or -1),
        "date_estimate_input_hash": str(processing_meta.get("date_estimate_input_hash") or "").strip(),
        "ocr_authority_signature": str(processing_meta.get("ocr_authority_signature") or "").strip(),
        "ocr_authority_hash": str(processing_meta.get("ocr_authority_hash") or "").strip(),
        "analysis_mode": str(processing_meta.get("analysis_mode") or "").strip(),
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
    location_sublocation: str = "",
    source_text: str,
    ocr_text: str,
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    create_date: str = "",
    dc_date: str | list[str] | tuple[str, ...] = "",
    date_time_original: str = "",
    history_when: str = "",
    image_width: int = 0,
    image_height: int = 0,
    page_number: int = 0,
    scan_number: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
    locations_shown: list[dict] | None = None,
) -> ET.ElementTree:
    del subphotos, image_width, image_height, scan_number
    desc = _get_or_create_rdf_desc(tree)
    _set_bag(desc, f"{{{DC_NS}}}subject", subjects)
    _set_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", person_names)
    _set_alt_text(desc, f"{{{DC_NS}}}title", title)
    _set_description_with_text_layers(
        desc,
        f"{{{DC_NS}}}description",
        description,
        ocr_text,
        author_text,
        scene_text,
    )
    normalized_dc_dates = _normalize_dc_dates(dc_date)
    _set_seq_text(desc, f"{{{DC_NS}}}date", normalized_dc_dates)
    _set_simple_text(
        desc,
        f"{{{EXIF_NS}}}DateTimeOriginal",
        _resolve_date_time_original(dc_date=normalized_dc_dates, date_time_original=date_time_original),
    )
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    _set_gps_fields(desc, gps_latitude, gps_longitude)
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())
    _set_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}PageNumber", str(page_number) if page_number > 0 else "")
    _remove_field(desc, f"{{{IMAGO_NS}}}ScanNumber")
    _set_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())
    _set_simple_text(
        desc,
        f"{{{XMP_NS}}}CreatorTool",
        str(creator_tool or "").strip() or "https://github.com/cove/imago",
    )
    clean_ocr = str(ocr_text or "").strip()
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRText", clean_ocr)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRLang", str(ocr_lang or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AuthorText", str(author_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}SceneText", str(scene_text or "").strip())
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
    _remove_field(desc, f"{{{IPTC_EXT_NS}}}ImageRegion")
    _set_locations_shown_bag(desc, list(locations_shown) if locations_shown else [])
    _set_processing_history(
        desc,
        _build_processing_history(
            creator_tool=creator_tool,
            history_when=history_when,
            stitch_key=stitch_key,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
            ocr_authority_source=ocr_authority_source,
        ),
    )
    for legacy_tag in (
        f"{{{IMAGO_NS}}}StitchKey",
        f"{{{IMAGO_NS}}}OcrRan",
        f"{{{IMAGO_NS}}}PeopleDetected",
        f"{{{IMAGO_NS}}}PeopleIdentified",
        f"{{{IMAGO_NS}}}SubPhotos",
        f"{{{XMP_NS}}}CreateDate",
        f"{{{XMPDM_NS}}}album",
    ):
        _remove_field(desc, legacy_tag)
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
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    album_title: str = "",
    gps_latitude: str = "",
    gps_longitude: str = "",
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    location_sublocation: str = "",
    source_text: str = "",
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
    stitch_key: str = "",
    ocr_authority_source: str = "",
    create_date: str = "",
    dc_date: str | list[str] | tuple[str, ...] = "",
    date_time_original: str = "",
    history_when: str = "",
    image_width: int = 0,
    image_height: int = 0,
    ocr_ran: bool = False,
    people_detected: bool = False,
    people_identified: bool = False,
    locations_shown: list[dict] | None = None,
) -> Path:
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, _page_str = parse_album_filename(path.stem)
    page_number = int(_page_str) if _page_str.isdigit() else 0
    _scan_m = SCAN_NAME_RE.search(path.name)
    scan_number = int(_scan_m.group("scan")) if _scan_m else 0
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
            location_sublocation=location_sublocation,
            source_text=source_text,
            ocr_text=ocr_text,
            ocr_lang=ocr_lang,
            author_text=author_text,
            scene_text=scene_text,
            detections_payload=detections_payload,
            subphotos=subphotos,
            stitch_key=stitch_key,
            ocr_authority_source=ocr_authority_source,
            create_date=create_date,
            dc_date=dc_date,
            date_time_original=date_time_original,
            history_when=history_when,
            image_width=image_width,
            image_height=image_height,
            page_number=page_number,
            scan_number=scan_number,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
            locations_shown=locations_shown,
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
            location_sublocation=location_sublocation,
            source_text=source_text,
            ocr_text=ocr_text,
            ocr_lang=ocr_lang,
            author_text=author_text,
            scene_text=scene_text,
            detections_payload=detections_payload,
            subphotos=subphotos,
            stitch_key=stitch_key,
            ocr_authority_source=ocr_authority_source,
            create_date=create_date,
            dc_date=dc_date,
            date_time_original=date_time_original,
            history_when=history_when,
            image_width=image_width,
            image_height=image_height,
            page_number=page_number,
            scan_number=scan_number,
            ocr_ran=ocr_ran,
            people_detected=people_detected,
            people_identified=people_identified,
            locations_shown=locations_shown,
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
