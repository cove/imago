from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ._caption_text import dedupe as _dedupe
from ..naming import SCAN_NAME_RE, VIEW_PAGE_RE, is_pages_dir, is_photos_dir, parse_album_filename

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
CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"

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
ET.register_namespace("crs", CRS_NS)
ET.register_namespace("mwg-rs", MWGRS_NS)
ET.register_namespace("stArea", STAREA_NS)


_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_ALT = f"{{{RDF_NS}}}Alt"
_RDF_SEQ = f"{{{RDF_NS}}}Seq"
_RDF_LI = f"{{{RDF_NS}}}li"
_RDF_PARSE_TYPE = f"{{{RDF_NS}}}parseType"

DESCRIPTION_ROLE_PLAIN = "plain"
DESCRIPTION_ROLE_PAGE = "page"
DESCRIPTION_ROLE_CROP = "crop"


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


def _format_location_created(
    *,
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    location_sublocation: str = "",
) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in (location_sublocation, location_city, location_state, location_country):
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        parts.append(clean)
        seen.add(clean)
    return ", ".join(parts)


def _with_location_detections(
    detections_payload: dict | None,
    *,
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    location_sublocation: str = "",
) -> dict | None:
    payload = dict(detections_payload or {})
    location = dict(payload.get("location") or {}) if isinstance(payload.get("location"), dict) else {}
    if str(location_city or "").strip() and not str(location.get("city") or "").strip():
        location["city"] = str(location_city).strip()
    if str(location_state or "").strip() and not str(location.get("state") or "").strip():
        location["state"] = str(location_state).strip()
    if str(location_country or "").strip() and not str(location.get("country") or "").strip():
        location["country"] = str(location_country).strip()
    if str(location_sublocation or "").strip() and not str(location.get("sublocation") or "").strip():
        location["sublocation"] = str(location_sublocation).strip()
    if location:
        payload["location"] = location
    return payload or None


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


def _add_region_location_struct(parent: ET.Element, tag: str, payload: dict[str, object]) -> None:
    if not isinstance(payload, dict):
        return
    clean_payload = {
        key: str(payload.get(key) or "").strip()
        for key in ("address", "city", "state", "country", "sublocation", "gps_latitude", "gps_longitude")
    }
    if not any(clean_payload.values()):
        return
    field = ET.SubElement(parent, tag)
    field.set(_RDF_PARSE_TYPE, "Resource")
    if clean_payload["address"]:
        ET.SubElement(field, f"{{{IMAGO_NS}}}Address").text = _normalize_xmp_text(clean_payload["address"])
    if clean_payload["city"]:
        ET.SubElement(field, f"{{{PHOTOSHOP_NS}}}City").text = _normalize_xmp_text(clean_payload["city"])
    if clean_payload["state"]:
        ET.SubElement(field, f"{{{PHOTOSHOP_NS}}}State").text = _normalize_xmp_text(clean_payload["state"])
    if clean_payload["country"]:
        ET.SubElement(field, f"{{{PHOTOSHOP_NS}}}Country").text = _normalize_xmp_text(clean_payload["country"])
    if clean_payload["sublocation"]:
        ET.SubElement(field, f"{{{IPTC_EXT_NS}}}Sublocation").text = _normalize_xmp_text(clean_payload["sublocation"])
    if clean_payload["gps_latitude"]:
        ET.SubElement(field, f"{{{EXIF_NS}}}GPSLatitude").text = _format_xmp_gps_coordinate(
            clean_payload["gps_latitude"],
            axis="lat",
        )
    if clean_payload["gps_longitude"]:
        ET.SubElement(field, f"{{{EXIF_NS}}}GPSLongitude").text = _format_xmp_gps_coordinate(
            clean_payload["gps_longitude"],
            axis="lon",
        )


def _read_region_location_struct(parent: ET.Element, tag: str) -> dict[str, str]:
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel

        field = parent.find(tag)
        if field is None:
            return {}
        payload = {
            "address": str(field.findtext(f"{{{IMAGO_NS}}}Address", default="") or "").strip(),
            "city": str(field.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or "").strip(),
            "state": str(field.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or "").strip(),
            "country": str(field.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or "").strip(),
            "sublocation": str(field.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or "").strip(),
            "gps_latitude": _xmp_gps_to_decimal(field.findtext(f"{{{EXIF_NS}}}GPSLatitude", default=""), axis="lat"),
            "gps_longitude": _xmp_gps_to_decimal(field.findtext(f"{{{EXIF_NS}}}GPSLongitude", default=""), axis="lon"),
        }
        return payload if any(payload.values()) else {}
    except Exception:
        return {}


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


def _page_description_summary(ocr_text: str, scene_text: str) -> str:
    clean_ocr = ocr_text.strip()
    clean_scene = scene_text.strip()
    if clean_ocr and clean_scene:
        return f"Caption:\n{clean_ocr}\n\nScene Text:\n{clean_scene}"
    return clean_ocr or clean_scene


def _description_role_for_sidecar_path(sidecar_path: str | Path) -> str:
    path = Path(sidecar_path)
    if is_photos_dir(path.parent):
        return DESCRIPTION_ROLE_CROP
    if is_pages_dir(path.parent) and VIEW_PAGE_RE.search(path.stem):
        return DESCRIPTION_ROLE_PAGE
    return DESCRIPTION_ROLE_PLAIN


def _description_default_text(
    *,
    description_role: str,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> str:
    clean_description = _normalize_xmp_text(description, multiline=True)
    clean_ocr = _normalize_xmp_text(ocr_text, multiline=True)
    clean_author = _normalize_xmp_text(author_text, multiline=True)
    clean_scene = _normalize_xmp_text(scene_text, multiline=True)
    if description_role == DESCRIPTION_ROLE_PAGE:
        return _page_description_summary(clean_ocr, clean_scene) or clean_description
    if description_role == DESCRIPTION_ROLE_CROP:
        return clean_description
    return clean_description or clean_ocr or clean_author or clean_scene


def _add_description_with_text_layers(
    parent: ET.Element,
    tag: str,
    description_role: str,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> None:
    text = _description_default_text(
        description_role=description_role,
        description=description,
        ocr_text=ocr_text,
        author_text=author_text,
        scene_text=scene_text,
    )
    if not text:
        return
    _add_alt_text(parent, tag, text)


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
        ET.SubElement(li, f"{{{IPTC_EXT_NS}}}RCtype").text = "face-identified"
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


def _image_region_is_face(item: ET.Element) -> bool:
    region_type = str(item.findtext(f"{{{IPTC_EXT_NS}}}RCtype", default="") or "").strip().lower()
    if region_type.startswith("face-"):
        return True
    region_id = str(item.findtext(f"{{{IPTC_EXT_NS}}}rId", default="") or "").strip()
    return region_id.startswith("face-")


def _replace_iptc_face_regions(
    parent: ET.Element,
    people: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    field = parent.find(f"{{{IPTC_EXT_NS}}}ImageRegion")
    if field is not None:
        bag = field.find(_RDF_BAG)
        if bag is not None:
            for item in list(bag.findall(_RDF_LI)):
                if _image_region_is_face(item):
                    bag.remove(item)
            if not list(bag):
                field.remove(bag)
        if not list(field):
            parent.remove(field)
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
    parent_ocr_text: str = "",
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    description_role: str = DESCRIPTION_ROLE_PLAIN,
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
    del subphotos, scan_number
    detections_payload = _with_location_detections(
        detections_payload,
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
        location_sublocation=location_sublocation,
    )
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
        description_role,
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
    clean_create_date = _normalize_xmp_datetime(create_date)
    if clean_create_date:
        _add_simple_text(desc, f"{{{XMP_NS}}}CreateDate", clean_create_date)
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
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    if str(location_state or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    if str(location_country or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())
    if str(location_sublocation or "").strip():
        _add_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation or "").strip())
    _add_simple_text(
        desc,
        f"{{{IPTC_EXT_NS}}}LocationCreated",
        _format_location_created(
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            location_sublocation=location_sublocation,
        ),
    )
    if page_number > 0:
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}PageNumber", str(page_number))
    _add_simple_text(desc, f"{{{DC_NS}}}source", _normalize_xmp_text(source_text))

    creator = ET.SubElement(desc, f"{{{XMP_NS}}}CreatorTool")
    creator.text = _normalize_xmp_text(creator_tool) or "https://github.com/cove/imago"

    clean_ocr = _normalize_xmp_text(ocr_text, multiline=True)
    if clean_ocr:
        ocr = ET.SubElement(desc, f"{{{IMAGO_NS}}}OCRText")
        ocr.text = clean_ocr
    clean_parent_ocr = _normalize_xmp_text(parent_ocr_text, multiline=True)
    if clean_parent_ocr:
        parent_ocr = ET.SubElement(desc, f"{{{IMAGO_NS}}}ParentOCRText")
        parent_ocr.text = clean_parent_ocr
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

    detections_payload = _with_processing_state(
        detections_payload,
        people_detected=people_detected,
        people_identified=people_identified,
    )
    if detections_payload:
        payload = ET.SubElement(desc, f"{{{IMAGO_NS}}}Detections")
        payload.text = json.dumps(detections_payload, ensure_ascii=False, sort_keys=True)
    if isinstance(detections_payload, dict) and "people" in detections_payload:
        _replace_iptc_face_regions(
            desc,
            [row for row in list(detections_payload.get("people") or []) if isinstance(row, dict)],
            image_width,
            image_height,
        )
    _set_locations_shown_bag(desc, list(locations_shown) if locations_shown else [])

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
    description_role: str,
    description: str,
    ocr_text: str,
    author_text: str,
    scene_text: str,
) -> None:
    text = _description_default_text(
        description_role=description_role,
        description=description,
        ocr_text=ocr_text,
        author_text=author_text,
        scene_text=scene_text,
    )
    existing = parent.find(tag)
    if not text:
        if existing is not None:
            parent.remove(existing)
        return
    _set_alt_text(parent, tag, text)


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


def _load_or_create_xmp_tree(path: str | Path) -> ET.ElementTree:
    path = Path(path)
    if path.is_file():
        try:
            return ET.parse(path)  # type: ignore[return-value]
        except ET.ParseError:
            pass
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    xmpmeta.set(f"{{{X_NS}}}xmptk", "imago")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    ET.SubElement(rdf, _RDF_DESC).set(f"{{{RDF_NS}}}about", "")
    return ET.ElementTree(xmpmeta)


def _save_xmp_tree(tree: ET.ElementTree, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _read_detections_payload(desc: ET.Element) -> dict[str, object]:
    detections_text = str(desc.findtext(f"{{{IMAGO_NS}}}Detections", default="") or "").strip()
    if not detections_text:
        return {}
    try:
        parsed = json.loads(detections_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_detections_payload(tree: ET.ElementTree, path: str | Path, detections: dict[str, object]) -> None:
    desc = _get_or_create_rdf_desc(tree)
    if detections:
        _set_simple_text(
            desc,
            f"{{{IMAGO_NS}}}Detections",
            json.dumps(detections, ensure_ascii=False, sort_keys=True),
        )
    else:
        _set_simple_text(desc, f"{{{IMAGO_NS}}}Detections", "")
    _save_xmp_tree(tree, path)


def _with_processing_state(
    detections_payload: dict[str, object] | None,
    *,
    people_detected: bool,
    people_identified: bool,
) -> dict[str, object] | None:
    if detections_payload is None:
        return None
    if not isinstance(detections_payload.get("processing"), dict):
        return detections_payload
    merged = dict(detections_payload)
    processing = dict(merged.get("processing") or {})
    processing["people_detected"] = bool(people_detected)
    processing["people_identified"] = bool(people_identified)
    merged["processing"] = processing
    return merged


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


def _get_bag_values(parent: ET.Element, tag: str) -> list[str]:
    field = parent.find(tag)
    if field is None:
        return []
    bag = field.find(_RDF_BAG)
    if bag is None:
        return []
    values: list[str] = []
    for item in bag.findall(_RDF_LI):
        text = str(item.text or "").strip()
        if text:
            values.append(text)
    return values


def _get_description_value(parent: ET.Element, *, legacy_caption_first: bool = False) -> str:
    if legacy_caption_first:
        legacy_caption = _get_alt_text(parent, f"{{{DC_NS}}}description", prefer_lang="x-caption", fallback_to_any=False)
        if legacy_caption:
            return legacy_caption
    default_text = _get_alt_text(parent, f"{{{DC_NS}}}description", prefer_lang="x-default", fallback_to_any=False)
    if default_text:
        return default_text
    return _get_alt_text(parent, f"{{{DC_NS}}}description", prefer_lang="x-caption", fallback_to_any=False)


def _coalesce_text(value: str, existing: str) -> str:
    clean_value = str(value or "").strip()
    return clean_value or str(existing or "").strip()


def _coalesce_gps(value: str, existing: str, *, axis: str) -> str:
    clean_value = str(value or "").strip()
    if clean_value:
        return clean_value
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel

        return _xmp_gps_to_decimal(existing, axis=axis)
    except Exception:
        return ""


def _read_locations_shown_from_desc(desc: ET.Element) -> list[dict[str, str]]:
    _LOCATION_TAG = f"{{{IPTC_EXT_NS}}}LocationShown"
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel

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


def _read_xmp_bool(desc: ET.Element, tag: str) -> bool | None:
    """Return True/False if the tag is present with a boolean value, else None if absent."""
    raw = desc.findtext(tag)
    if raw is None:
        return None
    return str(raw or "").strip().lower() == "true"


def _normalize_pipeline_entry(entry: object) -> dict[str, object]:
    """Normalise a legacy {"completed": ts} entry to the new schema on read."""
    if not isinstance(entry, dict):
        return {}
    if "completed" in entry and "timestamp" not in entry:
        normalized: dict[str, object] = dict(entry)
        normalized["timestamp"] = normalized["completed"]
        normalized.setdefault("result", "ok")
        normalized.setdefault("input_hash", "")
        return normalized
    return dict(entry)


def xmp_datetime_now() -> str:
    """Return current UTC time formatted as XMP ISO-8601 string."""
    return _xmp_datetime_now()


def _has_legacy_pipeline_entries(xmp_path: Path) -> bool:
    try:
        tree = ET.parse(xmp_path)
    except ET.ParseError:
        return False
    desc = _get_rdf_desc(tree)
    if desc is None:
        return False
    pipeline = _read_detections_payload(desc).get("pipeline")
    if not isinstance(pipeline, dict):
        return False
    return any(
        isinstance(v, dict) and "completed" in v and "timestamp" not in v
        for v in pipeline.values()
    )


def migrate_pipeline_records(xmp_path: str | Path) -> bool:
    """Rewrite legacy {"completed": ts} pipeline entries to new schema in-place. Returns True if file was modified."""
    path = Path(xmp_path)
    if not path.is_file():
        return False
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return False
    desc = _get_rdf_desc(tree)
    if desc is None:
        return False
    detections = _read_detections_payload(desc)
    pipeline = detections.get("pipeline")
    if not isinstance(pipeline, dict):
        return False
    changed = False
    new_pipeline: dict[str, object] = {}
    for k, v in pipeline.items():
        if isinstance(v, dict) and "completed" in v and "timestamp" not in v:
            entry = dict(v)
            entry["timestamp"] = entry["completed"]
            entry.setdefault("result", "ok")
            entry.setdefault("input_hash", "")
            new_pipeline[k] = entry
            changed = True
        else:
            new_pipeline[k] = v
    if changed:
        detections["pipeline"] = new_pipeline
        _write_detections_payload(tree, path, detections)
    return changed


def find_sidecars_with_legacy_pipeline_records(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.xmp") if _has_legacy_pipeline_entries(p)]


def migrate_tree_pipeline_records(root: Path) -> dict[str, int]:
    migrated = 0
    skipped = 0
    for xmp_path in root.rglob("*.xmp"):
        if migrate_pipeline_records(xmp_path):
            migrated += 1
        else:
            skipped += 1
    return {"migrated": migrated, "skipped": skipped}


def read_pipeline_state(xmp_path: str | Path) -> dict[str, object]:
    path = Path(xmp_path)
    if not path.is_file():
        return {}
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return {}
    desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
    if desc is None:
        return {}
    pipeline = _read_detections_payload(desc).get("pipeline")
    if not isinstance(pipeline, dict):
        return {}
    return {k: _normalize_pipeline_entry(v) for k, v in pipeline.items()}


def read_pipeline_step(xmp_path: str | Path, step_name: str) -> dict[str, object] | None:
    step = read_pipeline_state(xmp_path).get(step_name)
    return dict(step) if isinstance(step, dict) else None


def write_pipeline_step(
    xmp_path: str | Path,
    step_name: str,
    *,
    model: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    path = Path(xmp_path)
    tree = _load_or_create_xmp_tree(path)
    desc = _get_or_create_rdf_desc(tree)
    detections = _read_detections_payload(desc)
    pipeline = dict(detections.get("pipeline") or {})
    entry: dict[str, object] = {"completed": _xmp_datetime_now()}
    if model is not None:
        entry["model"] = str(model)
    if extra:
        entry.update(extra)
    pipeline[step_name] = entry
    detections["pipeline"] = pipeline
    _write_detections_payload(tree, path, detections)


def write_pipeline_steps(xmp_path: str | Path, updates: dict[str, dict]) -> None:
    """Merge new-schema step records into imago:Detections["pipeline"], preserving existing keys."""
    if not updates:
        return
    path = Path(xmp_path)
    tree = _load_or_create_xmp_tree(path)
    desc = _get_or_create_rdf_desc(tree)
    detections = _read_detections_payload(desc)
    pipeline = dict(detections.get("pipeline") or {})
    pipeline.update(updates)
    detections["pipeline"] = pipeline
    _write_detections_payload(tree, path, detections)


def clear_pipeline_steps(xmp_path: str | Path, step_names: list[str]) -> None:
    path = Path(xmp_path)
    if not path.is_file():
        return
    tree = _load_or_create_xmp_tree(path)
    desc = _get_or_create_rdf_desc(tree)
    detections = _read_detections_payload(desc)
    pipeline = dict(detections.get("pipeline") or {})
    changed = False
    for step_name in step_names:
        if step_name in pipeline:
            del pipeline[step_name]
            changed = True
    if not changed:
        return
    if pipeline:
        detections["pipeline"] = pipeline
    else:
        detections.pop("pipeline", None)
    _write_detections_payload(tree, path, detections)


def is_step_stale(step_name: str, depends_on: list[str], pipeline_state: dict[str, object]) -> bool:
    """Return True if step has no completed entry or any dependency's completed is newer."""
    from datetime import datetime

    def _parse_ts(entry: object) -> datetime | None:
        if not isinstance(entry, dict):
            return None
        # Support both new "timestamp" and legacy "completed" keys
        ts_str = str(entry.get("timestamp") or entry.get("completed") or "").strip()
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return None

    step_entry = pipeline_state.get(step_name)
    step_ts = _parse_ts(step_entry)
    if step_ts is None:
        return True

    for dep_id in depends_on:
        dep_ts = _parse_ts(pipeline_state.get(dep_id))
        if dep_ts is not None and dep_ts > step_ts:
            return True

    return False


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
    try:
        path = Path(sidecar_path)
        if not path.is_file():
            return []
        tree = ET.parse(path)
        desc = _get_rdf_desc(tree)  # type: ignore[arg-type]
        if desc is None:
            return []
        return _read_locations_shown_from_desc(desc)
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
    detections_payload = _read_detections_payload(desc) or None
    processing_meta: dict[str, object] = dict((detections_payload or {}).get("processing") or {})
    processing_history = _read_processing_history(desc)
    processing_state = _derive_processing_state(processing_history)
    dc_date_values = _normalize_dc_dates(_get_seq_values(desc, f"{{{DC_NS}}}date"))
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - defensive import fallback
        _xmp_gps_to_decimal = None
    location_payload = dict((detections_payload or {}).get("location") or {})
    location_city = str(desc.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or "").strip()
    location_state = str(desc.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or "").strip()
    location_country = str(desc.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or "").strip()
    location_sublocation = str(desc.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or "").strip()
    location_created = str(desc.findtext(f"{{{IPTC_EXT_NS}}}LocationCreated", default="") or "").strip()
    if not location_city:
        location_city = str(location_payload.get("city") or "").strip()
    if not location_state:
        location_state = str(location_payload.get("state") or "").strip()
    if not location_country:
        location_country = str(location_payload.get("country") or "").strip()
    if not location_sublocation:
        location_sublocation = str(location_payload.get("sublocation") or "").strip()
    if not location_created:
        location_created = _format_location_created(
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            location_sublocation=location_sublocation,
        )
    description_role = _description_role_for_sidecar_path(path)
    legacy_crop_parent_ocr = (
        str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "").strip()
        if description_role == DESCRIPTION_ROLE_CROP
        else ""
    )
    return {
        "creator_tool": str(desc.findtext(f"{{{XMP_NS}}}CreatorTool", default="") or "").strip(),
        "create_date": _normalize_xmp_datetime(str(desc.findtext(f"{{{XMP_NS}}}CreateDate", default="") or "").strip()),
        "dc_date": dc_date_values[0] if dc_date_values else "",
        "dc_date_values": dc_date_values,
        "date_time_original": _normalize_exif_date_time_original(
            str(desc.findtext(f"{{{EXIF_NS}}}DateTimeOriginal", default="") or "").strip()
        ),
        "title": _get_alt_text(desc, f"{{{DC_NS}}}title", prefer_lang="x-default"),
        "description": _get_description_value(desc, legacy_caption_first=description_role == DESCRIPTION_ROLE_CROP),
        "album_title": str(
            desc.findtext(f"{{{XMPDM_NS}}}album", default="")
            or desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="")
            or ""
        ).strip(),
        "source_text": str(desc.findtext(f"{{{DC_NS}}}source", default="") or "").strip(),
        "gps_latitude": (
            _xmp_gps_to_decimal(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default=""), axis="lat")
            if _xmp_gps_to_decimal is not None
            else str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or "").strip()
        ),
        "gps_longitude": (
            _xmp_gps_to_decimal(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default=""), axis="lon")
            if _xmp_gps_to_decimal is not None
            else str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or "").strip()
        ),
        "location_city": location_city,
        "location_state": location_state,
        "location_country": location_country,
        "location_sublocation": location_sublocation,
        "location_created": location_created,
        "ocr_text": str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "").strip(),
        "parent_ocr_text": (
            str(desc.findtext(f"{{{IMAGO_NS}}}ParentOCRText", default="") or "").strip() or legacy_crop_parent_ocr
        ),
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
            else (
                bool(processing_meta["people_detected"])
                if isinstance(processing_meta.get("people_detected"), bool)
                else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleDetected")
            )
        ),
        "people_identified": (
            processing_state["people_identified"]
            if "people_identified" in processing_state
            else (
                bool(processing_meta["people_identified"])
                if isinstance(processing_meta.get("people_identified"), bool)
                else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}PeopleIdentified")
            )
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
    parent_ocr_text: str = "",
    ocr_lang: str = "",
    author_text: str = "",
    scene_text: str = "",
    description_role: str = DESCRIPTION_ROLE_PLAIN,
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
    del subphotos, scan_number
    desc = _get_or_create_rdf_desc(tree)
    existing_detections_payload = _read_detections_payload(desc)
    merged_subjects = _dedupe(_get_bag_values(desc, f"{{{DC_NS}}}subject") + list(subjects or []))
    title = _coalesce_text(title, _get_alt_text(desc, f"{{{DC_NS}}}title", prefer_lang="x-default"))
    title_source = _coalesce_text(title_source, str(desc.findtext(f"{{{IMAGO_NS}}}TitleSource", default="") or ""))
    description = _coalesce_text(
        description,
        _get_description_value(desc, legacy_caption_first=description_role == DESCRIPTION_ROLE_CROP),
    )
    existing_dc_dates = _normalize_dc_dates(_get_seq_values(desc, f"{{{DC_NS}}}date"))
    normalized_dc_dates = _dedupe(existing_dc_dates + _normalize_dc_dates(dc_date)) or existing_dc_dates
    create_date = _coalesce_text(
        _normalize_xmp_datetime(create_date),
        _normalize_xmp_datetime(str(desc.findtext(f"{{{XMP_NS}}}CreateDate", default="") or "").strip()),
    )
    date_time_original = _resolve_date_time_original(
        dc_date=normalized_dc_dates,
        date_time_original=_coalesce_text(
            date_time_original,
            str(desc.findtext(f"{{{EXIF_NS}}}DateTimeOriginal", default="") or "").strip(),
        ),
    )
    album_title = _coalesce_text(
        album_title,
        str(
            desc.findtext(f"{{{XMPDM_NS}}}album", default="")
            or desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="")
            or ""
        ).strip(),
    )
    gps_latitude = _coalesce_gps(
        gps_latitude,
        str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or ""),
        axis="lat",
    )
    gps_longitude = _coalesce_gps(
        gps_longitude,
        str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or ""),
        axis="lon",
    )
    location_city = _coalesce_text(location_city, str(desc.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or ""))
    location_state = _coalesce_text(location_state, str(desc.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or ""))
    location_country = _coalesce_text(
        location_country,
        str(desc.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or ""),
    )
    location_sublocation = _coalesce_text(
        location_sublocation,
        str(desc.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or ""),
    )
    location_created = _format_location_created(
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
        location_sublocation=location_sublocation,
    ) or str(desc.findtext(f"{{{IPTC_EXT_NS}}}LocationCreated", default="") or "").strip()
    source_text = _coalesce_text(source_text, str(desc.findtext(f"{{{DC_NS}}}source", default="") or ""))
    existing_ocr_text = str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "")
    existing_parent_ocr_text = str(desc.findtext(f"{{{IMAGO_NS}}}ParentOCRText", default="") or "")
    if description_role == DESCRIPTION_ROLE_CROP:
        parent_ocr_text = _coalesce_text(parent_ocr_text, existing_parent_ocr_text or existing_ocr_text)
        ocr_text = _coalesce_text(ocr_text, existing_ocr_text if existing_parent_ocr_text else "")
    else:
        ocr_text = _coalesce_text(ocr_text, existing_ocr_text)
        parent_ocr_text = _coalesce_text(parent_ocr_text, existing_parent_ocr_text)
    ocr_lang = _coalesce_text(ocr_lang, str(desc.findtext(f"{{{IMAGO_NS}}}OCRLang", default="") or ""))
    author_text = _coalesce_text(author_text, str(desc.findtext(f"{{{IMAGO_NS}}}AuthorText", default="") or ""))
    scene_text = _coalesce_text(scene_text, str(desc.findtext(f"{{{IMAGO_NS}}}SceneText", default="") or ""))
    ocr_authority_source = _coalesce_text(
        ocr_authority_source,
        str(desc.findtext(f"{{{IMAGO_NS}}}OCRAuthoritySource", default="") or ""),
    )
    stitch_key = _coalesce_text(stitch_key, str(desc.findtext(f"{{{IMAGO_NS}}}StitchKey", default="") or ""))
    merged_locations_shown = list(locations_shown) if locations_shown else _read_locations_shown_from_desc(desc)
    merged_detections_payload = detections_payload if detections_payload is not None else (existing_detections_payload or None)
    merged_detections_payload = _with_location_detections(
        merged_detections_payload,
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
        location_sublocation=location_sublocation,
    )
    merged_detections_payload = _with_processing_state(
        merged_detections_payload,
        people_detected=people_detected,
        people_identified=people_identified,
    )

    _set_bag(desc, f"{{{DC_NS}}}subject", merged_subjects)
    _set_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", person_names)
    _set_alt_text(desc, f"{{{DC_NS}}}title", title)
    _set_description_with_text_layers(
        desc,
        f"{{{DC_NS}}}description",
        description_role,
        description,
        ocr_text,
        author_text,
        scene_text,
    )
    _set_seq_text(desc, f"{{{DC_NS}}}date", normalized_dc_dates)
    _set_simple_text(desc, f"{{{EXIF_NS}}}DateTimeOriginal", date_time_original)
    _set_simple_text(desc, f"{{{XMP_NS}}}CreateDate", create_date)
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AlbumTitle", str(album_title or "").strip())
    _set_gps_fields(desc, gps_latitude, gps_longitude)
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())
    _set_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation or "").strip())
    _set_simple_text(
        desc,
        f"{{{IPTC_EXT_NS}}}LocationCreated",
        location_created,
    )
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
    _set_simple_text(desc, f"{{{IMAGO_NS}}}ParentOCRText", str(parent_ocr_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRLang", str(ocr_lang or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AuthorText", str(author_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}SceneText", str(scene_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}TitleSource", str(title_source or "").strip())
    _set_simple_text(
        desc,
        f"{{{IMAGO_NS}}}OCRAuthoritySource",
        str(ocr_authority_source or "").strip(),
    )
    if merged_detections_payload:
        _set_simple_text(
            desc,
            f"{{{IMAGO_NS}}}Detections",
            json.dumps(merged_detections_payload, ensure_ascii=False, sort_keys=True),
        )
    else:
        _set_simple_text(desc, f"{{{IMAGO_NS}}}Detections", "")
    if isinstance(merged_detections_payload, dict) and "people" in merged_detections_payload:
        _replace_iptc_face_regions(
            desc,
            [row for row in list(merged_detections_payload.get("people") or []) if isinstance(row, dict)],
            image_width,
            image_height,
        )
    _set_locations_shown_bag(desc, merged_locations_shown)
    _set_processing_history(desc, [])
    for legacy_tag in (
        f"{{{IMAGO_NS}}}StitchKey",
        f"{{{IMAGO_NS}}}OcrRan",
        f"{{{IMAGO_NS}}}PeopleDetected",
        f"{{{IMAGO_NS}}}PeopleIdentified",
        f"{{{IMAGO_NS}}}SubPhotos",
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
    parent_ocr_text: str = "",
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
    description_role = _description_role_for_sidecar_path(path)
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
            description_role=description_role,
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
            parent_ocr_text=parent_ocr_text,
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
            description_role=description_role,
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
            parent_ocr_text=parent_ocr_text,
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


def propagate_archive_copy_safe_fields(
    view_xmp_path: str | Path,
    archive_sidecar_path: str | Path,
) -> bool:
    """Populate copy-safe fields absent in view_xmp_path from archive_sidecar_path.

    Only writes a field when the view's current value is empty. Returns True if any
    field was written.
    """
    view_xmp = Path(view_xmp_path)
    archive_xmp = Path(archive_sidecar_path)

    if not view_xmp.is_file() or not archive_xmp.is_file():
        return False

    archive_state = read_ai_sidecar_state(archive_xmp)
    if not isinstance(archive_state, dict):
        return False

    view_state = read_ai_sidecar_state(view_xmp) or {}

    archive_locations = read_locations_shown(archive_xmp)
    view_locations = read_locations_shown(view_xmp)
    locations_shown = archive_locations if (archive_locations and not view_locations) else None

    archive_description = str(archive_state.get("description") or "").strip()
    view_description = str(view_state.get("description") or "").strip()
    description = archive_description if (archive_description and not view_description) else ""

    archive_ocr_text = str(archive_state.get("ocr_text") or "").strip()
    view_ocr_text = str(view_state.get("ocr_text") or "").strip()
    ocr_text = archive_ocr_text if (archive_ocr_text and not view_ocr_text) else ""

    gps_latitude = str(view_state.get("gps_latitude") or archive_state.get("gps_latitude") or "")
    gps_longitude = str(view_state.get("gps_longitude") or archive_state.get("gps_longitude") or "")
    location_city = str(view_state.get("location_city") or archive_state.get("location_city") or "")
    location_state = str(view_state.get("location_state") or archive_state.get("location_state") or "")
    location_country = str(view_state.get("location_country") or archive_state.get("location_country") or "")
    location_sublocation = str(view_state.get("location_sublocation") or archive_state.get("location_sublocation") or "")

    has_gps = bool(gps_latitude and gps_longitude)
    has_location = bool(location_city or location_country)

    if not locations_shown and not description and not ocr_text and not has_gps and not has_location:
        return False

    existing_people = read_person_in_image(view_xmp)

    write_xmp_sidecar(
        view_xmp,
        creator_tool=str(view_state.get("creator_tool") or archive_state.get("creator_tool") or ""),
        person_names=existing_people,
        subjects=[],
        description=description,
        ocr_text=ocr_text,
        ocr_lang=str(view_state.get("ocr_lang") or archive_state.get("ocr_lang") or ""),
        author_text=str(view_state.get("author_text") or archive_state.get("author_text") or ""),
        scene_text=str(view_state.get("scene_text") or archive_state.get("scene_text") or ""),
        album_title=str(view_state.get("album_title") or archive_state.get("album_title") or ""),
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
        location_sublocation=location_sublocation,
        source_text=str(view_state.get("source_text") or archive_state.get("source_text") or ""),
        dc_date=str(view_state.get("dc_date") or archive_state.get("dc_date") or ""),
        date_time_original=str(
            view_state.get("date_time_original") or archive_state.get("date_time_original") or ""
        ),
        locations_shown=locations_shown,
    )
    return True


# ---------------------------------------------------------------------------
# MWG-RS region list helpers
# ---------------------------------------------------------------------------


def write_region_list(
    xmp_path: str | Path,
    regions_with_captions: list,
    img_w: int,
    img_h: int,
) -> None:
    """Write (or replace) an mwg-rs:RegionList in the XMP sidecar.

    regions_with_captions is a list of RegionWithCaption objects from
    photoalbums.lib.ai_view_regions.

    The XMP file is created if it does not exist (minimal wrapper).
    Any existing mwg-rs:RegionInfo block is removed and replaced.
    """
    from .ai_view_regions import pixel_to_mwgrs  # pylint: disable=import-outside-toplevel

    path = Path(xmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_regions: list[dict] = []
    if path.is_file():
        try:
            existing_regions = read_region_list(path, img_w, img_h)
        except Exception:
            existing_regions = []
        try:
            tree: ET.ElementTree = ET.parse(str(path))  # type: ignore[assignment]
        except ET.ParseError:
            tree = None  # type: ignore[assignment]
    else:
        tree = None  # type: ignore[assignment]

    if tree is None:
        xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
        xmpmeta.set(f"{{{X_NS}}}xmptk", "imago")
        rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
        ET.SubElement(rdf, _RDF_DESC).set(f"{{{RDF_NS}}}about", "")
        tree = ET.ElementTree(xmpmeta)

    desc = _get_or_create_rdf_desc(tree)

    # Remove existing RegionInfo block
    region_info_tag = f"{{{MWGRS_NS}}}RegionInfo"
    existing = desc.find(region_info_tag)
    if existing is not None:
        desc.remove(existing)

    if not regions_with_captions:
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
        return

    # Build mwg-rs:RegionInfo
    region_info = ET.SubElement(desc, region_info_tag)
    region_info.set(_RDF_PARSE_TYPE, "Resource")

    # mwg-rs:AppliedToDimensions
    applied = ET.SubElement(region_info, f"{{{MWGRS_NS}}}AppliedToDimensions")
    applied.set(_RDF_PARSE_TYPE, "Resource")
    applied.set(f"{{{STAREA_NS}}}w", str(img_w))
    applied.set(f"{{{STAREA_NS}}}h", str(img_h))
    applied.set(f"{{{STAREA_NS}}}unit", "pixel")

    # mwg-rs:RegionList
    region_list_el = ET.SubElement(region_info, f"{{{MWGRS_NS}}}RegionList")
    bag = ET.SubElement(region_list_el, _RDF_BAG)

    for rwc in regions_with_captions:
        r = rwc.region
        cx, cy, nw, nh = pixel_to_mwgrs(r.x, r.y, r.width, r.height, img_w, img_h)
        region_caption = str(rwc.caption or getattr(r, "caption_hint", "") or "").strip()

        li = ET.SubElement(bag, _RDF_LI)
        li.set(_RDF_PARSE_TYPE, "Resource")
        li.set(f"{{{MWGRS_NS}}}Type", "Photo")
        li.set(f"{{{MWGRS_NS}}}Name", region_caption)

        # stArea coordinates (centre-point, normalised)
        li.set(f"{{{STAREA_NS}}}x", f"{cx:.6f}")
        li.set(f"{{{STAREA_NS}}}y", f"{cy:.6f}")
        li.set(f"{{{STAREA_NS}}}w", f"{nw:.6f}")
        li.set(f"{{{STAREA_NS}}}h", f"{nh:.6f}")
        li.set(f"{{{STAREA_NS}}}unit", "normalized")

        caption_hint = str(getattr(r, "caption_hint", "") or "").strip()
        if caption_hint:
            li.set(f"{{{IMAGO_NS}}}CaptionHint", caption_hint)

        person_names = list(getattr(r, "person_names", ()) or ())
        person_names = [str(n).strip() for n in person_names if str(n).strip()]
        if person_names:
            pn_el = ET.SubElement(li, f"{{{IMAGO_NS}}}PersonNames")
            bag = ET.SubElement(pn_el, _RDF_BAG)
            for name in person_names:
                item = ET.SubElement(bag, _RDF_LI)
                item.text = name

        existing_region = existing_regions[r.index] if 0 <= int(r.index) < len(existing_regions) else {}
        location_payload = dict(getattr(r, "location_payload", {}) or {})
        location_override = dict(getattr(r, "location_override", {}) or {}) or dict(
            existing_region.get("location_override") or {}
        )
        _add_region_location_struct(li, f"{{{IMAGO_NS}}}LocationAssigned", location_payload)
        _add_region_location_struct(li, f"{{{IMAGO_NS}}}LocationOverride", location_override)

    # Update modify date
    desc.set(f"{{{XMP_NS}}}ModifyDate", _xmp_datetime_now())

    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def read_region_list(xmp_path: str | Path, img_w: int, img_h: int) -> list[dict]:
    """Read the mwg-rs:RegionList from an XMP sidecar.

    Returns a list of dicts with keys:
      index, name, x, y, width, height (pixel, top-left),
      cx, cy, nw, nh (normalised MWG-RS),
      caption, type.
    Returns [] if no region list is present or on parse error.
    """
    path = Path(xmp_path)
    if not path.is_file():
        return []
    try:
        tree: ET.ElementTree = ET.parse(str(path))  # type: ignore[assignment]
    except ET.ParseError:
        return []

    results: list[dict] = []
    idx = 0
    for li in tree.iter(f"{{{RDF_NS}}}li"):
        rtype = li.get(f"{{{MWGRS_NS}}}Type")
        if rtype != "Photo":
            continue
        cx_t = li.get(f"{{{STAREA_NS}}}x")
        cy_t = li.get(f"{{{STAREA_NS}}}y")
        nw_t = li.get(f"{{{STAREA_NS}}}w")
        nh_t = li.get(f"{{{STAREA_NS}}}h")
        if not all((cx_t, cy_t, nw_t, nh_t)):
            continue
        try:
            cx = float(cx_t)
            cy = float(cy_t)
            nw = float(nw_t)
            nh = float(nh_t)
        except (TypeError, ValueError):
            continue
        px = max(0, int(round((cx - nw / 2.0) * img_w)))
        py = max(0, int(round((cy - nh / 2.0) * img_h)))
        pw = max(1, int(round(nw * img_w)))
        ph = max(1, int(round(nh * img_h)))

        caption = str(li.get(f"{{{MWGRS_NS}}}Name") or "").strip()
        if not caption:
            desc_el = li.find(f".//{{{DC_NS}}}description")
            if desc_el is not None:
                li_text = desc_el.find(f".//{{{RDF_NS}}}li")
                if li_text is not None and li_text.text:
                    caption = li_text.text.strip()

        caption_hint = str(li.get(f"{{{IMAGO_NS}}}CaptionHint") or "").strip()

        person_names: list[str] = []
        pn_el = li.find(f"{{{IMAGO_NS}}}PersonNames")
        if pn_el is not None:
            for pn_li in pn_el.iter(f"{{{RDF_NS}}}li"):
                name = str(pn_li.text or "").strip()
                if name:
                    person_names.append(name)

        results.append(
            {
                "index": idx,
                "name": caption or f"photo_{idx + 1}",
                "x": px,
                "y": py,
                "width": pw,
                "height": ph,
                "cx": cx,
                "cy": cy,
                "nw": nw,
                "nh": nh,
                "caption": caption,
                "caption_hint": caption_hint,
                "location_payload": _read_region_location_struct(li, f"{{{IMAGO_NS}}}LocationAssigned"),
                "location_override": _read_region_location_struct(li, f"{{{IMAGO_NS}}}LocationOverride"),
                "person_names": person_names,
                "type": rtype,
            }
        )
        idx += 1
    return results
