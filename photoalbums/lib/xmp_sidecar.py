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
    parts = _partial_dc_date_parts(text)
    if not parts:
        return ""
    return _normalize_partial_dc_date_parts(parts)


def _partial_dc_date_parts(text: str) -> list[str]:
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
        return []
    return parts


def _normalize_partial_dc_date_parts(parts: list[str]) -> str:
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
    location_address: str = "",
    location_city: str = "",
    location_state: str = "",
    location_country: str = "",
    location_sublocation: str = "",
) -> str:
    address = str(location_address or "").strip()
    if address:
        return address
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
    for target_key, value in (
        ("city", location_city),
        ("state", location_state),
        ("country", location_country),
        ("sublocation", location_sublocation),
    ):
        _fill_missing_text(location, target_key, value)
    if location:
        payload["location"] = location
    return payload or None


def _fill_missing_text(target: dict, key: str, value: object) -> None:
    clean = str(value or "").strip()
    if clean and not str(target.get(key) or "").strip():
        target[key] = clean


def _location_shown_identity(location: dict) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "name",
        "world_region",
        "country_code",
        "country_name",
        "province_or_state",
        "city",
        "sublocation",
        "gps_latitude",
        "gps_longitude",
    ):
        values.append(" ".join(str(location.get(key) or "").casefold().replace(",", " ").split()))
    return tuple(values)


def _dedupe_locations_shown(locations: list[dict] | None) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for location in list(locations or []):
        if not isinstance(location, dict):
            continue
        if not any(str(value or "").strip() for value in location.values()):
            continue
        identity = _location_shown_identity(location)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(location)
    return deduped


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
        _add_location_shown_text_fields(li, loc)
        _add_location_shown_gps_fields(li, loc)


def _add_location_shown_text_fields(parent: ET.Element, loc: dict) -> None:
    for key, tag in (
        ("world_region", f"{{{IPTC_EXT_NS}}}WorldRegion"),
        ("country_code", f"{{{IPTC_EXT_NS}}}CountryCode"),
        ("country_name", f"{{{IPTC_EXT_NS}}}CountryName"),
        ("province_or_state", f"{{{IPTC_EXT_NS}}}ProvinceState"),
        ("city", f"{{{IPTC_EXT_NS}}}City"),
        ("sublocation", f"{{{IPTC_EXT_NS}}}Sublocation"),
    ):
        clean = str(loc.get(key) or "").strip()
        if clean:
            ET.SubElement(parent, tag).text = _normalize_xmp_text(clean)


def _add_location_shown_gps_fields(parent: ET.Element, loc: dict) -> None:
    for key, tag, axis in (
        ("gps_latitude", f"{{{EXIF_NS}}}GPSLatitude", "lat"),
        ("gps_longitude", f"{{{EXIF_NS}}}GPSLongitude", "lon"),
    ):
        clean = str(loc.get(key) or "").strip()
        if clean:
            ET.SubElement(parent, tag).text = _format_xmp_gps_coordinate(clean, axis=axis)


def _set_locations_shown_bag(parent: ET.Element, locations: list[dict]) -> None:
    existing = parent.find(f"{{{IPTC_EXT_NS}}}LocationShown")
    if existing is not None:
        parent.remove(existing)
    _add_locations_shown_bag(parent, _dedupe_locations_shown(locations))


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
        ET.SubElement(field, f"{{{IPTC_EXT_NS}}}City").text = _normalize_xmp_text(clean_payload["city"])
    if clean_payload["state"]:
        ET.SubElement(field, f"{{{IPTC_EXT_NS}}}ProvinceState").text = _normalize_xmp_text(clean_payload["state"])
    if clean_payload["country"]:
        ET.SubElement(field, f"{{{IPTC_EXT_NS}}}CountryName").text = _normalize_xmp_text(clean_payload["country"])
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
            "city": str(field.findtext(f"{{{IPTC_EXT_NS}}}City", default="") or "").strip(),
            "state": str(field.findtext(f"{{{IPTC_EXT_NS}}}ProvinceState", default="") or "").strip(),
            "country": str(field.findtext(f"{{{IPTC_EXT_NS}}}CountryName", default="") or "").strip(),
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


def _add_xmp_date_fields(
    desc: ET.Element,
    *,
    dc_date: str | list[str] | tuple[str, ...],
    date_time_original: str,
    create_date: str,
) -> None:
    _add_xmp_date_fields(
        desc,
        dc_date=dc_date,
        date_time_original=date_time_original,
        create_date=create_date,
    )


def _add_xmp_location_fields(
    desc: ET.Element,
    *,
    description_role: str,
    gps_latitude: str,
    gps_longitude: str,
    location_address: str,
    location_city: str,
    location_state: str,
    location_country: str,
    location_sublocation: str,
) -> None:
    if str(gps_latitude or "").strip() and str(gps_longitude or "").strip():
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSLatitude", _format_xmp_gps_coordinate(gps_latitude, axis="lat"))
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSLongitude", _format_xmp_gps_coordinate(gps_longitude, axis="lon"))
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSMapDatum", "WGS-84")
        _add_simple_text(desc, f"{{{EXIF_NS}}}GPSVersionID", "2.3.0.0")
    write_photoshop_location = description_role != DESCRIPTION_ROLE_CROP
    if write_photoshop_location and str(location_city or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    if write_photoshop_location and str(location_state or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    if write_photoshop_location and str(location_country or "").strip():
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())
    if str(location_sublocation or "").strip():
        _add_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation or "").strip())
    _add_simple_text(
        desc,
        f"{{{IPTC_EXT_NS}}}LocationCreated",
        _format_location_created(
            location_address=location_address,
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            location_sublocation=location_sublocation,
        ),
    )


def _add_xmp_text_fields(
    desc: ET.Element,
    *,
    description_role: str,
    ocr_text: str,
    parent_ocr_text: str,
    ocr_lang: str,
    author_text: str,
    scene_text: str,
    title_source: str,
    ocr_authority_source: str,
) -> None:
    write_ocr_text_fields = description_role != DESCRIPTION_ROLE_CROP
    text_fields = (
        (write_ocr_text_fields, f"{{{IMAGO_NS}}}OCRText", _normalize_xmp_text(ocr_text, multiline=True)),
        (
            write_ocr_text_fields,
            f"{{{IMAGO_NS}}}ParentOCRText",
            _normalize_xmp_text(parent_ocr_text, multiline=True),
        ),
        (True, f"{{{IMAGO_NS}}}OCRLang", _normalize_xmp_text(ocr_lang)),
        (True, f"{{{IMAGO_NS}}}AuthorText", _normalize_xmp_text(author_text, multiline=True)),
        (True, f"{{{IMAGO_NS}}}SceneText", _normalize_xmp_text(scene_text, multiline=True)),
        (True, f"{{{IMAGO_NS}}}TitleSource", _normalize_xmp_text(title_source)),
        (True, f"{{{IMAGO_NS}}}OCRAuthoritySource", _normalize_xmp_text(ocr_authority_source)),
    )
    for should_write, tag, text in text_fields:
        if should_write and text:
            _add_simple_text(desc, tag, text)


def _add_xmp_detection_fields(
    desc: ET.Element,
    *,
    detections_payload: dict | None,
    people_detected: bool,
    people_identified: bool,
    image_width: int,
    image_height: int,
    locations_shown: list[dict] | None,
) -> None:
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


def build_xmp_tree(
    *,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    location_address: str = "",
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
    _add_xmp_location_fields(
        desc,
        description_role=description_role,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_address=location_address,
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
        location_sublocation=location_sublocation,
    )
    if page_number > 0:
        _add_simple_text(desc, f"{{{PHOTOSHOP_NS}}}PageNumber", str(page_number))
    _add_simple_text(desc, f"{{{DC_NS}}}source", _normalize_xmp_text(source_text))

    _add_xmp_text_fields(
        desc,
        description_role=description_role,
        ocr_text=ocr_text,
        parent_ocr_text=parent_ocr_text,
        ocr_lang=ocr_lang,
        author_text=author_text,
        scene_text=scene_text,
        title_source=title_source,
        ocr_authority_source=ocr_authority_source,
    )
    _add_xmp_detection_fields(
        desc,
        detections_payload=detections_payload,
        people_detected=people_detected,
        people_identified=people_identified,
        image_width=image_width,
        image_height=image_height,
        locations_shown=locations_shown,
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
        legacy_caption = _get_alt_text(
            parent, f"{{{DC_NS}}}description", prefer_lang="x-caption", fallback_to_any=False
        )
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
        return _dedupe_locations_shown(rows)
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
    return any(isinstance(v, dict) and "completed" in v and "timestamp" not in v for v in pipeline.values())


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


def _read_sidecar_location_state(
    desc: ET.Element,
    *,
    detections_payload: dict | None,
    description_role: str,
    xmp_gps_to_decimal,
) -> dict[str, str]:
    location_payload = dict((detections_payload or {}).get("location") or {})
    location_state = {
        "gps_latitude": (
            xmp_gps_to_decimal(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default=""), axis="lat")
            if xmp_gps_to_decimal is not None
            else str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or "").strip()
        ),
        "gps_longitude": (
            xmp_gps_to_decimal(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default=""), axis="lon")
            if xmp_gps_to_decimal is not None
            else str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or "").strip()
        ),
        "location_city": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or "").strip(),
        "location_state": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or "").strip(),
        "location_country": str(desc.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or "").strip(),
        "location_sublocation": str(desc.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or "").strip(),
        "location_created": str(desc.findtext(f"{{{IPTC_EXT_NS}}}LocationCreated", default="") or "").strip(),
    }
    _fill_location_state_from_payload(
        location_state,
        location_payload,
        (
            ("location_city", "city"),
            ("location_state", "state"),
            ("location_country", "country"),
            ("location_sublocation", "sublocation"),
        ),
    )

    crop_locations_shown = _read_locations_shown_from_desc(desc)
    crop_location = (
        crop_locations_shown[0] if description_role == DESCRIPTION_ROLE_CROP and len(crop_locations_shown) == 1 else {}
    )
    _fill_location_state_from_payload(
        location_state,
        crop_location,
        (
            ("location_city", "city"),
            ("location_state", "province_or_state"),
            ("location_country", "country_name"),
            ("location_sublocation", "sublocation"),
            ("gps_latitude", "gps_latitude"),
            ("gps_longitude", "gps_longitude"),
        ),
    )

    location_state["location_created"] = location_state["location_created"] or _format_location_created(
        location_city=location_state["location_city"],
        location_state=location_state["location_state"],
        location_country=location_state["location_country"],
        location_sublocation=location_state["location_sublocation"],
    )
    return location_state


def _fill_location_state_from_payload(
    location_state: dict[str, str],
    payload: dict,
    fields: tuple[tuple[str, str], ...],
) -> None:
    for target_key, payload_key in fields:
        location_state[target_key] = location_state[target_key] or str(payload.get(payload_key) or "").strip()


def _processing_bool(
    desc: ET.Element,
    processing_state: dict,
    processing_meta: dict,
    state_key: str,
    meta_key: str,
    tag: str,
) -> bool:
    if state_key in processing_state:
        return bool(processing_state[state_key])
    if isinstance(processing_meta.get(meta_key), bool):
        return bool(processing_meta[meta_key])
    return _read_xmp_bool(desc, tag)


def _processing_text(processing_state: dict, desc: ET.Element, state_key: str, tag: str) -> str:
    return str(processing_state.get(state_key) or desc.findtext(tag, default="") or "").strip()


def _sidecar_processing_meta_values(
    desc: ET.Element,
    *,
    processing_state: dict,
    processing_meta: dict,
) -> dict[str, object]:
    return {
        "ocr_authority_source": _processing_text(
            processing_state,
            desc,
            "ocr_authority_source",
            f"{{{IMAGO_NS}}}OCRAuthoritySource",
        ),
        "stitch_key": _processing_text(processing_state, desc, "stitch_key", f"{{{IMAGO_NS}}}StitchKey"),
        "ocr_ran": (
            bool(processing_state["ocr_ran"])
            if "ocr_ran" in processing_state
            else _read_xmp_bool(desc, f"{{{IMAGO_NS}}}OcrRan")
        ),
        "people_detected": _processing_bool(
            desc,
            processing_state,
            processing_meta,
            "people_detected",
            "people_detected",
            f"{{{IMAGO_NS}}}PeopleDetected",
        ),
        "people_identified": _processing_bool(
            desc,
            processing_state,
            processing_meta,
            "people_identified",
            "people_identified",
            f"{{{IMAGO_NS}}}PeopleIdentified",
        ),
    }


def _sidecar_processing_signature_values(processing_meta: dict) -> dict[str, object]:
    return {
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


def _optional_xmp_gps_to_decimal():
    try:
        from .ai_location import _xmp_gps_to_decimal  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - defensive import fallback
        return None
    return _xmp_gps_to_decimal


def _sidecar_role_text(desc: ET.Element, *, description_role: str, tag: str) -> str:
    if description_role == DESCRIPTION_ROLE_CROP:
        return ""
    return str(desc.findtext(tag, default="") or "").strip()


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
    description_role = _description_role_for_sidecar_path(path)
    location_state = _read_sidecar_location_state(
        desc,
        detections_payload=detections_payload,
        description_role=description_role,
        xmp_gps_to_decimal=_optional_xmp_gps_to_decimal(),
    )
    processing_values = _sidecar_processing_meta_values(
        desc,
        processing_state=processing_state,
        processing_meta=processing_meta,
    )
    return {
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
        "gps_latitude": location_state["gps_latitude"],
        "gps_longitude": location_state["gps_longitude"],
        "location_city": location_state["location_city"],
        "location_state": location_state["location_state"],
        "location_country": location_state["location_country"],
        "location_sublocation": location_state["location_sublocation"],
        "location_created": location_state["location_created"],
        "ocr_text": _sidecar_role_text(desc, description_role=description_role, tag=f"{{{IMAGO_NS}}}OCRText"),
        "parent_ocr_text": _sidecar_role_text(
            desc,
            description_role=description_role,
            tag=f"{{{IMAGO_NS}}}ParentOCRText",
        ),
        "ocr_lang": str(desc.findtext(f"{{{IMAGO_NS}}}OCRLang", default="") or "").strip(),
        "author_text": str(desc.findtext(f"{{{IMAGO_NS}}}AuthorText", default="") or "").strip(),
        "scene_text": str(desc.findtext(f"{{{IMAGO_NS}}}SceneText", default="") or "").strip(),
        "title_source": str(desc.findtext(f"{{{IMAGO_NS}}}TitleSource", default="") or "").strip(),
        "ocr_authority_source": processing_values["ocr_authority_source"],
        "stitch_key": processing_values["stitch_key"],
        "processing_history": processing_history,
        "detections": detections_payload,
        "ocr_ran": processing_values["ocr_ran"],
        "people_detected": processing_values["people_detected"],
        "people_identified": processing_values["people_identified"],
        **_sidecar_processing_signature_values(processing_meta),
    }


def sidecar_has_expected_ai_fields(
    sidecar_path: str | Path,
    *,
    enable_people: bool,
    enable_objects: bool,
    ocr_engine: str,
    caption_engine: str,
) -> bool:
    state = read_ai_sidecar_state(sidecar_path)
    if not isinstance(state, dict):
        return False
    detections = state.get("detections")
    if not isinstance(detections, dict):
        return False
    if not _sidecar_has_expected_detection_blocks(
        detections,
        enable_people=enable_people,
        enable_objects=enable_objects,
        ocr_engine=ocr_engine,
        caption_engine=caption_engine,
    ):
        return False
    return _sidecar_text_fields_are_expected(state)


def _sidecar_has_expected_detection_blocks(
    detections: dict,
    *,
    enable_people: bool,
    enable_objects: bool,
    ocr_engine: str,
    caption_engine: str,
) -> bool:
    if bool(enable_people) and not _people_detection_block_is_expected(detections):
        return False
    if bool(enable_objects) and not isinstance(detections.get("objects"), list):
        return False
    if str(ocr_engine or "").strip().lower() != "none" and not isinstance(detections.get("ocr"), dict):
        return False
    return str(caption_engine or "").strip().lower() == "none" or isinstance(detections.get("caption"), dict)


def _people_detection_block_is_expected(detections: dict) -> bool:
    people = detections.get("people")
    if not isinstance(people, list):
        return False
    if not people:
        return True
    return any(isinstance(p, dict) and isinstance(p.get("bbox"), list) and len(p["bbox"]) >= 4 for p in people)


def _caption_reasoning_checker():
    try:
        from .ai_caption import _looks_like_reasoning_or_prompt_echo  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - defensive import fallback
        return None
    return _looks_like_reasoning_or_prompt_echo


def _ocr_reasoning_checker():
    try:
        from .ai_ocr import _looks_like_ocr_reasoning  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - defensive import fallback
        return None
    return _looks_like_ocr_reasoning


def _sidecar_text_fields_are_expected(state: dict) -> bool:
    caption_checker = _caption_reasoning_checker()
    for field_name in ("description", "author_text", "scene_text"):
        value = str(state.get(field_name) or "").strip()
        if value and caption_checker is not None and caption_checker(value):
            return False
    ocr_checker = _ocr_reasoning_checker()
    ocr_text = str(state.get("ocr_text") or "").strip()
    return not (ocr_text and ocr_checker is not None and ocr_checker(ocr_text))


def _merged_xmp_values(
    desc: ET.Element,
    *,
    person_names: list[str],
    subjects: list[str],
    title: str,
    title_source: str,
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    location_address: str,
    location_city: str,
    location_state: str,
    location_country: str,
    location_sublocation: str,
    source_text: str,
    ocr_text: str,
    parent_ocr_text: str,
    ocr_lang: str,
    author_text: str,
    scene_text: str,
    description_role: str,
    detections_payload: dict | None,
    ocr_authority_source: str,
    create_date: str,
    dc_date: str | list[str] | tuple[str, ...],
    date_time_original: str,
    replace_dc_date: bool,
    stitch_key: str,
    people_detected: bool,
    people_identified: bool,
    locations_shown: list[dict] | None,
) -> dict[str, object]:
    existing_detections_payload = _read_detections_payload(desc)
    normalized_dc_dates = _merged_dc_dates(desc, dc_date=dc_date, replace_dc_date=replace_dc_date)
    values: dict[str, object] = _merged_base_xmp_values(
        desc,
        person_names=person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=description,
        album_title=album_title,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        source_text=source_text,
        ocr_lang=ocr_lang,
        author_text=author_text,
        scene_text=scene_text,
        ocr_authority_source=ocr_authority_source,
        stitch_key=stitch_key,
        create_date=create_date,
        description_role=description_role,
        normalized_dc_dates=normalized_dc_dates,
        locations_shown=locations_shown,
    )
    values["date_time_original"] = _resolve_date_time_original(
        dc_date=normalized_dc_dates,
        date_time_original=_coalesce_text(
            date_time_original,
            str(desc.findtext(f"{{{EXIF_NS}}}DateTimeOriginal", default="") or "").strip(),
        ),
    )
    values.update(
        _merged_role_xmp_values(
            desc,
            description_role=description_role,
            location_address=location_address,
            location_city=location_city,
            location_state=location_state,
            location_country=location_country,
            location_sublocation=location_sublocation,
            ocr_text=ocr_text,
            parent_ocr_text=parent_ocr_text,
        )
    )
    values["location_created"] = _merged_location_created(desc, values=values, description_role=description_role)
    values["merged_detections_payload"] = _merged_detections_with_processing(
        detections_payload if detections_payload is not None else (existing_detections_payload or None),
        values=values,
        people_detected=people_detected,
        people_identified=people_identified,
    )
    return values


def _merged_dc_dates(
    desc: ET.Element,
    *,
    dc_date: str | list[str] | tuple[str, ...],
    replace_dc_date: bool,
) -> list[str]:
    existing_dc_dates = _normalize_dc_dates(_get_seq_values(desc, f"{{{DC_NS}}}date"))
    incoming_dc_dates = _normalize_dc_dates(dc_date)
    return incoming_dc_dates if replace_dc_date else _dedupe(existing_dc_dates + incoming_dc_dates) or existing_dc_dates


def _merged_base_xmp_values(
    desc: ET.Element,
    *,
    person_names: list[str],
    subjects: list[str],
    title: str,
    title_source: str,
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    source_text: str,
    ocr_lang: str,
    author_text: str,
    scene_text: str,
    ocr_authority_source: str,
    stitch_key: str,
    create_date: str,
    description_role: str,
    normalized_dc_dates: list[str],
    locations_shown: list[dict] | None,
) -> dict[str, object]:
    return {
        "merged_subjects": _dedupe(_get_bag_values(desc, f"{{{DC_NS}}}subject") + list(subjects or [])),
        "person_names": person_names,
        "title": _coalesce_text(title, _get_alt_text(desc, f"{{{DC_NS}}}title", prefer_lang="x-default")),
        "title_source": _coalesce_text(
            title_source, str(desc.findtext(f"{{{IMAGO_NS}}}TitleSource", default="") or "")
        ),
        "description": _coalesce_text(
            description,
            _get_description_value(desc, legacy_caption_first=description_role == DESCRIPTION_ROLE_CROP),
        ),
        "normalized_dc_dates": normalized_dc_dates,
        "create_date": _coalesce_text(
            _normalize_xmp_datetime(create_date),
            _normalize_xmp_datetime(str(desc.findtext(f"{{{XMP_NS}}}CreateDate", default="") or "").strip()),
        ),
        "album_title": _coalesce_text(
            album_title,
            str(
                desc.findtext(f"{{{XMPDM_NS}}}album", default="")
                or desc.findtext(f"{{{IMAGO_NS}}}AlbumTitle", default="")
                or ""
            ).strip(),
        ),
        "gps_latitude": _coalesce_gps(
            gps_latitude,
            str(desc.findtext(f"{{{EXIF_NS}}}GPSLatitude", default="") or ""),
            axis="lat",
        ),
        "gps_longitude": _coalesce_gps(
            gps_longitude,
            str(desc.findtext(f"{{{EXIF_NS}}}GPSLongitude", default="") or ""),
            axis="lon",
        ),
        "source_text": _coalesce_text(source_text, str(desc.findtext(f"{{{DC_NS}}}source", default="") or "")),
        "ocr_lang": _coalesce_text(ocr_lang, str(desc.findtext(f"{{{IMAGO_NS}}}OCRLang", default="") or "")),
        "author_text": _coalesce_text(author_text, str(desc.findtext(f"{{{IMAGO_NS}}}AuthorText", default="") or "")),
        "scene_text": _coalesce_text(scene_text, str(desc.findtext(f"{{{IMAGO_NS}}}SceneText", default="") or "")),
        "ocr_authority_source": _coalesce_text(
            ocr_authority_source,
            str(desc.findtext(f"{{{IMAGO_NS}}}OCRAuthoritySource", default="") or ""),
        ),
        "stitch_key": _coalesce_text(stitch_key, str(desc.findtext(f"{{{IMAGO_NS}}}StitchKey", default="") or "")),
        "merged_locations_shown": (
            list(locations_shown) if locations_shown is not None else _read_locations_shown_from_desc(desc)
        ),
    }


def _merged_role_xmp_values(
    desc: ET.Element,
    *,
    description_role: str,
    location_address: str,
    location_city: str,
    location_state: str,
    location_country: str,
    location_sublocation: str,
    ocr_text: str,
    parent_ocr_text: str,
) -> dict[str, object]:
    if description_role == DESCRIPTION_ROLE_CROP:
        return {
            "location_address": str(location_address or "").strip(),
            "location_city": str(location_city or "").strip(),
            "location_state": str(location_state or "").strip(),
            "location_country": str(location_country or "").strip(),
            "location_sublocation": str(location_sublocation or "").strip(),
            "ocr_text": "",
            "parent_ocr_text": "",
        }
    return {
        "location_address": str(location_address or "").strip(),
        "location_city": _coalesce_text(location_city, str(desc.findtext(f"{{{PHOTOSHOP_NS}}}City", default="") or "")),
        "location_state": _coalesce_text(
            location_state, str(desc.findtext(f"{{{PHOTOSHOP_NS}}}State", default="") or "")
        ),
        "location_country": _coalesce_text(
            location_country,
            str(desc.findtext(f"{{{PHOTOSHOP_NS}}}Country", default="") or ""),
        ),
        "location_sublocation": _coalesce_text(
            location_sublocation,
            str(desc.findtext(f"{{{IPTC_EXT_NS}}}Sublocation", default="") or ""),
        ),
        "ocr_text": _coalesce_text(ocr_text, str(desc.findtext(f"{{{IMAGO_NS}}}OCRText", default="") or "")),
        "parent_ocr_text": _coalesce_text(
            parent_ocr_text,
            str(desc.findtext(f"{{{IMAGO_NS}}}ParentOCRText", default="") or ""),
        ),
    }


def _merged_location_created(desc: ET.Element, *, values: dict[str, object], description_role: str) -> str:
    formatted_location_created = _format_location_created(
        location_address=str(values["location_address"]),
        location_city=str(values["location_city"]),
        location_state=str(values["location_state"]),
        location_country=str(values["location_country"]),
        location_sublocation=str(values["location_sublocation"]),
    )
    if description_role == DESCRIPTION_ROLE_CROP:
        return formatted_location_created
    return (
        formatted_location_created or str(desc.findtext(f"{{{IPTC_EXT_NS}}}LocationCreated", default="") or "").strip()
    )


def _merged_detections_with_processing(
    merged_detections_payload: dict | None,
    *,
    values: dict[str, object],
    people_detected: bool,
    people_identified: bool,
) -> dict | None:
    merged_detections_payload = _with_location_detections(
        merged_detections_payload,
        location_city=str(values["location_city"]),
        location_state=str(values["location_state"]),
        location_country=str(values["location_country"]),
        location_sublocation=str(values["location_sublocation"]),
    )
    return _with_processing_state(
        merged_detections_payload,
        people_detected=people_detected,
        people_identified=people_identified,
    )


def _merge_xmp_tree(
    tree: ET.ElementTree,
    *,
    person_names: list[str],
    subjects: list[str],
    title: str = "",
    title_source: str = "",
    description: str,
    album_title: str,
    gps_latitude: str,
    gps_longitude: str,
    location_address: str = "",
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
    replace_dc_date: bool = False,
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
    values = _merged_xmp_values(
        desc,
        person_names=person_names,
        subjects=subjects,
        title=title,
        title_source=title_source,
        description=description,
        album_title=album_title,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_address=location_address,
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
        description_role=description_role,
        detections_payload=detections_payload,
        ocr_authority_source=ocr_authority_source,
        create_date=create_date,
        dc_date=dc_date,
        date_time_original=date_time_original,
        replace_dc_date=replace_dc_date,
        stitch_key=stitch_key,
        people_detected=people_detected,
        people_identified=people_identified,
        locations_shown=locations_shown,
    )
    merged_subjects = list(values["merged_subjects"])
    person_names = list(values["person_names"])
    title = str(values["title"])
    title_source = str(values["title_source"])
    description = str(values["description"])
    normalized_dc_dates = list(values["normalized_dc_dates"])
    create_date = str(values["create_date"])
    date_time_original = str(values["date_time_original"])
    album_title = str(values["album_title"])
    gps_latitude = str(values["gps_latitude"])
    gps_longitude = str(values["gps_longitude"])
    location_city = str(values["location_city"])
    location_state = str(values["location_state"])
    location_country = str(values["location_country"])
    location_sublocation = str(values["location_sublocation"])
    location_created = str(values["location_created"])
    source_text = str(values["source_text"])
    ocr_text = str(values["ocr_text"])
    parent_ocr_text = str(values["parent_ocr_text"])
    ocr_lang = str(values["ocr_lang"])
    author_text = str(values["author_text"])
    scene_text = str(values["scene_text"])
    ocr_authority_source = str(values["ocr_authority_source"])
    merged_locations_shown = list(values["merged_locations_shown"])
    merged_detections_payload = values["merged_detections_payload"]

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
    _set_role_location_fields(
        desc,
        description_role=description_role,
        location_city=location_city,
        location_state=location_state,
        location_country=location_country,
    )
    _set_simple_text(desc, f"{{{IPTC_EXT_NS}}}Sublocation", str(location_sublocation or "").strip())
    _set_simple_text(
        desc,
        f"{{{IPTC_EXT_NS}}}LocationCreated",
        location_created,
    )
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}PageNumber", str(page_number) if page_number > 0 else "")
    _remove_field(desc, f"{{{IMAGO_NS}}}ScanNumber")
    _set_simple_text(desc, f"{{{DC_NS}}}source", str(source_text or "").strip())
    _set_role_ocr_fields(
        desc,
        description_role=description_role,
        ocr_text=ocr_text,
        parent_ocr_text=parent_ocr_text,
    )
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRLang", str(ocr_lang or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}AuthorText", str(author_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}SceneText", str(scene_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}TitleSource", str(title_source or "").strip())
    _set_simple_text(
        desc,
        f"{{{IMAGO_NS}}}OCRAuthoritySource",
        str(ocr_authority_source or "").strip(),
    )
    _set_detections_fields(desc, merged_detections_payload, image_width=image_width, image_height=image_height)
    _set_locations_shown_bag(desc, merged_locations_shown)
    _set_processing_history(desc, [])
    for legacy_tag in (
        f"{{{IMAGO_NS}}}StitchKey",
        f"{{{IMAGO_NS}}}OcrRan",
        f"{{{IMAGO_NS}}}PeopleDetected",
        f"{{{IMAGO_NS}}}PeopleIdentified",
        f"{{{IMAGO_NS}}}SubPhotos",
        f"{{{XMPDM_NS}}}album",
        f"{{{XMP_NS}}}CreatorTool",
    ):
        _remove_field(desc, legacy_tag)
    ET.indent(tree, space="  ")
    return tree


def _set_role_location_fields(
    desc: ET.Element,
    *,
    description_role: str,
    location_city: str,
    location_state: str,
    location_country: str,
) -> None:
    if description_role == DESCRIPTION_ROLE_CROP:
        for tag in (f"{{{PHOTOSHOP_NS}}}City", f"{{{PHOTOSHOP_NS}}}State", f"{{{PHOTOSHOP_NS}}}Country"):
            _remove_field(desc, tag)
        return
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}City", str(location_city or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}State", str(location_state or "").strip())
    _set_simple_text(desc, f"{{{PHOTOSHOP_NS}}}Country", str(location_country or "").strip())


def _set_role_ocr_fields(
    desc: ET.Element,
    *,
    description_role: str,
    ocr_text: str,
    parent_ocr_text: str,
) -> None:
    if description_role == DESCRIPTION_ROLE_CROP:
        _remove_field(desc, f"{{{IMAGO_NS}}}OCRText")
        _remove_field(desc, f"{{{IMAGO_NS}}}ParentOCRText")
        return
    _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRText", str(ocr_text or "").strip())
    _set_simple_text(desc, f"{{{IMAGO_NS}}}ParentOCRText", str(parent_ocr_text or "").strip())


def _set_detections_fields(
    desc: ET.Element,
    merged_detections_payload: object,
    *,
    image_width: int,
    image_height: int,
) -> None:
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


def write_xmp_sidecar(
    sidecar_path: str | Path,
    *,
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
    location_address: str = "",
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
    replace_dc_date: bool = False,
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
            person_names=person_names,
            subjects=subjects,
            description_role=description_role,
            title=title,
            title_source=title_source,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_address=location_address,
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
            person_names=person_names,
            subjects=subjects,
            description_role=description_role,
            title=title,
            title_source=title_source,
            description=description,
            album_title=album_title,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location_address=location_address,
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
            replace_dc_date=replace_dc_date,
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


def _archive_copy_safe_updates(view_xmp: Path, archive_xmp: Path) -> dict[str, object] | None:
    archive_state = read_ai_sidecar_state(archive_xmp)
    if not isinstance(archive_state, dict):
        return None

    view_state = read_ai_sidecar_state(view_xmp) or {}
    archive_locations = read_locations_shown(archive_xmp)
    view_locations = read_locations_shown(view_xmp)
    locations_shown = archive_locations if (archive_locations and not view_locations) else None
    description = _copy_if_empty(archive_state, view_state, "description")
    ocr_text = _copy_if_empty(archive_state, view_state, "ocr_text")
    merged_location = _archive_copy_location_values(view_state, archive_state)
    if not _has_archive_copy_updates(
        locations_shown=locations_shown,
        description=description,
        ocr_text=ocr_text,
        merged_location=merged_location,
    ):
        return None
    return {
        "view_state": view_state,
        "archive_state": archive_state,
        "locations_shown": locations_shown,
        "description": description,
        "ocr_text": ocr_text,
        **merged_location,
    }


def _archive_copy_location_values(view_state: dict, archive_state: dict) -> dict[str, str]:
    return {
        key: str(view_state.get(key) or archive_state.get(key) or "")
        for key in (
            "gps_latitude",
            "gps_longitude",
            "location_city",
            "location_state",
            "location_country",
            "location_sublocation",
        )
    }


def _has_archive_copy_updates(
    *,
    locations_shown: list[dict[str, str]] | None,
    description: str,
    ocr_text: str,
    merged_location: dict[str, str],
) -> bool:
    has_gps = bool(merged_location["gps_latitude"] and merged_location["gps_longitude"])
    has_location = bool(merged_location["location_city"] or merged_location["location_country"])
    return bool(locations_shown or description or ocr_text or has_gps or has_location)


def _copy_if_empty(archive_state: dict, view_state: dict, key: str) -> str:
    archive_value = str(archive_state.get(key) or "").strip()
    view_value = str(view_state.get(key) or "").strip()
    return archive_value if (archive_value and not view_value) else ""


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

    updates = _archive_copy_safe_updates(view_xmp, archive_xmp)
    if updates is None:
        return False
    view_state = dict(updates["view_state"])
    archive_state = dict(updates["archive_state"])

    existing_people = read_person_in_image(view_xmp)

    write_xmp_sidecar(
        view_xmp,
        person_names=existing_people,
        subjects=[],
        description=str(updates["description"]),
        ocr_text=str(updates["ocr_text"]),
        ocr_lang=str(view_state.get("ocr_lang") or archive_state.get("ocr_lang") or ""),
        author_text=str(view_state.get("author_text") or archive_state.get("author_text") or ""),
        scene_text=str(view_state.get("scene_text") or archive_state.get("scene_text") or ""),
        album_title=str(view_state.get("album_title") or archive_state.get("album_title") or ""),
        gps_latitude=str(updates["gps_latitude"]),
        gps_longitude=str(updates["gps_longitude"]),
        location_city=str(updates["location_city"]),
        location_state=str(updates["location_state"]),
        location_country=str(updates["location_country"]),
        location_sublocation=str(updates["location_sublocation"]),
        source_text=str(view_state.get("source_text") or archive_state.get("source_text") or ""),
        dc_date=str(view_state.get("dc_date") or archive_state.get("dc_date") or ""),
        date_time_original=str(view_state.get("date_time_original") or archive_state.get("date_time_original") or ""),
        locations_shown=updates["locations_shown"],  # type: ignore[arg-type]
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

    existing_regions = _read_existing_region_list(path, img_w=img_w, img_h=img_h)
    tree = _read_or_create_xmp_tree(path)

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

    # Look up the AI metadata's verbatim per-photo response, if any, so we
    # can fill in mwg-rs:Name even when the caller (e.g. detect-regions)
    # passes regions with empty captions. The detections payload is the
    # source of truth — see _do_metadata in ai_index_analysis.
    detections_payload = _read_detections_payload(desc)
    metadata_photos_by_number = _metadata_photos_by_number(detections_payload)

    # mwg-rs:RegionList
    region_list_el = ET.SubElement(region_info, f"{{{MWGRS_NS}}}RegionList")
    bag = ET.SubElement(region_list_el, _RDF_BAG)

    for rwc in regions_with_captions:
        _add_mwgrs_region_item(
            bag,
            rwc,
            img_w=img_w,
            img_h=img_h,
            existing_regions=existing_regions,
            metadata_photos_by_number=metadata_photos_by_number,
            pixel_to_mwgrs=pixel_to_mwgrs,
        )

    # Update modify date
    desc.set(f"{{{XMP_NS}}}ModifyDate", _xmp_datetime_now())

    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def _read_existing_region_list(path: Path, *, img_w: int, img_h: int) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return read_region_list(path, img_w, img_h)
    except Exception:
        return []


def _read_or_create_xmp_tree(path: Path) -> ET.ElementTree:
    if path.is_file():
        try:
            return ET.parse(str(path))  # type: ignore[return-value]
        except ET.ParseError:
            pass
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    xmpmeta.set(f"{{{X_NS}}}xmptk", "imago")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    ET.SubElement(rdf, _RDF_DESC).set(f"{{{RDF_NS}}}about", "")
    return ET.ElementTree(xmpmeta)


def _metadata_photos_by_number(detections_payload: dict) -> dict[int, dict]:
    metadata_photos_by_number: dict[int, dict] = {}
    caption_block = detections_payload.get("caption")
    if not isinstance(caption_block, dict):
        return metadata_photos_by_number
    for entry in list(caption_block.get("photos") or []):
        if not isinstance(entry, dict):
            continue
        try:
            photo_number = int(entry.get("photo_number") or 0)
        except (TypeError, ValueError):
            photo_number = 0
        if photo_number > 0:
            metadata_photos_by_number[photo_number] = entry
    return metadata_photos_by_number


def _add_mwgrs_region_item(
    bag: ET.Element,
    rwc,
    *,
    img_w: int,
    img_h: int,
    existing_regions: list[dict],
    metadata_photos_by_number: dict[int, dict],
    pixel_to_mwgrs,
) -> None:
    r = rwc.region
    cx, cy, nw, nh = pixel_to_mwgrs(r.x, r.y, r.width, r.height, img_w, img_h)
    existing_region = existing_regions[r.index] if 0 <= int(r.index) < len(existing_regions) else {}
    photo_number = int(getattr(r, "photo_number", 0) or 0) or int(existing_region.get("photo_number") or 0)
    region_caption = _region_caption_text(
        rwc, photo_number=photo_number, metadata_photos_by_number=metadata_photos_by_number
    )

    li = ET.SubElement(bag, _RDF_LI)
    li.set(_RDF_PARSE_TYPE, "Resource")
    li.set(f"{{{MWGRS_NS}}}Type", "Photo")
    li.set(f"{{{MWGRS_NS}}}Name", region_caption)
    _set_mwgrs_coordinates(li, cx=cx, cy=cy, nw=nw, nh=nh)
    _set_region_optional_attrs(li, r, photo_number=photo_number)
    _add_region_person_names(li, getattr(r, "person_names", ()) or ())
    _add_region_location_struct(li, f"{{{IMAGO_NS}}}LocationAssigned", dict(getattr(r, "location_payload", {}) or {}))
    location_override = dict(getattr(r, "location_override", {}) or {}) or dict(
        existing_region.get("location_override") or {}
    )
    _add_region_location_struct(li, f"{{{IMAGO_NS}}}LocationOverride", location_override)


def _region_caption_text(rwc, *, photo_number: int, metadata_photos_by_number: dict[int, dict]) -> str:
    region_caption = str(rwc.caption or "").strip()
    if region_caption or photo_number <= 0:
        return region_caption
    ai_entry = metadata_photos_by_number.get(photo_number) or {}
    return str(ai_entry.get("corrected_caption") or ai_entry.get("caption") or "").strip()


def _set_mwgrs_coordinates(li: ET.Element, *, cx: float, cy: float, nw: float, nh: float) -> None:
    li.set(f"{{{STAREA_NS}}}x", f"{cx:.6f}")
    li.set(f"{{{STAREA_NS}}}y", f"{cy:.6f}")
    li.set(f"{{{STAREA_NS}}}w", f"{nw:.6f}")
    li.set(f"{{{STAREA_NS}}}h", f"{nh:.6f}")
    li.set(f"{{{STAREA_NS}}}unit", "normalized")


def _set_region_optional_attrs(li: ET.Element, region, *, photo_number: int) -> None:
    caption_hint = str(getattr(region, "caption_hint", "") or "").strip()
    if caption_hint:
        li.set(f"{{{IMAGO_NS}}}CaptionHint", caption_hint)
    if photo_number > 0:
        li.set(f"{{{IMAGO_NS}}}PhotoNumber", str(photo_number))


def _add_region_person_names(parent: ET.Element, names) -> None:
    person_names = [str(name).strip() for name in list(names or ()) if str(name).strip()]
    if not person_names:
        return
    pn_el = ET.SubElement(parent, f"{{{IMAGO_NS}}}PersonNames")
    pn_bag = ET.SubElement(pn_el, _RDF_BAG)
    for name in person_names:
        item = ET.SubElement(pn_bag, _RDF_LI)
        item.text = name


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
        photo_number = int(li.get(f"{{{IMAGO_NS}}}PhotoNumber") or 0)

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
                "photo_number": photo_number,
                "location_payload": _read_region_location_struct(li, f"{{{IMAGO_NS}}}LocationAssigned"),
                "location_override": _read_region_location_struct(li, f"{{{IMAGO_NS}}}LocationOverride"),
                "person_names": person_names,
                "type": rtype,
            }
        )
        idx += 1
    return results
