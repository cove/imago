from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
IMAGO_NS = "https://imago.local/ns/1.0/"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("imago", IMAGO_NS)


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
    bag = ET.SubElement(field, f"{{{RDF_NS}}}Bag")
    for value in values:
        item = ET.SubElement(bag, f"{{{RDF_NS}}}li")
        item.text = value


def _add_alt_text(parent: ET.Element, tag: str, value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    field = ET.SubElement(parent, tag)
    alt = ET.SubElement(field, f"{{{RDF_NS}}}Alt")
    item = ET.SubElement(alt, f"{{{RDF_NS}}}li")
    item.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    item.text = text


def _add_simple_text(parent: ET.Element, tag: str, value: str | int | float) -> None:
    text = str(value)
    field = ET.SubElement(parent, tag)
    field.text = text


def _add_subphotos(parent: ET.Element, subphotos: list[dict]) -> None:
    if not subphotos:
        return
    field = ET.SubElement(parent, f"{{{IMAGO_NS}}}SubPhotos")
    seq = ET.SubElement(field, f"{{{RDF_NS}}}Seq")
    for row in subphotos:
        item = ET.SubElement(seq, f"{{{RDF_NS}}}li")
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
    ocr_text: str,
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
) -> ET.ElementTree:
    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{RDF_NS}}}RDF")
    desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
    desc.set(f"{{{RDF_NS}}}about", "")

    _add_bag(desc, f"{{{DC_NS}}}subject", _dedupe(subjects))
    _add_bag(desc, f"{{{IPTC_EXT_NS}}}PersonInImage", _dedupe(person_names))
    _add_alt_text(desc, f"{{{DC_NS}}}description", description)

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


def write_xmp_sidecar(
    sidecar_path: str | Path,
    *,
    creator_tool: str,
    person_names: list[str],
    subjects: list[str],
    description: str,
    ocr_text: str,
    detections_payload: dict | None = None,
    subphotos: list[dict] | None = None,
) -> Path:
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_xmp_tree(
        creator_tool=creator_tool,
        person_names=person_names,
        subjects=subjects,
        description=description,
        ocr_text=ocr_text,
        detections_payload=detections_payload,
        subphotos=subphotos,
    )
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
