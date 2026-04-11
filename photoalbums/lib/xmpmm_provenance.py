"""xmpMM provenance and pipeline state helpers.

Provides functions for writing DocumentID, DerivedFrom, Pantry, and tracking
pipeline step completion in imago:Detections on XMP sidecars.
"""
from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# Namespace constants (duplicated here to avoid circular import with xmp_sidecar)
_X_NS = "adobe:ns:meta/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XMPMM_NS = "http://ns.adobe.com/xap/1.0/mm/"
_IMAGO_NS = "https://imago.local/ns/1.0/"
_ST_REF_NS = "http://ns.adobe.com/xap/1.0/sType/ResourceRef#"

_RDF_ROOT = f"{{{_RDF_NS}}}RDF"
_RDF_DESC = f"{{{_RDF_NS}}}Description"
_RDF_BAG = f"{{{_RDF_NS}}}Bag"
_RDF_LI = f"{{{_RDF_NS}}}li"
_RDF_PARSE_TYPE = f"{{{_RDF_NS}}}parseType"

# Register namespaces so ElementTree serialises them with clean prefixes
for _prefix, _uri in [
    ("x", _X_NS),
    ("rdf", _RDF_NS),
    ("xmpMM", _XMPMM_NS),
    ("imago", _IMAGO_NS),
    ("stRef", _ST_REF_NS),
]:
    ET.register_namespace(_prefix, _uri)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_tree(path: Path) -> ET.ElementTree:
    """Load an XMP tree, creating a minimal skeleton if the file is missing or invalid."""
    if path.is_file():
        try:
            return ET.parse(str(path))  # type: ignore[return-value]
        except ET.ParseError:
            pass
    xmpmeta = ET.Element(f"{{{_X_NS}}}xmpmeta")
    xmpmeta.set(f"{{{_X_NS}}}xmptk", "imago")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    ET.SubElement(rdf, _RDF_DESC).set(f"{{{_RDF_NS}}}about", "")
    return ET.ElementTree(xmpmeta)


def _get_or_create_desc(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    assert root is not None
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        rdf = ET.SubElement(root, _RDF_ROOT)  # type: ignore[arg-type]
    desc = rdf.find(_RDF_DESC)
    if desc is None:
        desc = ET.SubElement(rdf, _RDF_DESC)
        desc.set(f"{{{_RDF_NS}}}about", "")
    elif f"{{{_RDF_NS}}}about" not in desc.attrib:
        desc.set(f"{{{_RDF_NS}}}about", "")
    return desc


def _save_tree(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Document ID
# ---------------------------------------------------------------------------

def read_document_id(xmp_path: str | Path) -> str:
    """Return existing xmpMM:DocumentID or '' if absent."""
    path = Path(xmp_path)
    if not path.is_file():
        return ""
    try:
        tree = ET.parse(str(path))
    except ET.ParseError:
        return ""
    root = tree.getroot()
    if root is None:
        return ""
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        return ""
    desc = rdf.find(_RDF_DESC)
    if desc is None:
        return ""
    return str(desc.get(f"{{{_XMPMM_NS}}}DocumentID") or desc.findtext(f"{{{_XMPMM_NS}}}DocumentID") or "").strip()


def assign_document_id(xmp_path: str | Path) -> str:
    """Assign xmpMM:DocumentID to the sidecar if not already present.

    Returns the DocumentID (whether newly assigned or pre-existing).
    The ID is written as a simple attribute on the rdf:Description element.
    """
    path = Path(xmp_path)
    tree = _load_tree(path)
    desc = _get_or_create_desc(tree)

    existing = str(desc.get(f"{{{_XMPMM_NS}}}DocumentID") or "").strip()
    if not existing:
        # Also check child element form
        child = desc.find(f"{{{_XMPMM_NS}}}DocumentID")
        if child is not None:
            existing = str(child.text or "").strip()
    if existing:
        return existing

    doc_id = f"xmp:uuid:{uuid.uuid4()}"
    desc.set(f"{{{_XMPMM_NS}}}DocumentID", doc_id)
    _save_tree(tree, path)
    return doc_id


# ---------------------------------------------------------------------------
# DerivedFrom
# ---------------------------------------------------------------------------

def write_derived_from(xmp_path: str | Path, source_document_id: str, source_path: str = "") -> None:
    """Write xmpMM:DerivedFrom referencing the given source documentID.

    Replaces any existing DerivedFrom block. ``source_path`` is written as
    stRef:filePath if provided.
    """
    path = Path(xmp_path)
    tree = _load_tree(path)
    desc = _get_or_create_desc(tree)

    tag = f"{{{_XMPMM_NS}}}DerivedFrom"
    existing = desc.find(tag)
    if existing is not None:
        desc.remove(existing)

    df = ET.SubElement(desc, tag)
    df.set(_RDF_PARSE_TYPE, "Resource")
    ET.SubElement(df, f"{{{_ST_REF_NS}}}documentID").text = source_document_id
    if source_path:
        ET.SubElement(df, f"{{{_ST_REF_NS}}}filePath").text = source_path

    _save_tree(tree, path)


# ---------------------------------------------------------------------------
# Pantry
# ---------------------------------------------------------------------------

def write_pantry_entry(xmp_path: str | Path, source_document_id: str, source_path: str = "") -> None:
    """Add a xmpMM:Pantry entry for the given source if not already present.

    Pantry is an rdf:Bag of rdf:Description items, each with xmpMM:DocumentID.
    Duplicate entries (by documentID) are skipped.
    """
    path = Path(xmp_path)
    tree = _load_tree(path)
    desc = _get_or_create_desc(tree)

    pantry_tag = f"{{{_XMPMM_NS}}}Pantry"
    pantry = desc.find(pantry_tag)
    if pantry is None:
        pantry = ET.SubElement(desc, pantry_tag)
        bag = ET.SubElement(pantry, _RDF_BAG)
    else:
        bag = pantry.find(_RDF_BAG)
        if bag is None:
            bag = ET.SubElement(pantry, _RDF_BAG)

    # Check for existing entry with the same documentID
    for li in bag.findall(_RDF_LI):
        existing_id = str(
            li.get(f"{{{_XMPMM_NS}}}DocumentID") or li.findtext(f"{{{_XMPMM_NS}}}DocumentID") or ""
        ).strip()
        if existing_id == source_document_id:
            return  # Already present

    li = ET.SubElement(bag, _RDF_LI)
    li.set(_RDF_PARSE_TYPE, "Resource")
    ET.SubElement(li, f"{{{_XMPMM_NS}}}DocumentID").text = source_document_id
    if source_path:
        ET.SubElement(li, f"{{{_ST_REF_NS}}}filePath").text = source_path

    _save_tree(tree, path)


# ---------------------------------------------------------------------------
# Pipeline state (imago:Detections -> pipeline key)
# ---------------------------------------------------------------------------

def _read_detections(desc: ET.Element) -> dict:
    text = str(desc.findtext(f"{{{_IMAGO_NS}}}Detections", default="") or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _write_detections(desc: ET.Element, tree: ET.ElementTree, path: Path, detections: dict) -> None:
    tag = f"{{{_IMAGO_NS}}}Detections"
    existing = desc.find(tag)
    if existing is not None:
        desc.remove(existing)
    if detections:
        el = ET.SubElement(desc, tag)
        el.text = json.dumps(detections, ensure_ascii=False, sort_keys=True)
    _save_tree(tree, path)


def write_pipeline_step(xmp_path: str | Path, step_name: str, *, extra: dict | None = None) -> None:
    """Record a pipeline step completion in imago:Detections -> pipeline -> step_name.

    Writes ``{"completed": "<iso-timestamp>"}`` plus any ``extra`` fields.
    """
    path = Path(xmp_path)
    tree = _load_tree(path)
    desc = _get_or_create_desc(tree)

    detections = _read_detections(desc)
    pipeline = dict(detections.get("pipeline") or {})
    entry: dict = {"completed": _iso_now()}
    if extra:
        entry.update(extra)
    pipeline[step_name] = entry
    detections["pipeline"] = pipeline
    _write_detections(desc, tree, path, detections)


def clear_pipeline_steps(xmp_path: str | Path, step_names: list[str]) -> None:
    """Remove named pipeline step records from imago:Detections -> pipeline."""
    path = Path(xmp_path)
    if not path.is_file():
        return
    tree = _load_tree(path)
    desc = _get_or_create_desc(tree)

    detections = _read_detections(desc)
    pipeline = dict(detections.get("pipeline") or {})
    changed = False
    for name in step_names:
        if name in pipeline:
            del pipeline[name]
            changed = True
    if not changed:
        return
    if pipeline:
        detections["pipeline"] = pipeline
    else:
        detections.pop("pipeline", None)
    _write_detections(desc, tree, path, detections)


def read_pipeline_step(xmp_path: str | Path, step_name: str) -> dict | None:
    """Return the pipeline step record dict, or None if absent."""
    path = Path(xmp_path)
    if not path.is_file():
        return None
    try:
        tree = ET.parse(str(path))
    except ET.ParseError:
        return None
    root = tree.getroot()
    if root is None:
        return None
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        return None
    desc = rdf.find(_RDF_DESC)
    if desc is None:
        return None
    detections = _read_detections(desc)
    pipeline = detections.get("pipeline")
    if not isinstance(pipeline, dict):
        return None
    step = pipeline.get(step_name)
    return step if isinstance(step, dict) else None
