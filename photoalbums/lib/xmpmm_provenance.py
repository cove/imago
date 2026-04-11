"""xmpMM provenance and pipeline state helpers.

Provides functions for writing DocumentID, DerivedFrom, Pantry, and tracking
pipeline step completion in imago:Detections on XMP sidecars.
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
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

# Re-use the shared helper from xmp_sidecar to avoid duplication
from .xmp_sidecar import (  # noqa: E402
    _get_or_create_rdf_desc as _get_or_create_desc,
    clear_pipeline_steps,
    read_pipeline_state,
    read_pipeline_step,
    write_pipeline_step,
)


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


def write_creation_provenance(
    xmp_path: str | Path,
    *,
    derived_from: dict,
    pantry_sources: list[dict],
) -> None:
    """Write DerivedFrom and Pantry to a sidecar while preserving all other fields.

    ``derived_from`` should be a dict with keys ``source_document_id`` and
    optionally ``source_path``.
    ``pantry_sources`` is a list of such dicts; duplicates are skipped.
    """
    src_id = str(derived_from.get("source_document_id") or "").strip()
    src_path = str(derived_from.get("source_path") or "").strip()
    if src_id:
        write_derived_from(xmp_path, src_id, src_path)
    for entry in pantry_sources:
        entry_id = str(entry.get("source_document_id") or "").strip()
        entry_path = str(entry.get("source_path") or "").strip()
        if entry_id:
            write_pantry_entry(xmp_path, entry_id, entry_path)


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
