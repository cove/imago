from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

# Namespaces copied from photoalbums/lib/xmp_sidecar.py (stdlib only, no cross-project import)
X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
DC_NS = "http://purl.org/dc/elements/1.1/"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_EXT_NS)
ET.register_namespace("dc", DC_NS)

_PERSON_TAG = f"{{{IPTC_EXT_NS}}}PersonInImage"
_DC_DESC_TAG = f"{{{DC_NS}}}description"
_RDF_BAG = f"{{{RDF_NS}}}Bag"
_RDF_LI = f"{{{RDF_NS}}}li"
_RDF_DESC = f"{{{RDF_NS}}}Description"
_RDF_ROOT = f"{{{RDF_NS}}}RDF"
_XMP_CREATOR = f"{{{XMP_NS}}}CreatorTool"


def _dedupe(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def read_person_in_image(sidecar_path: Path) -> list[str]:
    """
    Parse an existing .xmp sidecar and return the Iptc4xmpExt:PersonInImage
    bag as a list of name strings. Returns [] if the file doesn't exist,
    is malformed, or has no people.
    """
    try:
        sidecar_path = Path(sidecar_path)
        if not sidecar_path.exists():
            return []
        tree = ET.parse(str(sidecar_path))
        root = tree.getroot()
        names: list[str] = []
        for elem in root.iter(_PERSON_TAG):
            bag = elem.find(_RDF_BAG)
            if bag is None:
                continue
            for li in bag.findall(_RDF_LI):
                text = (li.text or "").strip()
                if text:
                    names.append(text)
        return _dedupe(names)
    except Exception:
        return []


def merge_persons_xmp(
    sidecar_path: Path,
    person_names: list[str],
    *,
    creator_tool: str = "cast-label-photos",
) -> Path:
    """
    Write PersonInImage names to a .xmp sidecar.

    - If the sidecar already exists, only the Iptc4xmpExt:PersonInImage bag is
      updated; all other fields (dc:description, dc:subject, imago:*, etc.) are
      preserved untouched.
    - If the sidecar does not exist, a minimal XMP file is created containing
      only PersonInImage and xmp:CreatorTool.

    Returns the path that was written.
    """
    sidecar_path = Path(sidecar_path)
    names = _dedupe(person_names)

    if sidecar_path.exists():
        try:
            tree = ET.parse(str(sidecar_path))
        except ET.ParseError:
            tree = None
        if tree is not None:
            _merge_into_tree(tree, names)
            tree.write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)
            return sidecar_path

    # No existing sidecar (or parse failure) — write a minimal file
    _write_minimal(sidecar_path, names, creator_tool=creator_tool)
    return sidecar_path


def _merge_into_tree(tree: ET.ElementTree, names: list[str]) -> None:
    """Update or create the PersonInImage bag inside an existing XMP tree."""
    root = tree.getroot()

    # Find or create the rdf:RDF element
    rdf_rdf = root.find(_RDF_ROOT)
    if rdf_rdf is None:
        rdf_rdf = ET.SubElement(root, _RDF_ROOT)

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


def _write_minimal(
    sidecar_path: Path,
    names: list[str],
    *,
    creator_tool: str,
) -> None:
    """Create a minimal XMP sidecar with only PersonInImage and CreatorTool."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    xmpmeta = ET.Element(f"{{{X_NS}}}xmpmeta")
    xmpmeta.set(f"{{{X_NS}}}xmptk", creator_tool)
    rdf_rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    rdf_rdf.set(f"xmlns:rdf", RDF_NS)
    desc = ET.SubElement(rdf_rdf, _RDF_DESC)
    desc.set(f"{{{RDF_NS}}}about", "")

    creator = ET.SubElement(desc, _XMP_CREATOR)
    creator.text = creator_tool

    person_elem = ET.SubElement(desc, _PERSON_TAG)
    bag = ET.SubElement(person_elem, _RDF_BAG)
    for name in names:
        li = ET.SubElement(bag, _RDF_LI)
        li.text = name

    tree = ET.ElementTree(xmpmeta)
    tree.write(str(sidecar_path), encoding="UTF-8", xml_declaration=True)
