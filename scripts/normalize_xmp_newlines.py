from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
IMAGO_NS = "https://imago.local/ns/1.0/"
EXIF_NS = "http://ns.adobe.com/exif/1.0/"
PHOTOSHOP_NS = "http://ns.adobe.com/photoshop/1.0/"
XMPDM_NS = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"

MULTILINE_TAGS = {
    f"{{{DC_NS}}}description",
    f"{{{DC_NS}}}title",
    f"{{{IMAGO_NS}}}OCRText",
    f"{{{IMAGO_NS}}}AuthorText",
    f"{{{IMAGO_NS}}}SceneText",
}

TEXT_TAGS = {
    f"{{{DC_NS}}}source",
    f"{{{DC_NS}}}description",
    f"{{{DC_NS}}}title",
    f"{{{XMP_NS}}}CreateDate",
    f"{{{IPTC_EXT_NS}}}PersonInImage",
    f"{{{IPTC_EXT_NS}}}LocationShown",
    f"{{{IMAGO_NS}}}AlbumTitle",
    f"{{{IMAGO_NS}}}OCRText",
    f"{{{IMAGO_NS}}}OCRLang",
    f"{{{IMAGO_NS}}}AuthorText",
    f"{{{IMAGO_NS}}}SceneText",
    f"{{{IMAGO_NS}}}TitleSource",
    f"{{{IMAGO_NS}}}OCRAuthoritySource",
    f"{{{IMAGO_NS}}}Detections",
    f"{{{EXIF_NS}}}DateTimeOriginal",
    f"{{{EXIF_NS}}}GPSLatitude",
    f"{{{EXIF_NS}}}GPSLongitude",
    f"{{{EXIF_NS}}}GPSMapDatum",
    f"{{{EXIF_NS}}}GPSVersionID",
    f"{{{PHOTOSHOP_NS}}}City",
    f"{{{PHOTOSHOP_NS}}}State",
    f"{{{PHOTOSHOP_NS}}}Country",
    f"{{{PHOTOSHOP_NS}}}PageNumber",
    f"{{{XMPDM_NS}}}album",
}


def _normalize_text(text: str, *, multiline: bool) -> str:
    clean = str(text or "")
    if not clean or r"\n" not in clean:
        return clean
    return clean.replace(r"\n", "\n" if multiline else " ")


def _walk_and_normalize(node: ET.Element) -> bool:
    changed = False
    if node.tag in TEXT_TAGS and node.text is not None:
        normalized = _normalize_text(node.text, multiline=node.tag in MULTILINE_TAGS)
        if normalized != node.text:
            node.text = normalized
            changed = True
    for child in list(node):
        if _walk_and_normalize(child):
            changed = True
    return changed


def normalize_xmp_file(path: Path) -> bool:
    tree = ET.parse(path)
    root = tree.getroot()
    changed = _walk_and_normalize(root)
    if changed:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize literal \\n sequences in XMP sidecars.")
    parser.add_argument(
        "root",
        nargs="?",
        default=r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums",
        help="Photo Albums root directory",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    changed = 0
    scanned = 0
    for path in root.rglob("*.xmp"):
        scanned += 1
        if normalize_xmp_file(path):
            changed += 1
            print(f"updated {path}")
    print(f"scanned={scanned} updated={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
