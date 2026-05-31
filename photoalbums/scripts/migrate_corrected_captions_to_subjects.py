"""Move corrected album captions from text fields into dc:subject keywords."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ..lib.xmp_sidecar import DC_NS, IMAGO_NS, RDF_NS, _dedupe

ET.register_namespace("dc", DC_NS)
ET.register_namespace("imago", IMAGO_NS)
ET.register_namespace("rdf", RDF_NS)


def _rdf_description(root: ET.Element) -> ET.Element | None:
    return root.find(f".//{{{RDF_NS}}}Description")


def _simple_text(desc: ET.Element, tag: str) -> str:
    return str(desc.findtext(tag, default="") or "").strip()


def _set_simple_text(desc: ET.Element, tag: str, text: str) -> bool:
    existing = desc.find(tag)
    clean = str(text or "").strip()
    if not clean:
        return False
    if existing is None:
        existing = ET.SubElement(desc, tag)
    if str(existing.text or "") == clean:
        return False
    existing.text = clean
    return True


def _bag_values(desc: ET.Element, tag: str) -> list[str]:
    bag = desc.find(f"{tag}/{{{RDF_NS}}}Bag")
    if bag is None:
        return []
    return [str(li.text or "").strip() for li in bag.findall(f"{{{RDF_NS}}}li") if str(li.text or "").strip()]


def _set_bag(desc: ET.Element, tag: str, values: list[str]) -> bool:
    existing = _bag_values(desc, tag)
    deduped = _dedupe(values)
    if existing == deduped:
        return False
    subject = desc.find(tag)
    if subject is None:
        subject = ET.SubElement(desc, tag)
    for child in list(subject):
        subject.remove(child)
    bag = ET.SubElement(subject, f"{{{RDF_NS}}}Bag")
    for value in deduped:
        li = ET.SubElement(bag, f"{{{RDF_NS}}}li")
        li.text = value
    return True


def _set_default_description(desc: ET.Element, text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    description = desc.find(f"{{{DC_NS}}}description")
    if description is None:
        description = ET.SubElement(desc, f"{{{DC_NS}}}description")
    alt = description.find(f"{{{RDF_NS}}}Alt")
    if alt is None:
        for child in list(description):
            description.remove(child)
        alt = ET.SubElement(description, f"{{{RDF_NS}}}Alt")
    li = alt.find(f"{{{RDF_NS}}}li[@{{http://www.w3.org/XML/1998/namespace}}lang='x-default']")
    if li is None:
        li = ET.SubElement(alt, f"{{{RDF_NS}}}li")
        li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    if str(li.text or "") == clean:
        return False
    li.text = clean
    return True


def _caption_texts(payload: dict) -> tuple[list[str], list[str], bool]:
    caption_block = payload.get("caption")
    if not isinstance(caption_block, dict):
        return [], [], False
    captions: list[str] = []
    corrections: list[str] = []
    changed = False
    for photo in list(caption_block.get("photos") or []):
        if not isinstance(photo, dict):
            continue
        corrected = str(photo.get("corrected_caption") or "").strip()
        original = str(photo.get("OriginalCaption") or photo.get("OriginalCapation") or photo.get("caption") or "").strip()
        caption = str(photo.get("caption") or "").strip()
        if corrected and original and corrected.casefold() != original.casefold():
            corrections.append(corrected)
            if caption != original:
                photo["caption"] = original
                changed = True
            if photo.get("OriginalCaption") != original:
                photo["OriginalCaption"] = original
                changed = True
            if "OriginalCapation" in photo:
                del photo["OriginalCapation"]
                changed = True
        if original and original.casefold() not in {c.casefold() for c in captions}:
            captions.append(original)
    return captions, corrections, changed


def migrate_sidecar(path: Path, *, dry_run: bool = False) -> bool:
    tree = ET.parse(path)
    desc = _rdf_description(tree.getroot())
    if desc is None:
        return False
    detections_el = desc.find(f"{{{IMAGO_NS}}}Detections")
    if detections_el is None or not str(detections_el.text or "").strip():
        return False
    payload = json.loads(str(detections_el.text or ""))
    if not isinstance(payload, dict):
        return False
    captions, corrections, detections_changed = _caption_texts(payload)
    if not corrections:
        return False
    author_text = " ".join(captions)
    scene_texts = [
        str(photo.get("scene_ocr") or "").strip()
        for photo in list((payload.get("caption") or {}).get("photos") or [])
        if isinstance(photo, dict) and str(photo.get("scene_ocr") or "").strip()
    ]
    scene_text = "\n".join(_dedupe(scene_texts)) or _simple_text(desc, f"{{{IMAGO_NS}}}SceneText")
    ocr_text = "\n".join(text for text in (author_text, scene_text) if text)

    changed = detections_changed
    changed |= _set_bag(desc, f"{{{DC_NS}}}subject", _bag_values(desc, f"{{{DC_NS}}}subject") + corrections)
    changed |= _set_simple_text(desc, f"{{{IMAGO_NS}}}AuthorText", author_text)
    changed |= _set_simple_text(desc, f"{{{IMAGO_NS}}}OCRText", ocr_text)
    changed |= _set_default_description(desc, author_text)
    if detections_changed:
        detections_el.text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if changed and not dry_run:
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Photo album root containing XMP sidecars.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    changed = 0
    scanned = 0
    for path in root.rglob("*.xmp"):
        scanned += 1
        try:
            if migrate_sidecar(path, dry_run=bool(args.dry_run)):
                changed += 1
                print(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to migrate {path}: {exc}") from exc
    action = "would update" if args.dry_run else "updated"
    print(f"{action} {changed} of {scanned} XMP sidecar(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
