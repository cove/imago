"""Tests for merge_face_regions_xmp in cast.xmp_writer."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cast.xmp_writer import (
    merge_face_regions_xmp,
    merge_persons_xmp,
    read_person_in_image,
)

IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _parse_face_regions(xmp_path: Path) -> list[dict]:
    tree = ET.parse(str(xmp_path))
    root = tree.getroot()
    rdf_rdf = root.find(f"{{{RDF_NS}}}RDF")
    assert rdf_rdf is not None
    desc = rdf_rdf.find(f"{{{RDF_NS}}}Description")
    assert desc is not None
    ir = desc.find(f"{{{IPTC_NS}}}ImageRegion")
    if ir is None:
        return []
    bag = ir.find(f"{{{RDF_NS}}}Bag")
    if bag is None:
        return []
    regions = []
    for li in bag.findall(f"{{{RDF_NS}}}li"):
        boundary = li.find(f"{{{IPTC_NS}}}RegionBoundary")
        name_elem = li.find(f"{{{IPTC_NS}}}Name")
        name = ""
        if name_elem is not None:
            alt = name_elem.find(f"{{{RDF_NS}}}Alt")
            if alt is not None:
                for item in alt.findall(f"{{{RDF_NS}}}li"):
                    name = item.text or ""
                    break
        row: dict = {
            "rctype": li.findtext(f"{{{IPTC_NS}}}RCtype", default=""),
            "rid": li.findtext(f"{{{IPTC_NS}}}rId", default=""),
            "name": name,
        }
        if boundary is not None:
            row["rbShape"] = boundary.findtext(f"{{{IPTC_NS}}}rbShape", default="")
            row["rbUnit"] = boundary.findtext(f"{{{IPTC_NS}}}rbUnit", default="")
            row["rbX"] = boundary.findtext(f"{{{IPTC_NS}}}rbX", default="")
            row["rbY"] = boundary.findtext(f"{{{IPTC_NS}}}rbY", default="")
            row["rbW"] = boundary.findtext(f"{{{IPTC_NS}}}rbW", default="")
            row["rbH"] = boundary.findtext(f"{{{IPTC_NS}}}rbH", default="")
        regions.append(row)
    return regions


def test_create_minimal_sidecar_with_face_regions(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    regions = [{"name": "Alice", "rx": 0.1, "ry": 0.2, "rw": 0.3, "rh": 0.4}]

    result = merge_face_regions_xmp(xmp, regions)

    assert result == xmp
    assert xmp.is_file()
    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Alice"
    assert parsed[0]["rctype"] == "face-identified"
    assert parsed[0]["rid"] == "face-1"
    assert parsed[0]["rbShape"] == "rectangle"
    assert parsed[0]["rbUnit"] == "relative"
    assert abs(float(parsed[0]["rbX"]) - 0.1) < 1e-5
    assert abs(float(parsed[0]["rbY"]) - 0.2) < 1e-5
    assert abs(float(parsed[0]["rbW"]) - 0.3) < 1e-5
    assert abs(float(parsed[0]["rbH"]) - 0.4) < 1e-5


def test_multiple_regions_numbered_sequentially(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    regions = [
        {"name": "Alice", "rx": 0.0, "ry": 0.0, "rw": 0.2, "rh": 0.2},
        {"name": "Bob", "rx": 0.5, "ry": 0.5, "rw": 0.2, "rh": 0.2},
    ]

    merge_face_regions_xmp(xmp, regions)
    parsed = _parse_face_regions(xmp)

    assert len(parsed) == 2
    assert parsed[0]["name"] == "Alice"
    assert parsed[0]["rid"] == "face-1"
    assert parsed[1]["name"] == "Bob"
    assert parsed[1]["rid"] == "face-2"


def test_existing_persons_in_image_preserved(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    merge_persons_xmp(xmp, ["Alice", "Bob"])

    regions = [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]
    merge_face_regions_xmp(xmp, regions)

    names = read_person_in_image(xmp)
    assert "Alice" in names
    assert "Bob" in names
    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Alice"


def test_second_call_replaces_regions(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    first = [{"name": "Alice", "rx": 0.0, "ry": 0.0, "rw": 0.2, "rh": 0.2}]
    second = [
        {"name": "Bob", "rx": 0.5, "ry": 0.5, "rw": 0.2, "rh": 0.2},
        {"name": "Carol", "rx": 0.1, "ry": 0.1, "rw": 0.15, "rh": 0.15},
    ]

    merge_face_regions_xmp(xmp, first)
    merge_face_regions_xmp(xmp, second)

    parsed = _parse_face_regions(xmp)
    names = [r["name"] for r in parsed]
    assert "Alice" not in names
    assert "Bob" in names
    assert "Carol" in names


def test_empty_regions_removes_face_entries(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    merge_face_regions_xmp(xmp, [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}])
    assert len(_parse_face_regions(xmp)) == 1

    merge_face_regions_xmp(xmp, [])
    assert _parse_face_regions(xmp) == []


def test_zero_width_region_skipped(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    regions = [
        {"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.0, "rh": 0.2},
        {"name": "Bob", "rx": 0.5, "ry": 0.5, "rw": 0.2, "rh": 0.0},
        {"name": "Carol", "rx": 0.2, "ry": 0.2, "rw": 0.1, "rh": 0.1},
    ]

    merge_face_regions_xmp(xmp, regions)
    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Carol"


def test_empty_name_region_skipped(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    regions = [
        {"name": "", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2},
        {"name": "  ", "rx": 0.3, "ry": 0.3, "rw": 0.1, "rh": 0.1},
        {"name": "Alice", "rx": 0.5, "ry": 0.5, "rw": 0.2, "rh": 0.2},
    ]

    merge_face_regions_xmp(xmp, regions)
    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Alice"


def test_non_face_regions_preserved(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    # Manually write an XMP with a non-face ImageRegion
    raw = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">'
        "<Iptc4xmpExt:ImageRegion>"
        "<rdf:Bag>"
        '<rdf:li rdf:parseType="Resource">'
        "<Iptc4xmpExt:RCtype>content-region</Iptc4xmpExt:RCtype>"
        "<Iptc4xmpExt:rId>content-1</Iptc4xmpExt:rId>"
        "</rdf:li>"
        "</rdf:Bag>"
        "</Iptc4xmpExt:ImageRegion>"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    )
    xmp.write_text(raw, encoding="utf-8")

    merge_face_regions_xmp(xmp, [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}])

    parsed = _parse_face_regions(xmp)
    rctypes = [r["rctype"] for r in parsed]
    assert "content-region" in rctypes
    assert "face-identified" in rctypes


def test_corrupt_xmp_creates_new_sidecar(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    xmp.write_text("not valid xml <<<", encoding="utf-8")

    regions = [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]
    merge_face_regions_xmp(xmp, regions)

    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Alice"
