"""Tests for merge_face_regions_xmp in cast.xmp_writer."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from cast.xmp_writer import (
    merge_face_regions_xmp,
    merge_persons_xmp,
    read_person_in_image,
)

IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
STDIM_NS = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
MP_NS = "http://ns.microsoft.com/photo/1.2/"
MPRI_NS = "http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
MPREG_NS = "http://ns.microsoft.com/photo/1.2/t/Region#"
# ExifTool uses a slightly different stArea namespace URI
EXIFTOOL_STAREA_NS = "http://ns.adobe.com/xmp/sType/Area#"


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


def _parse_exiftool_mwgrs_regions(xmp_path: Path) -> list[dict]:
    """Parse mwg-rs:Regions as written by ExifTool (child text elements)."""
    tree = ET.parse(str(xmp_path))
    root = tree.getroot()
    rdf_rdf = root.find(f"{{{RDF_NS}}}RDF")
    if rdf_rdf is None:
        return []
    regions = []
    # ExifTool may split namespaces across multiple rdf:Description elements
    for desc in rdf_rdf.findall(f"{{{RDF_NS}}}Description"):
        region_info = desc.find(f"{{{MWGRS_NS}}}Regions")
        if region_info is None:
            continue
        applied = region_info.find(f"{{{MWGRS_NS}}}AppliedToDimensions")
        dim_w = applied.findtext(f"{{{STDIM_NS}}}w") if applied is not None else ""
        dim_h = applied.findtext(f"{{{STDIM_NS}}}h") if applied is not None else ""
        region_list = region_info.find(f"{{{MWGRS_NS}}}RegionList")
        if region_list is None:
            continue
        bag = region_list.find(f"{{{RDF_NS}}}Bag")
        if bag is None:
            continue
        for li in bag.findall(f"{{{RDF_NS}}}li"):
            area = li.find(f"{{{MWGRS_NS}}}Area")
            regions.append({
                "type": str(li.findtext(f"{{{MWGRS_NS}}}Type") or ""),
                "name": str(li.findtext(f"{{{MWGRS_NS}}}Name") or ""),
                "x": str(area.findtext(f"{{{EXIFTOOL_STAREA_NS}}}x") if area is not None else ""),
                "y": str(area.findtext(f"{{{EXIFTOOL_STAREA_NS}}}y") if area is not None else ""),
                "w": str(area.findtext(f"{{{EXIFTOOL_STAREA_NS}}}w") if area is not None else ""),
                "h": str(area.findtext(f"{{{EXIFTOOL_STAREA_NS}}}h") if area is not None else ""),
                "unit": str(area.findtext(f"{{{EXIFTOOL_STAREA_NS}}}unit") if area is not None else ""),
                "dim_w": str(dim_w or ""),
                "dim_h": str(dim_h or ""),
            })
    return regions


def _parse_mp_regions(xmp_path: Path) -> list[dict]:
    """Parse MP:RegionInfo as written by ExifTool (child text elements)."""
    tree = ET.parse(str(xmp_path))
    root = tree.getroot()
    rdf_rdf = root.find(f"{{{RDF_NS}}}RDF")
    if rdf_rdf is None:
        return []
    for desc in rdf_rdf.findall(f"{{{RDF_NS}}}Description"):
        region_info = desc.find(f"{{{MP_NS}}}RegionInfo")
        if region_info is None:
            continue
        regions = region_info.find(f"{{{MPRI_NS}}}Regions")
        if regions is None:
            return []
        bag = regions.find(f"{{{RDF_NS}}}Bag")
        if bag is None:
            return []
        return [
            {
                "name": str(li.findtext(f"{{{MPREG_NS}}}PersonDisplayName") or ""),
                "rectangle": str(li.findtext(f"{{{MPREG_NS}}}Rectangle") or ""),
            }
            for li in bag.findall(f"{{{RDF_NS}}}li")
        ]
    return []


def test_create_minimal_sidecar_with_face_regions(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    regions = [{"name": "Alice", "rx": 0.1, "ry": 0.2, "rw": 0.3, "rh": 0.4, "image_width": 1000, "image_height": 800}]

    result = merge_face_regions_xmp(xmp, regions)

    assert result == xmp
    assert xmp.is_file()

    # IPTC regions (written via XML)
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

    # MWG-RS regions (written via ExifTool — mwg-rs:Regions, center-normalized coords)
    mwgrs = _parse_exiftool_mwgrs_regions(xmp)
    assert len(mwgrs) == 1
    assert mwgrs[0]["type"] == "Face"
    assert mwgrs[0]["name"] == "Alice"
    assert mwgrs[0]["unit"] == "normalized"
    assert abs(float(mwgrs[0]["x"]) - 0.25) < 1e-5  # rx + rw/2 = 0.1 + 0.15
    assert abs(float(mwgrs[0]["y"]) - 0.4) < 1e-5   # ry + rh/2 = 0.2 + 0.20
    assert abs(float(mwgrs[0]["w"]) - 0.3) < 1e-5
    assert abs(float(mwgrs[0]["h"]) - 0.4) < 1e-5
    assert mwgrs[0]["dim_w"] == "1000"
    assert mwgrs[0]["dim_h"] == "800"

    # MP regions (written via ExifTool — top-left normalized coords)
    mp = _parse_mp_regions(xmp)
    assert len(mp) == 1
    assert mp[0]["name"] == "Alice"
    assert mp[0]["rectangle"] == "0.100000, 0.200000, 0.300000, 0.400000"


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
    merge_face_regions_xmp(xmp, [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2,
                                   "image_width": 100, "image_height": 100}])
    assert len(_parse_face_regions(xmp)) == 1
    assert len(_parse_exiftool_mwgrs_regions(xmp)) == 1
    assert len(_parse_mp_regions(xmp)) == 1

    merge_face_regions_xmp(xmp, [])
    assert _parse_face_regions(xmp) == []
    assert _parse_exiftool_mwgrs_regions(xmp) == []
    assert _parse_mp_regions(xmp) == []


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


def test_old_compact_mwgrs_region_info_cleared_on_write(tmp_path: Path) -> None:
    """Old hand-written mwg-rs:RegionInfo (compact attrs form) is removed; ExifTool writes mwg-rs:Regions."""
    xmp = tmp_path / "photo.xmp"
    raw = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"'
        ' xmlns:stArea="http://ns.adobe.com/xap/1.0/sType/Area#">'
        '<mwg-rs:RegionInfo rdf:parseType="Resource">'
        "<mwg-rs:RegionList><rdf:Bag>"
        '<rdf:li rdf:parseType="Resource" mwg-rs:Type="Photo" mwg-rs:Name="Existing photo"'
        ' stArea:x="0.5" stArea:y="0.5" stArea:w="0.4" stArea:h="0.4" stArea:unit="normalized" />'
        '<rdf:li rdf:parseType="Resource" mwg-rs:Type="Face" mwg-rs:Name="Old face"'
        ' stArea:x="0.1" stArea:y="0.1" stArea:w="0.1" stArea:h="0.1" stArea:unit="normalized" />'
        "</rdf:Bag></mwg-rs:RegionList>"
        "</mwg-rs:RegionInfo>"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    )
    xmp.write_text(raw, encoding="utf-8")

    merge_face_regions_xmp(
        xmp,
        [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2, "image_width": 1000, "image_height": 800}],
    )

    # Old compact mwg-rs:RegionInfo is cleared (not preserved after migration)
    tree = ET.parse(str(xmp))
    root = tree.getroot()
    rdf_rdf = root.find(f"{{{RDF_NS}}}RDF")
    assert rdf_rdf is not None
    for desc in rdf_rdf.findall(f"{{{RDF_NS}}}Description"):
        assert desc.find(f"{{{MWGRS_NS}}}RegionInfo") is None, "old compact RegionInfo should be gone"

    # ExifTool's mwg-rs:Regions has Alice as Face
    et_mwgrs = _parse_exiftool_mwgrs_regions(xmp)
    assert [r["type"] for r in et_mwgrs] == ["Face"]
    assert [r["name"] for r in et_mwgrs] == ["Alice"]

    # MP has Alice
    assert _parse_mp_regions(xmp) == [{"name": "Alice", "rectangle": "0.100000, 0.100000, 0.200000, 0.200000"}]


def test_corrupt_xmp_creates_new_sidecar(tmp_path: Path) -> None:
    xmp = tmp_path / "photo.xmp"
    xmp.write_text("not valid xml <<<", encoding="utf-8")

    regions = [{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]
    merge_face_regions_xmp(xmp, regions)

    parsed = _parse_face_regions(xmp)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Alice"
