"""Tests for xmpmm_provenance: DocumentID, DerivedFrom, Pantry, write_creation_provenance."""

from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.xmpmm_provenance import (
    assign_document_id,
    read_document_id,
    write_creation_provenance,
    write_derived_from,
    write_pantry_entry,
)


def _minimal_xmp(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""/>'
        "</rdf:RDF>"
        "</x:xmpmeta>",
        encoding="utf-8",
    )


def _xmp_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestAssignDocumentId(unittest.TestCase):
    def test_assigns_uuid_on_first_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            doc_id = assign_document_id(xmp)
            self.assertTrue(doc_id.startswith("xmp:uuid:"))
            self.assertIn(doc_id, _xmp_text(xmp))

    def test_idempotent_second_call_returns_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            id1 = assign_document_id(xmp)
            id2 = assign_document_id(xmp)
            self.assertEqual(id1, id2)

    def test_creates_file_if_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "new.xmp"
            self.assertFalse(xmp.exists())
            doc_id = assign_document_id(xmp)
            self.assertTrue(xmp.exists())
            self.assertTrue(doc_id.startswith("xmp:uuid:"))

    def test_preserves_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            xmp.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                "<dc:description>Keep me</dc:description>"
                "</rdf:Description>"
                "</rdf:RDF>"
                "</x:xmpmeta>",
                encoding="utf-8",
            )
            assign_document_id(xmp)
            self.assertIn("Keep me", _xmp_text(xmp))


class TestWriteDerivedFrom(unittest.TestCase):
    def test_writes_source_document_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            write_derived_from(xmp, "xmp:uuid:abc123", "source.tif")
            xml = _xmp_text(xmp)
            self.assertIn("DerivedFrom", xml)
            self.assertIn("abc123", xml)
            self.assertIn("source.tif", xml)

    def test_overwrites_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            write_derived_from(xmp, "xmp:uuid:first", "first.tif")
            write_derived_from(xmp, "xmp:uuid:second", "second.tif")
            xml = _xmp_text(xmp)
            self.assertIn("second", xml)
            self.assertNotIn("first", xml)


class TestWritePantryEntry(unittest.TestCase):
    def test_adds_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            write_pantry_entry(xmp, "xmp:uuid:scan1", "scan1.tif")
            xml = _xmp_text(xmp)
            self.assertIn("Pantry", xml)
            self.assertIn("scan1", xml)

    def test_deduplicates_by_document_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            write_pantry_entry(xmp, "xmp:uuid:dup", "dup.tif")
            write_pantry_entry(xmp, "xmp:uuid:dup", "dup.tif")
            root = ET.parse(xmp).getroot()
            li_elements = root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")
            self.assertEqual(len(li_elements), 1)

    def test_adds_multiple_distinct_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "test.xmp"
            _minimal_xmp(xmp)
            write_pantry_entry(xmp, "xmp:uuid:s1", "scan1.tif")
            write_pantry_entry(xmp, "xmp:uuid:s2", "scan2.tif")
            xml = _xmp_text(xmp)
            self.assertIn("s1", xml)
            self.assertIn("s2", xml)


class TestWriteCreationProvenance(unittest.TestCase):
    def test_writes_derived_from_and_pantry(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "view.xmp"
            _minimal_xmp(xmp)
            write_creation_provenance(
                xmp,
                derived_from={"source_document_id": "xmp:uuid:primary", "source_path": "primary.tif"},
                pantry_sources=[
                    {"source_document_id": "xmp:uuid:primary", "source_path": "primary.tif"},
                    {"source_document_id": "xmp:uuid:secondary", "source_path": "secondary.tif"},
                ],
            )
            xml = _xmp_text(xmp)
            self.assertIn("DerivedFrom", xml)
            self.assertIn("Pantry", xml)
            self.assertIn("primary", xml)
            self.assertIn("secondary", xml)

    def test_preserves_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "view.xmp"
            xmp.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                "<dc:description>Preserved</dc:description>"
                "</rdf:Description>"
                "</rdf:RDF>"
                "</x:xmpmeta>",
                encoding="utf-8",
            )
            write_creation_provenance(
                xmp,
                derived_from={"source_document_id": "xmp:uuid:x"},
                pantry_sources=[],
            )
            self.assertIn("Preserved", _xmp_text(xmp))

    def test_single_scan_pantry(self):
        """Single-scan page: DerivedFrom and one Pantry entry for the same scan."""
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "view.xmp"
            _minimal_xmp(xmp)
            write_creation_provenance(
                xmp,
                derived_from={"source_document_id": "xmp:uuid:only", "source_path": "page_S01.tif"},
                pantry_sources=[{"source_document_id": "xmp:uuid:only", "source_path": "page_S01.tif"}],
            )
            xml = _xmp_text(xmp)
            self.assertIn("DerivedFrom", xml)
            self.assertIn("Pantry", xml)

    def test_multi_scan_pantry(self):
        """Multi-scan stitch: DerivedFrom points to S01, Pantry has both."""
        with tempfile.TemporaryDirectory() as tmp:
            xmp = Path(tmp) / "view.xmp"
            _minimal_xmp(xmp)
            write_creation_provenance(
                xmp,
                derived_from={"source_document_id": "xmp:uuid:s01", "source_path": "page_S01.tif"},
                pantry_sources=[
                    {"source_document_id": "xmp:uuid:s01", "source_path": "page_S01.tif"},
                    {"source_document_id": "xmp:uuid:s02", "source_path": "page_S02.tif"},
                ],
            )
            xml = _xmp_text(xmp)
            self.assertIn("s01", xml)
            self.assertIn("s02", xml)
            root = ET.parse(xmp).getroot()
            pantry_lis = root.findall(
                ".//{http://ns.adobe.com/xap/1.0/mm/}Pantry"
                "/{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag"
                "/{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
            )
            self.assertEqual(len(pantry_lis), 2)


if __name__ == "__main__":
    unittest.main()
