import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import xmp_sidecar


class TestXMPSidecar(unittest.TestCase):
    def test_write_xmp_sidecar_outputs_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=["Alice", "Bob", "Alice"],
                subjects=["dog", "park", "dog"],
                description="People: Alice, Bob. Objects: dog, park.",
                album_title="Mainland China Book II",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                source_text="Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
                ocr_text="Welcome to the park",
                detections_payload={"objects": [{"label": "dog", "score": 0.9}]},
                subphotos=[
                    {
                        "index": 1,
                        "bounds": {"x": 10, "y": 20, "width": 300, "height": 200},
                        "description": "A dog in the park.",
                        "people": ["Alice"],
                        "subjects": ["dog", "park"],
                    }
                ],
                image_width=1000,
                image_height=1500,
            )

            self.assertTrue(out.exists())
            root = ET.parse(out).getroot()
            xml = ET.tostring(root, encoding="unicode")
            self.assertIn("Alice", xml)
            self.assertIn("Bob", xml)
            self.assertIn("dog", xml)
            self.assertIn("Welcome to the park", xml)
            self.assertIn("Mainland China Book II", xml)
            self.assertIn("GPSLatitude", xml)
            self.assertIn("39,47.25N", xml)
            self.assertIn("GPSLongitude", xml)
            self.assertIn("100,18.43332E", xml)
            self.assertIn("GPSMapDatum", xml)
            self.assertIn("Family_2020_B01_P01_S01.tif", xml)
            self.assertNotIn("ImageRegion", xml)
            self.assertNotIn("RegionInfo", xml)
            self.assertNotIn("SubPhotos", xml)

    def test_write_xmp_sidecar_writes_location_shown_bag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="BUDAPEST,HUNGARY - AUGUST 1988",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                    "location_shown_ran": True,
                    "locations_shown": [
                        {
                            "name": "Fisherman's Bastion",
                            "world_region": "Europe",
                            "country_code": "HU",
                            "country_name": "Hungary",
                            "province_or_state": "",
                            "city": "Budapest",
                            "sublocation": "Fisherman's Bastion",
                            "gps_latitude": "47.5020",
                            "gps_longitude": "19.0340",
                        }
                    ],
                },
                locations_shown=[
                    {
                        "name": "Fisherman's Bastion",
                        "world_region": "Europe",
                        "country_code": "HU",
                        "country_name": "Hungary",
                        "province_or_state": "",
                        "city": "Budapest",
                        "sublocation": "Fisherman's Bastion",
                        "gps_latitude": "47.5020",
                        "gps_longitude": "19.0340",
                    }
                ],
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("LocationShown", xml)
            self.assertIn("LocationName", xml)
            self.assertIn("<Iptc4xmpExt:CountryCode>HU</Iptc4xmpExt:CountryCode>", xml)
            self.assertIn("<Iptc4xmpExt:City>Budapest</Iptc4xmpExt:City>", xml)
            self.assertIn("<exif:GPSLatitude>47,30.12N</exif:GPSLatitude>", xml)
            self.assertIn("<exif:GPSLongitude>19,2.04E</exif:GPSLongitude>", xml)
            self.assertIn(
                "<Iptc4xmpExt:Sublocation>Fisherman's Bastion</Iptc4xmpExt:Sublocation>",
                xml,
            )
            self.assertEqual(
                xmp_sidecar.read_locations_shown(out),
                [
                    {
                        "name": "Fisherman's Bastion",
                        "world_region": "Europe",
                        "country_code": "HU",
                        "country_name": "Hungary",
                        "province_or_state": "",
                        "city": "Budapest",
                        "sublocation": "Fisherman's Bastion",
                        "gps_latitude": "47.502",
                        "gps_longitude": "19.034",
                    }
                ],
            )

    def test_write_xmp_sidecar_round_trips_ocr_authority_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_authority_source="archive_stitched",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["ocr_authority_source"], "archive_stitched")

    def test_write_xmp_sidecar_round_trips_top_level_sublocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                location_city="Vienna",
                location_country="Austria",
                location_sublocation="1 Rathausplatz",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn(
                "<Iptc4xmpExt:LocationCreated>1 Rathausplatz, Vienna, Austria</Iptc4xmpExt:LocationCreated>",
                xml,
            )
            self.assertIn("<photoshop:City>Vienna</photoshop:City>", xml)
            self.assertIn("<photoshop:Country>Austria</photoshop:Country>", xml)
            self.assertIn("<Iptc4xmpExt:Sublocation>1 Rathausplatz</Iptc4xmpExt:Sublocation>", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["location_city"], "Vienna")
            self.assertEqual(state["location_country"], "Austria")
            self.assertEqual(state["location_sublocation"], "1 Rathausplatz")
            self.assertEqual(state["location_created"], "1 Rathausplatz, Vienna, Austria")

    def test_write_xmp_sidecar_round_trips_create_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="scan_001.tif",
                ocr_text="HELLO",
                ocr_authority_source="archive_stitched",
                create_date="2026:03:25 12:34:56-07:00",
                history_when="2026-03-25T19:35:00Z",
                stitch_key="Family_1986_B01_P01",
                ocr_ran=True,
                people_detected=True,
                people_identified=False,
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("<xmp:CreateDate>2026-03-25T12:34:56-07:00</xmp:CreateDate>", xml)
            self.assertNotIn("xmpMM:History", xml)
            self.assertNotIn("imago:StitchKey", xml)
            self.assertNotIn("imago:OcrRan", xml)
            self.assertNotIn("imago:PeopleDetected", xml)
            self.assertNotIn("imago:PeopleIdentified", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["create_date"], "2026-03-25T12:34:56-07:00")
            self.assertEqual(state["ocr_authority_source"], "archive_stitched")
            history = state["processing_history"]
            assert isinstance(history, list)
            self.assertEqual(len(history), 0)

    def test_write_xmp_sidecar_round_trips_dc_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="scan_001.tif",
                ocr_text="March 15, 1975",
                dc_date="1975-03-15",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("<dc:date>", xml)
            self.assertIn("<rdf:Seq>", xml)
            self.assertIn(">1975-03-15<", xml)
            self.assertIn("<exif:DateTimeOriginal>1975-03-15T12:00:00</exif:DateTimeOriginal>", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["dc_date"], "1975-03-15")
            self.assertEqual(state["dc_date_values"], ["1975-03-15"])
            self.assertEqual(state["date_time_original"], "1975-03-15T12:00:00")

    def test_write_xmp_sidecar_round_trips_multiple_dc_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="scan_001.tif",
                ocr_text="September 1934 and October 1934",
                dc_date=["1934-09", "1934-10"],
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("<dc:date>", xml)
            self.assertEqual(xml.count("<rdf:li>1934-09</rdf:li>"), 1)
            self.assertEqual(xml.count("<rdf:li>1934-10</rdf:li>"), 1)
            self.assertIn("<exif:DateTimeOriginal>1934-09-15T12:00:00</exif:DateTimeOriginal>", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["dc_date"], "1934-09")
            self.assertEqual(state["dc_date_values"], ["1934-09", "1934-10"])
            self.assertEqual(state["date_time_original"], "1934-09-15T12:00:00")

    def test_normalize_dc_date_coerces_partial_formatting_errors(self):
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988-1"), "1988-01")
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988/01/00"), "1988-01")
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988-00"), "1988")
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988-00-00"), "1988")
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988:1:5"), "1988-01-05")
        self.assertEqual(xmp_sidecar._normalize_dc_date("1988-13-00"), "")

    def test_resolve_date_time_original_uses_midpoints_for_partial_dates(self):
        self.assertEqual(
            xmp_sidecar._resolve_date_time_original(dc_date="1975"),
            "1975-07-01T12:00:00",
        )
        self.assertEqual(
            xmp_sidecar._resolve_date_time_original(dc_date="1975-03"),
            "1975-03-15T12:00:00",
        )
        self.assertEqual(
            xmp_sidecar._resolve_date_time_original(dc_date="1975-03-22"),
            "1975-03-22T12:00:00",
        )

    def test_write_xmp_sidecar_round_trips_text_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages_dir = Path(tmp) / "China_1986_B02_Pages"
            pages_dir.mkdir()
            out = pages_dir / "China_1986_B02_P01_V.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                title="Temple of Heaven",
                title_source="author_text",
                description="Ignored summary",
                source_text="",
                ocr_text="Temple of Heaven\nNO SMOKING",
                author_text="Temple of Heaven",
                scene_text="NO SMOKING",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["title"], "Temple of Heaven")
            self.assertEqual(state["title_source"], "author_text")
            self.assertEqual(state["description"], "Caption:\nTemple of Heaven\nNO SMOKING\n\nScene Text:\nNO SMOKING")
            xml = out.read_text(encoding="utf-8")
            self.assertIn('xml:lang="x-default">Caption:\nTemple of Heaven\nNO SMOKING\n\nScene Text:\nNO SMOKING</rdf:li>', xml)
            self.assertNotIn('xml:lang="x-caption"', xml)
            self.assertNotIn('xml:lang="x-scene"', xml)
            self.assertIn("<imago:OCRText>Temple of Heaven", xml)
            self.assertIn("<imago:AuthorText>Temple of Heaven</imago:AuthorText>", xml)
            self.assertIn("<imago:SceneText>NO SMOKING</imago:SceneText>", xml)
            self.assertEqual(state["author_text"], "Temple of Heaven")
            self.assertEqual(state["scene_text"], "NO SMOKING")

    def test_region_list_round_trips_location_override_and_assigned_payload(self):
        from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "page.xmp"
            xmp_sidecar.write_region_list(
                out,
                [
                    RegionWithCaption(
                        RegionResult(
                            index=0,
                            x=0,
                            y=0,
                            width=400,
                            height=300,
                            caption_hint="Temple visit",
                            location_payload={
                                "gps_latitude": "25.6872",
                                "gps_longitude": "32.6396",
                                "city": "Luxor",
                                "country": "Egypt",
                            },
                        ),
                        "",
                    ),
                ],
                800,
                600,
            )

            tree = ET.parse(out)
            desc = xmp_sidecar._get_rdf_desc(tree)
            assert desc is not None
            first_li = next(desc.iter(f"{{{xmp_sidecar.RDF_NS}}}li"))
            xmp_sidecar._add_region_location_struct(
                first_li,
                f"{{{xmp_sidecar.IMAGO_NS}}}LocationOverride",
                {"address": "Luxor Temple", "city": "Luxor", "country": "Egypt"},
            )
            tree.write(out, encoding="utf-8", xml_declaration=True)

            regions = xmp_sidecar.read_region_list(out, 800, 600)
            xml = out.read_text(encoding="utf-8")

        self.assertEqual(regions[0]["location_payload"]["city"], "Luxor")
        self.assertEqual(regions[0]["location_override"]["address"], "Luxor Temple")
        self.assertIn("<Iptc4xmpExt:City>Luxor</Iptc4xmpExt:City>", xml)
        self.assertIn("<Iptc4xmpExt:CountryName>Egypt</Iptc4xmpExt:CountryName>", xml)
        self.assertNotIn("photoshop:City", xml)
        self.assertNotIn("photoshop:Country", xml)

    def test_write_region_list_preserves_existing_location_override(self):
        from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "page.xmp"
            xmp_sidecar.write_region_list(
                out,
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=300), "Temple visit")],
                800,
                600,
            )

            tree = ET.parse(out)
            desc = xmp_sidecar._get_rdf_desc(tree)
            assert desc is not None
            first_li = next(desc.iter(f"{{{xmp_sidecar.RDF_NS}}}li"))
            xmp_sidecar._add_region_location_struct(
                first_li,
                f"{{{xmp_sidecar.IMAGO_NS}}}LocationOverride",
                {"address": "Luxor Temple", "city": "Luxor", "country": "Egypt"},
            )
            tree.write(out, encoding="utf-8", xml_declaration=True)

            xmp_sidecar.write_region_list(
                out,
                [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=300), "Updated caption")],
                800,
                600,
            )

            regions = xmp_sidecar.read_region_list(out, 800, 600)
            xml = out.read_text(encoding="utf-8")

        self.assertEqual(regions[0]["caption"], "Updated caption")
        self.assertEqual(regions[0]["location_override"]["address"], "Luxor Temple")
        self.assertIn("<Iptc4xmpExt:City>Luxor</Iptc4xmpExt:City>", xml)
        self.assertIn("<Iptc4xmpExt:CountryName>Egypt</Iptc4xmpExt:CountryName>", xml)
        self.assertNotIn("photoshop:City", xml)
        self.assertNotIn("photoshop:Country", xml)

    def test_write_xmp_sidecar_normalizes_literal_newline_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                title="Line 1\\nLine 2",
                description="First line\\nSecond line",
                source_text="Family_2020\\nB01_P01_S01.tif",
                ocr_text="OCR line 1\\nOCR line 2",
                author_text="Author line 1\\nAuthor line 2",
                scene_text="Scene line 1\\nScene line 2",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("<dc:title>", xml)
            self.assertIn("Line 1\nLine 2", xml)
            self.assertIn("First line\nSecond line", xml)
            self.assertIn("OCR line 1\nOCR line 2", xml)
            self.assertIn("Author line 1\nAuthor line 2", xml)
            self.assertIn("Scene line 1\nScene line 2", xml)
            self.assertIn("<dc:source>Family_2020 B01_P01_S01.tif</dc:source>", xml)

    def test_write_xmp_sidecar_puts_ocr_text_in_dc_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="EASTERN EUROPE SPAIN AND MOROCCO 1988",
                ocr_lang="en",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn('xml:lang="x-default">EASTERN EUROPE SPAIN AND MOROCCO 1988</rdf:li>', xml)
            self.assertIn("<imago:OCRText>EASTERN EUROPE SPAIN AND MOROCCO 1988</imago:OCRText>", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["description"], "EASTERN EUROPE SPAIN AND MOROCCO 1988")
            self.assertEqual(state["ocr_text"], "EASTERN EUROPE SPAIN AND MOROCCO 1988")

    def test_write_xmp_sidecar_strips_legacy_processing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "legacy.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <imago:StitchKey>true</imago:StitchKey>
      <imago:OcrRan>true</imago:OcrRan>
      <imago:PeopleDetected>false</imago:PeopleDetected>
      <imago:PeopleIdentified>false</imago:PeopleIdentified>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                stitch_key="true",
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertNotIn("xmpMM:History", xml)
            self.assertNotIn("imago:StitchKey", xml)
            self.assertNotIn("imago:OcrRan", xml)
            self.assertNotIn("imago:PeopleDetected", xml)
            self.assertNotIn("imago:PeopleIdentified", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["stitch_key"], "")
            self.assertIsNone(state["ocr_ran"])
            self.assertIsNone(state["people_detected"])
            self.assertIsNone(state["people_identified"])

    def test_write_xmp_sidecar_merges_existing_fields_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:custom="https://example.com/custom/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <custom:KeepMe>Preserve this field</custom:KeepMe>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Old description</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=["Dolores Cordell"],
                subjects=["baby"],
                description="Updated description",
                album_title="Family Book I",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                source_text="Family_1986_B01_P02_S01.tif",
                ocr_text="Dolores Cordell",
                detections_payload={"people": [{"name": "Dolores Cordell", "score": 0.95}]},
                subphotos=[],
            )

            root = ET.parse(out).getroot()
            xml = ET.tostring(root, encoding="unicode")
            self.assertIn("Preserve this field", xml)
            self.assertIn('xml:lang="x-default">Updated description</rdf:li>', xml)
            self.assertNotIn('xml:lang="x-caption"', xml)
            self.assertIn("Updated description", xml)
            self.assertIn("Dolores Cordell", xml)
            self.assertIn("Family Book I", xml)
            self.assertIn("39,47.25N", xml)
            self.assertNotIn("Old description", xml)

    def test_write_xmp_sidecar_merges_locations_shown_into_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:custom="https://example.com/custom/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <custom:KeepMe>Preserve this field</custom:KeepMe>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="OXFORD STREET LONDON,ENGLAND",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                    "location_shown_ran": True,
                    "locations_shown": [
                        {
                            "name": "Oxford Street",
                            "world_region": "Europe",
                            "country_code": "",
                            "country_name": "United Kingdom",
                            "province_or_state": "",
                            "city": "London",
                            "sublocation": "",
                            "gps_latitude": "51.5154",
                            "gps_longitude": "-0.1410",
                        }
                    ],
                },
                locations_shown=[
                    {
                        "name": "Oxford Street",
                        "world_region": "Europe",
                        "country_code": "",
                        "country_name": "United Kingdom",
                        "province_or_state": "",
                        "city": "London",
                        "sublocation": "",
                        "gps_latitude": "51.5154",
                        "gps_longitude": "-0.1410",
                    }
                ],
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("Preserve this field", xml)
            self.assertIn("LocationShown", xml)
            self.assertIn("Oxford Street", xml)
            self.assertIn("<Iptc4xmpExt:CountryName>United Kingdom</Iptc4xmpExt:CountryName>", xml)
            self.assertIn("<Iptc4xmpExt:City>London</Iptc4xmpExt:City>", xml)
            self.assertIn("<exif:GPSLatitude>51,30.924N</exif:GPSLatitude>", xml)
            self.assertIn("<exif:GPSLongitude>0,8.46W</exif:GPSLongitude>", xml)
            self.assertEqual(
                xmp_sidecar.read_locations_shown(out),
                [
                    {
                        "name": "Oxford Street",
                        "world_region": "Europe",
                        "country_code": "",
                        "country_name": "United Kingdom",
                        "province_or_state": "",
                        "city": "London",
                        "sublocation": "",
                        "gps_latitude": "51.5154",
                        "gps_longitude": "-0.141",
                    }
                ],
            )

    def test_write_xmp_sidecar_preserves_xmp_create_date_on_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <xmp:CreateDate>2026-03-25T12:34:56-07:00</xmp:CreateDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("<xmp:CreateDate>2026-03-25T12:34:56-07:00</xmp:CreateDate>", xml)

    def test_write_xmp_sidecar_preserves_inherited_rendered_metadata_on_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Egypt_1975_B00_P09_V.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=["travel", "egypt"],
                description="Pyramids at Giza",
                source_text="Egypt_1975_B00_P09_S01.tif; Egypt_1975_B00_P09_S02.tif",
                ocr_text="EGYPT 1975",
                author_text="Pyramids at Giza",
                scene_text="Tour bus nearby",
                location_city="Giza",
                location_country="Egypt",
                location_sublocation="Giza Plateau",
                create_date="2026-03-25T19:35:00Z",
                dc_date=["1975-03", "1975-04"],
                locations_shown=[
                    {
                        "name": "Giza Necropolis",
                        "world_region": "Africa",
                        "country_code": "EG",
                        "country_name": "Egypt",
                        "province_or_state": "Giza",
                        "city": "Giza",
                        "sublocation": "Giza Plateau",
                        "gps_latitude": "29.9792",
                        "gps_longitude": "31.1342",
                    }
                ],
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=["Alice Smith"],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={"people": [{"name": "Alice Smith", "bbox": [1, 2, 3, 4]}], "objects": [], "ocr": {}, "caption": {}},
                image_width=200,
                image_height=100,
            )

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["description"], "Pyramids at Giza")
            self.assertEqual(state["source_text"], "Egypt_1975_B00_P09_S01.tif; Egypt_1975_B00_P09_S02.tif")
            self.assertEqual(state["ocr_text"], "EGYPT 1975")
            self.assertEqual(state["author_text"], "Pyramids at Giza")
            self.assertEqual(state["scene_text"], "Tour bus nearby")
            self.assertEqual(state["create_date"], "2026-03-25T19:35:00Z")
            self.assertEqual(state["dc_date_values"], ["1975-03", "1975-04"])
            self.assertEqual(state["location_city"], "Giza")
            self.assertEqual(state["location_country"], "Egypt")
            self.assertEqual(state["location_sublocation"], "Giza Plateau")
            self.assertEqual(xmp_sidecar.read_locations_shown(out)[0]["name"], "Giza Necropolis")
            xml = out.read_text(encoding="utf-8")
            self.assertIn("<rdf:li>travel</rdf:li>", xml)
            self.assertIn("<rdf:li>egypt</rdf:li>", xml)

    def test_sidecar_has_expected_ai_fields_detects_complete_and_incomplete_sidecars(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "complete.xmp"
            incomplete = Path(tmp) / "incomplete.xmp"

            xmp_sidecar.write_xmp_sidecar(
                complete,
                person_names=[],
                subjects=["mainland", "china"],
                description="This is the cover or title page of Mainland China Book II.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "chars": 27,
                        "keywords": ["mainland"],
                    },
                    "caption": {
                        "requested_engine": "template",
                        "effective_engine": "template",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                subphotos=[],
            )
            incomplete.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Old description</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            self.assertTrue(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    complete,
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="local",
                    caption_engine="template",
                )
            )
            self.assertFalse(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    incomplete,
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="local",
                    caption_engine="template",
                )
            )

    def test_sidecar_has_expected_ai_fields_rejects_reasoning_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "reasoning.xmp"
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                person_names=[],
                subjects=["person"],
                description=(
                    "The user wants a detailed caption for the provided photo collage. "
                    "**1. Analyze the Input Data:** "
                    "* **Filename:** `China_1986_B02_P02_stitched.jpg`"
                ),
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [{"label": "person", "score": 0.84}],
                    "ocr": {
                        "engine": "lmstudio",
                        "language": "eng",
                        "chars": 0,
                        "keywords": [],
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                subphotos=[],
            )

            self.assertFalse(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    sidecar,
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="lmstudio",
                    caption_engine="lmstudio",
                )
            )

    def test_sidecar_has_expected_ai_fields_rejects_reasoning_ocr_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "reasoning_ocr.xmp"
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text=(
                    "The user wants me to extract text from the provided image.\n"
                    "1. **Analyze the image:**\n"
                    "2. **Transcribe the text found:**"
                ),
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "lmstudio",
                        "language": "eng",
                        "chars": 120,
                        "keywords": [],
                    },
                    "caption": {
                        "requested_engine": "template",
                        "effective_engine": "template",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                subphotos=[],
            )

            self.assertFalse(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    sidecar,
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="lmstudio",
                    caption_engine="template",
                )
            )

    def test_read_ai_sidecar_state_returns_album_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "album_title.xmp"
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                person_names=[],
                subjects=[],
                title="Album page caption",
                description="This is the cover or title page of Mainland China Book II.",
                album_title="Mainland China Book II",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            state = xmp_sidecar.read_ai_sidecar_state(sidecar)
            assert state is not None
            self.assertEqual(state["title"], "Album page caption")
            self.assertEqual(state["album_title"], "Mainland China Book II")
            self.assertEqual(state["gps_latitude"], "39.7875")
            self.assertEqual(state["gps_longitude"], "100.307222")

    def test_write_xmp_sidecar_derives_page_but_omits_scan_from_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "China_1986_B02_P17_S01.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )
            xml = out.read_text(encoding="utf-8")
            self.assertIn("PageNumber", xml)
            self.assertIn(">17<", xml)
            self.assertNotIn("ScanNumber", xml)

    def test_write_xmp_sidecar_omits_page_and_scan_for_unknown_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )
            xml = out.read_text(encoding="utf-8")
            self.assertNotIn("PageNumber", xml)
            self.assertNotIn("ScanNumber", xml)

    def test_write_xmp_sidecar_writes_album_title_to_imago_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                album_title="Mainland China 1986 Book II",
                source_text="",
                ocr_text="",
            )
            xml = out.read_text(encoding="utf-8")
            self.assertIn("imago:AlbumTitle", xml)
            self.assertIn("Mainland China 1986 Book II", xml)
            self.assertNotIn("xmpDM:album", xml)

    def test_write_xmp_sidecar_removes_legacy_xmpdm_album_on_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <xmpDM:album>Old Album Title</xmpDM:album>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                album_title="New Album Title",
                source_text="",
                ocr_text="",
            )
            xml = out.read_text(encoding="utf-8")
            self.assertNotIn("xmpDM:album", xml)
            self.assertIn("imago:AlbumTitle", xml)
            self.assertIn("New Album Title", xml)

    def test_read_pipeline_state_returns_empty_dict_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            self.assertEqual(xmp_sidecar.read_pipeline_state(out), {})

    def test_write_pipeline_step_preserves_existing_detection_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "location": {"city": "Cairo"},
                    "caption": {"model": "existing-caption-model"},
                },
            )

            xmp_sidecar.write_pipeline_step(
                out,
                "view_regions",
                model="gemma-4",
                extra={"result": "no_regions"},
            )

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            detections = state["detections"]
            assert isinstance(detections, dict)
            self.assertEqual(detections["location"], {"city": "Cairo"})
            self.assertEqual(detections["caption"], {"model": "existing-caption-model"})
            self.assertEqual(
                detections["pipeline"]["view_regions"]["result"],
                "no_regions",
            )
            self.assertEqual(
                detections["pipeline"]["view_regions"]["model"],
                "gemma-4",
            )

    def test_read_pipeline_state_normalises_legacy_completed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "pipeline": {
                        "view_regions": {"completed": "2026-04-11T07:00:00Z", "model": "gemma"},
                    },
                },
            )

            state = xmp_sidecar.read_pipeline_state(out)
            entry = state.get("view_regions")
            assert entry is not None
            self.assertEqual(entry["completed"], "2026-04-11T07:00:00Z")
            self.assertEqual(entry["timestamp"], "2026-04-11T07:00:00Z")
            self.assertEqual(entry["result"], "ok")
            self.assertEqual(entry["input_hash"], "")
            self.assertEqual(entry["model"], "gemma")

    def test_read_pipeline_state_empty_input_hash_treated_as_stale(self):
        """Entries with empty input_hash are always considered stale by StepRunner."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "pipeline": {
                        "ocr": {"timestamp": "2026-04-11T07:00:00Z", "result": "ok", "input_hash": ""},
                    },
                },
            )

            state = xmp_sidecar.read_pipeline_state(out)
            entry = state.get("ocr")
            assert entry is not None
            self.assertEqual(entry["input_hash"], "")

    def test_migrate_pipeline_records_rewrites_legacy_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "pipeline": {
                        "view_regions": {"completed": "2026-04-11T07:00:00Z", "model": "gemma"},
                        "crop_regions": {"timestamp": "2026-04-12T07:00:00Z", "result": "ok", "input_hash": "abc"},
                    },
                },
            )

            changed = xmp_sidecar.migrate_pipeline_records(out)

            self.assertTrue(changed)
            state = xmp_sidecar.read_pipeline_state(out)
            view = state["view_regions"]
            self.assertEqual(view["timestamp"], "2026-04-11T07:00:00Z")
            self.assertEqual(view["completed"], "2026-04-11T07:00:00Z")
            self.assertEqual(view["result"], "ok")
            self.assertEqual(view["input_hash"], "")
            crop = state["crop_regions"]
            self.assertEqual(crop["timestamp"], "2026-04-12T07:00:00Z")
            self.assertEqual(crop["input_hash"], "abc")

    def test_migrate_pipeline_records_leaves_sidecars_without_pipeline_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            changed = xmp_sidecar.migrate_pipeline_records(out)
            self.assertFalse(changed)

    def test_write_pipeline_steps_writes_new_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            xmp_sidecar.write_pipeline_steps(out, {
                "ocr": {"timestamp": "2026-04-11T07:00:00Z", "result": "ok", "input_hash": "abc123"},
            })

            state = xmp_sidecar.read_pipeline_state(out)
            entry = state.get("ocr")
            assert entry is not None
            self.assertEqual(entry["timestamp"], "2026-04-11T07:00:00Z")
            self.assertEqual(entry["result"], "ok")
            self.assertEqual(entry["input_hash"], "abc123")
            self.assertNotIn("completed", entry)

    def test_clear_pipeline_steps_removes_only_named_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "location": {"city": "Cairo"},
                    "pipeline": {
                        "view_regions": {"completed": "2026-04-11T00:00:00Z"},
                        "crop_regions": {"completed": "2026-04-11T01:00:00Z"},
                    },
                },
            )

            xmp_sidecar.clear_pipeline_steps(out, ["view_regions"])

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            detections = state["detections"]
            assert isinstance(detections, dict)
            self.assertEqual(detections["location"], {"city": "Cairo"})
            self.assertNotIn("view_regions", detections["pipeline"])
            self.assertIn("crop_regions", detections["pipeline"])


if __name__ == "__main__":
    unittest.main()
