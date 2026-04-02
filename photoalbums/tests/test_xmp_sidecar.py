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
                creator_tool="imago-test",
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
            self.assertIn("imago-test", xml)
            self.assertIn("Welcome to the park", xml)
            self.assertIn("Mainland China Book II", xml)
            self.assertIn("GPSLatitude", xml)
            self.assertIn("39,47.25N", xml)
            self.assertIn("GPSLongitude", xml)
            self.assertIn("100,18.43332E", xml)
            self.assertIn("GPSMapDatum", xml)
            self.assertIn("Family_2020_B01_P01_S01.tif", xml)
            self.assertIn("ImageRegion", xml)
            self.assertNotIn("RegionInfo", xml)
            self.assertIn("A dog in the park.", xml)
            self.assertNotIn("SubPhotos", xml)

    def test_write_xmp_sidecar_writes_location_shown_bag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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
                creator_tool="imago-test",
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

    def test_write_xmp_sidecar_omits_create_date_and_writes_processing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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
            self.assertNotIn("xmp:CreateDate", xml)
            self.assertIn("xmpMM:History", xml)
            self.assertNotIn("imago:StitchKey", xml)
            self.assertNotIn("imago:OcrRan", xml)
            self.assertNotIn("imago:PeopleDetected", xml)
            self.assertNotIn("imago:PeopleIdentified", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["create_date"], "")
            self.assertEqual(state["stitch_key"], "Family_1986_B01_P01")
            self.assertEqual(state["ocr_authority_source"], "archive_stitched")
            self.assertEqual(state["ocr_ran"], True)
            self.assertEqual(state["people_detected"], True)
            self.assertEqual(state["people_identified"], False)
            history = state["processing_history"]
            assert isinstance(history, list)
            self.assertEqual(len(history), 3)
            self.assertEqual(history[0]["when"], "2026-03-25T19:35:00Z")

    def test_write_xmp_sidecar_round_trips_dc_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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
                creator_tool="imago-test",
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
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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
            self.assertEqual(state["description"], "Temple of Heaven")
            xml = out.read_text(encoding="utf-8")
            self.assertIn('xml:lang="x-default">Temple of Heaven</rdf:li>', xml)
            self.assertIn('xml:lang="x-caption">Ignored summary</rdf:li>', xml)
            self.assertIn('xml:lang="x-scene">NO SMOKING</rdf:li>', xml)
            self.assertIn("<imago:OCRText>Temple of Heaven", xml)
            self.assertIn("<imago:AuthorText>Temple of Heaven</imago:AuthorText>", xml)
            self.assertIn("<imago:SceneText>NO SMOKING</imago:SceneText>", xml)
            self.assertEqual(state["author_text"], "Temple of Heaven")
            self.assertEqual(state["scene_text"], "NO SMOKING")

    def test_write_xmp_sidecar_puts_ocr_text_in_dc_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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

    def test_write_xmp_sidecar_migrates_legacy_processing_fields_to_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "legacy.xmp"
            out.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <xmp:CreatorTool>imago-test</xmp:CreatorTool>
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
                creator_tool="imago-test",
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
            self.assertIn("xmpMM:History", xml)
            self.assertNotIn("imago:StitchKey", xml)
            self.assertNotIn("imago:OcrRan", xml)
            self.assertNotIn("imago:PeopleDetected", xml)
            self.assertNotIn("imago:PeopleIdentified", xml)

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["stitch_key"], "true")
            self.assertEqual(state["ocr_ran"], True)
            self.assertEqual(state["people_detected"], False)
            self.assertEqual(state["people_identified"], False)

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
                creator_tool="imago-test",
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
            self.assertIn('xml:lang="x-default">Dolores Cordell</rdf:li>', xml)
            self.assertIn('xml:lang="x-caption">Updated description</rdf:li>', xml)
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
                creator_tool="imago-test",
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

    def test_write_xmp_sidecar_removes_legacy_xmp_create_date_on_merge(self):
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
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            xml = out.read_text(encoding="utf-8")
            self.assertNotIn("xmp:CreateDate", xml)

    def test_sidecar_has_expected_ai_fields_detects_complete_and_incomplete_sidecars(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "complete.xmp"
            incomplete = Path(tmp) / "incomplete.xmp"

            xmp_sidecar.write_xmp_sidecar(
                complete,
                creator_tool="imago-test",
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
      <xmp:CreatorTool>imago-test</xmp:CreatorTool>
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
                    creator_tool="imago-test",
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="local",
                    caption_engine="template",
                )
            )
            self.assertFalse(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    incomplete,
                    creator_tool="imago-test",
                    enable_people=True,
                    enable_objects=True,
                    ocr_engine="local",
                    caption_engine="template",
                )
            )

    def test_sidecar_has_expected_ai_fields_rejects_creator_tool_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "complete.xmp"
            xmp_sidecar.write_xmp_sidecar(
                complete,
                creator_tool="imago-photoalbums-ai-index",
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

            self.assertFalse(
                xmp_sidecar.sidecar_has_expected_ai_fields(
                    complete,
                    creator_tool="https://github.com/cove/imago",
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
                creator_tool="imago-test",
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
                    creator_tool="imago-test",
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
                creator_tool="imago-test",
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
                    creator_tool="imago-test",
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
                creator_tool="imago-test",
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
            self.assertEqual(state["gps_latitude"], "39,47.25N")
            self.assertEqual(state["gps_longitude"], "100,18.43332E")

    def test_write_xmp_sidecar_derives_page_and_scan_from_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "China_1986_B02_P17_S01.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )
            xml = out.read_text(encoding="utf-8")
            self.assertIn("PageNumber", xml)
            self.assertIn(">17<", xml)
            self.assertIn("ScanNumber", xml)
            self.assertIn(">1<", xml)

    def test_write_xmp_sidecar_omits_page_and_scan_for_unknown_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
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
                creator_tool="imago-test",
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
                creator_tool="imago-test",
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


if __name__ == "__main__":
    unittest.main()
