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
            self.assertIn("RegionInfo", xml)
            self.assertIn("A dog in the park.", xml)
            self.assertNotIn("SubPhotos", xml)

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
                description="Temple of Heaven",
                source_text="",
                ocr_text="Temple of Heaven\nNO SMOKING",
                author_text="Temple of Heaven",
                scene_text="NO SMOKING",
                annotation_scope="photo",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            state = xmp_sidecar.read_ai_sidecar_state(out)
            assert state is not None
            self.assertEqual(state["title"], "Temple of Heaven")
            self.assertEqual(state["title_source"], "author_text")
            self.assertEqual(state["author_text"], "Temple of Heaven")
            self.assertEqual(state["scene_text"], "NO SMOKING")
            self.assertEqual(state["annotation_scope"], "photo")

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
            self.assertIn("Updated description", xml)
            self.assertIn("Dolores Cordell", xml)
            self.assertIn("Family Book I", xml)
            self.assertIn("39,47.25N", xml)
            self.assertNotIn("Old description", xml)

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


if __name__ == "__main__":
    unittest.main()
