import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_index, ai_index_runner, xmp_sidecar


def _image_region_ids(xmp_path: Path) -> list[str]:
    root = ET.parse(xmp_path).getroot()
    return [
        str(item.text or "").strip()
        for item in root.findall(
            ".//"
            f"{{{xmp_sidecar.IPTC_EXT_NS}}}ImageRegion/"
            f"{{{xmp_sidecar.RDF_NS}}}Bag/"
            f"{{{xmp_sidecar.RDF_NS}}}li/"
            f"{{{xmp_sidecar.IPTC_EXT_NS}}}rId"
        )
        if str(item.text or "").strip()
    ]


class TestRenderFaceRegionRefresh(unittest.TestCase):
    def test_refresh_rendered_view_people_metadata_uses_fresh_names_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_B00_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            stat = image.stat()
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=["Old Name"],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, []),
                ocr_text="March 1975",
                dc_date="1975-03",
                detections_payload={
                    "people": [
                        {
                            "name": "Old Name",
                            "score": 0.91,
                            "certainty": 0.91,
                            "bbox": [5, 10, 15, 20],
                        }
                    ],
                    "objects": [],
                    "ocr": {"keywords": []},
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "test-model",
                        "people_present": True,
                        "estimated_people_count": 1,
                    },
                    "processing": {
                        "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                        "settings_signature": "",
                        "cast_store_signature": "old-sig",
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "date_estimate_input_hash": "",
                        "analysis_mode": "single_image",
                    },
                },
                image_width=200,
                image_height=100,
                people_detected=True,
                people_identified=True,
            )

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"
            fake_matcher.match_image.return_value = [
                SimpleNamespace(
                    name="New Name",
                    score=0.97,
                    certainty=0.97,
                    bbox=[20, 30, 40, 20],
                    face_id="face-123",
                    reviewed_by_human=False,
                )
            ]
            fake_matcher.last_faces_detected = 1
            fake_caption_engine = mock.Mock()
            fake_caption_engine.effective_model_name = "test-caption"
            fake_caption_engine.generate.return_value = SimpleNamespace(
                engine="lmstudio",
                fallback=False,
                error="",
                engine_error="",
            )

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(ai_index_runner, "_init_caption_engine", return_value=fake_caption_engine),
                mock.patch.object(ai_index_runner, "_compute_people_positions", return_value={"New Name": "center"}),
                mock.patch.object(ai_index_runner, "_resolve_people_count_metadata", return_value=(True, 1)),
                mock.patch.object(ai_index_runner, "_get_image_dimensions", return_value=(200, 100)),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                ai_index_runner.refresh_rendered_view_people_metadata(image)

            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["person_names"], ["New Name"])
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["people"][0]["name"],
                "New Name",
            )
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["processing"]["cast_store_signature"],
                "new-sig",
            )

    def test_write_xmp_sidecar_replaces_face_regions_but_preserves_non_face_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Egypt_1975_B00_P09_V.xmp"
            out.write_text(
                f"""<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="{xmp_sidecar.X_NS}" xmlns:rdf="{xmp_sidecar.RDF_NS}" xmlns:Iptc4xmpExt="{xmp_sidecar.IPTC_EXT_NS}">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <Iptc4xmpExt:ImageRegion>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>face-1</Iptc4xmpExt:rId>
            <Iptc4xmpExt:Name>
              <rdf:Alt>
                <rdf:li xml:lang="x-default">Old Name</rdf:li>
              </rdf:Alt>
            </Iptc4xmpExt:Name>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>photo-1</Iptc4xmpExt:rId>
          </rdf:li>
        </rdf:Bag>
      </Iptc4xmpExt:ImageRegion>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
                person_names=["New Name"],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [
                        {
                            "name": "New Name",
                            "score": 0.98,
                            "certainty": 0.98,
                            "bbox": [20, 10, 40, 30],
                        }
                    ],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                image_width=200,
                image_height=100,
            )

            xml = out.read_text(encoding="utf-8")
            self.assertNotIn("Old Name", xml)
            self.assertIn("New Name", xml)
            self.assertEqual(_image_region_ids(out), ["photo-1", "face-1"])

    def test_write_xmp_sidecar_removes_stale_face_regions_without_touching_non_face_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Egypt_1975_B00_P09_V.xmp"
            out.write_text(
                f"""<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="{xmp_sidecar.X_NS}" xmlns:rdf="{xmp_sidecar.RDF_NS}" xmlns:Iptc4xmpExt="{xmp_sidecar.IPTC_EXT_NS}">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <Iptc4xmpExt:ImageRegion>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>face-1</Iptc4xmpExt:rId>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>photo-1</Iptc4xmpExt:rId>
          </rdf:li>
        </rdf:Bag>
      </Iptc4xmpExt:ImageRegion>
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
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                image_width=200,
                image_height=100,
            )

            self.assertEqual(_image_region_ids(out), ["photo-1"])


if __name__ == "__main__":
    unittest.main()
