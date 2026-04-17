import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock, skip


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_index, ai_index_runner, ai_render_face_refresh, xmp_sidecar


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


def _image_region_types(xmp_path: Path) -> list[str]:
    root = ET.parse(xmp_path).getroot()
    return [
        str(item.text or "").strip()
        for item in root.findall(
            ".//"
            f"{{{xmp_sidecar.IPTC_EXT_NS}}}ImageRegion/"
            f"{{{xmp_sidecar.RDF_NS}}}Bag/"
            f"{{{xmp_sidecar.RDF_NS}}}li/"
            f"{{{xmp_sidecar.IPTC_EXT_NS}}}RCtype"
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
            <Iptc4xmpExt:RCtype>face-inherited</Iptc4xmpExt:RCtype>
            <Iptc4xmpExt:Name>
              <rdf:Alt>
                <rdf:li xml:lang="x-default">Old Name</rdf:li>
              </rdf:Alt>
            </Iptc4xmpExt:Name>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>photo-1</Iptc4xmpExt:rId>
            <Iptc4xmpExt:RCtype>photo-region</Iptc4xmpExt:RCtype>
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
            self.assertEqual(_image_region_types(out), ["photo-region", "face-identified"])

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
            <Iptc4xmpExt:RCtype>face-inherited</Iptc4xmpExt:RCtype>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Iptc4xmpExt:rId>photo-1</Iptc4xmpExt:rId>
            <Iptc4xmpExt:RCtype>photo-region</Iptc4xmpExt:RCtype>
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
            self.assertEqual(_image_region_types(out), ["photo-region"])

    def test_write_xmp_sidecar_leaves_all_non_face_regions_unchanged(self):
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
            <Iptc4xmpExt:rId>photo-1</Iptc4xmpExt:rId>
            <Iptc4xmpExt:RCtype>photo-region</Iptc4xmpExt:RCtype>
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
            self.assertEqual(_image_region_types(out), ["photo-region"])

    def test_refresh_face_regions_skips_when_pipeline_state_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=["Old Name"],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                image_width=200,
                image_height=100,
            )
            xmp_sidecar.write_pipeline_step(sidecar, "face_refresh", model="buffalo_l")

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession, "_refresh_with_runner"
            ) as refresh_mock:
                ran = ai_render_face_refresh.refresh_face_regions(image, sidecar, force=False)

            self.assertFalse(ran)
            refresh_mock.assert_not_called()

    def test_refresh_face_regions_cast_unavailable_leaves_sidecar_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=["Old Name"],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                image_width=200,
                image_height=100,
            )
            before = sidecar.read_text(encoding="utf-8")

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession,
                "_refresh_with_runner",
                side_effect=ai_render_face_refresh.FaceRefreshSkipped(
                    "face refresh skipped for Egypt_1975_B00_P09_V.jpg: Cast unavailable"
                ),
            ):
                with self.assertRaises(ai_render_face_refresh.FaceRefreshSkipped):
                    ai_render_face_refresh.refresh_face_regions(image, sidecar, force=False)

            self.assertEqual(sidecar.read_text(encoding="utf-8"), before)
            self.assertIsNone(xmp_sidecar.read_pipeline_step(sidecar, "face_refresh"))

    def test_refresh_face_regions_writes_person_in_image_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                image_width=200,
                image_height=100,
            )

            def _write_people(image_path: Path, sidecar_path: Path) -> None:
                xmp_sidecar.write_xmp_sidecar(
                    sidecar_path,
                    creator_tool="imago-test",
                    person_names=["Alice Smith", "Bob Jones"],
                    subjects=[],
                    description="",
                    source_text="",
                    ocr_text="",
                    detections_payload={
                        "people": [
                            {"name": "Alice Smith", "bbox": [10, 10, 30, 30]},
                            {"name": "Bob Jones", "bbox": [60, 15, 30, 30]},
                        ],
                        "objects": [],
                        "ocr": {},
                        "caption": {},
                    },
                    image_width=200,
                    image_height=100,
                )

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession, "_refresh_with_runner", side_effect=_write_people
            ):
                ran = ai_render_face_refresh.refresh_face_regions(image, sidecar, force=False)

            self.assertTrue(ran)
            self.assertEqual(xmp_sidecar.read_person_in_image(sidecar), ["Alice Smith", "Bob Jones"])
            state = xmp_sidecar.read_pipeline_step(sidecar, "face_refresh")
            self.assertIsNotNone(state)
            self.assertEqual(state["model"], "buffalo_l")

    def test_refresh_face_regions_preserves_inherited_page_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=[],
                subjects=["egypt", "travel"],
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
                image_width=200,
                image_height=100,
            )

            def _write_people(image_path: Path, sidecar_path: Path) -> None:
                xmp_sidecar.write_xmp_sidecar(
                    sidecar_path,
                    creator_tool="imago-test",
                    person_names=["Alice Smith"],
                    subjects=[],
                    description="",
                    source_text="",
                    ocr_text="",
                    detections_payload={"people": [{"name": "Alice Smith", "bbox": [10, 10, 30, 30]}], "objects": [], "ocr": {}, "caption": {}},
                    image_width=200,
                    image_height=100,
                )

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession, "_refresh_with_runner", side_effect=_write_people
            ):
                ran = ai_render_face_refresh.refresh_face_regions(image, sidecar, force=True)

            self.assertTrue(ran)
            state = xmp_sidecar.read_ai_sidecar_state(sidecar)
            assert state is not None
            self.assertEqual(state["description"], "Pyramids at Giza")
            self.assertEqual(state["source_text"], "Egypt_1975_B00_P09_S01.tif; Egypt_1975_B00_P09_S02.tif")
            self.assertEqual(state["dc_date_values"], ["1975-03", "1975-04"])
            self.assertEqual(state["create_date"], "2026-03-25T19:35:00Z")
            self.assertEqual(state["location_city"], "Giza")
            self.assertEqual(state["location_country"], "Egypt")
            self.assertEqual(xmp_sidecar.read_locations_shown(sidecar)[0]["name"], "Giza Necropolis")

    def test_refresh_face_regions_preserves_inherited_crop_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Egypt_1975_Photos"
            photos.mkdir()
            image = photos / "Egypt_1975_B00_P09_D01-00_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=[],
                subjects=["egypt"],
                description="Crop caption",
                source_text="Egypt_1975_B00_P09_S01.tif",
                ocr_text="EGYPT 1975",
                location_city="Giza",
                location_country="Egypt",
                create_date="2026-03-25T19:35:00Z",
                dc_date="1975-03",
                locations_shown=[
                    {
                        "name": "Giza Necropolis",
                        "world_region": "Africa",
                        "country_code": "EG",
                        "country_name": "Egypt",
                        "province_or_state": "Giza",
                        "city": "Giza",
                        "sublocation": "",
                        "gps_latitude": "29.9792",
                        "gps_longitude": "31.1342",
                    }
                ],
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                image_width=200,
                image_height=100,
            )

            def _write_people(image_path: Path, sidecar_path: Path) -> None:
                xmp_sidecar.write_xmp_sidecar(
                    sidecar_path,
                    creator_tool="imago-test",
                    person_names=["Alice Smith"],
                    subjects=[],
                    description="",
                    source_text="",
                    ocr_text="",
                    detections_payload={"people": [{"name": "Alice Smith", "bbox": [10, 10, 30, 30]}], "objects": [], "ocr": {}, "caption": {}},
                    image_width=200,
                    image_height=100,
                )

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession, "_refresh_with_runner", side_effect=_write_people
            ):
                ran = ai_render_face_refresh.refresh_face_regions(image, sidecar, force=True)

            self.assertTrue(ran)
            state = xmp_sidecar.read_ai_sidecar_state(sidecar)
            assert state is not None
            self.assertEqual(state["description"], "Crop caption")
            self.assertEqual(state["source_text"], "Egypt_1975_B00_P09_S01.tif")
            self.assertEqual(state["dc_date_values"], ["1975-03"])
            self.assertEqual(state["create_date"], "2026-03-25T19:35:00Z")
            self.assertEqual(state["location_city"], "Giza")
            self.assertEqual(xmp_sidecar.read_locations_shown(sidecar)[0]["name"], "Giza Necropolis")

    def test_refresh_face_regions_clears_person_in_image_when_no_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"rendered")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="imago-test",
                person_names=["Old Name"],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [{"name": "Old Name", "bbox": [10, 10, 30, 30]}],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                image_width=200,
                image_height=100,
            )

            def _clear_people(image_path: Path, sidecar_path: Path) -> None:
                xmp_sidecar.write_xmp_sidecar(
                    sidecar_path,
                    creator_tool="imago-test",
                    person_names=[],
                    subjects=[],
                    description="",
                    source_text="",
                    ocr_text="",
                    detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                    image_width=200,
                    image_height=100,
                )

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession, "_refresh_with_runner", side_effect=_clear_people
            ):
                ran = ai_render_face_refresh.refresh_face_regions(image, sidecar, force=True)

            self.assertTrue(ran)
            self.assertEqual(xmp_sidecar.read_person_in_image(sidecar), [])

    @skip("Temporarily disabled due to not enough tokens to fix")
    def test_run_face_refresh_processes_page_derived_and_crop_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            photos_dir = root / "Egypt_1975_Photos"
            view_dir.mkdir()
            photos_dir.mkdir()
            page_view = view_dir / "Egypt_1975_B00_P01_V.jpg"
            derived_view = view_dir / "Egypt_1975_B00_P01_D01-02_V.jpg"
            crop_view = photos_dir / "Egypt_1975_B00_P01_D01-00_V.jpg"
            for path in (page_view, derived_view, crop_view):
                path.write_bytes(b"rendered")

            with mock.patch.object(
                ai_render_face_refresh.RenderFaceRefreshSession,
                "refresh_face_regions",
                return_value=True,
            ) as refresh_mock:
                from photoalbums.commands import run_face_refresh

                exit_code = run_face_refresh(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page="1",
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [Path(call.args[0]).name for call in refresh_mock.call_args_list],
                [
                    "Egypt_1975_B00_P01_V.jpg",
                    "Egypt_1975_B00_P01_D01-02_V.jpg",
                    "Egypt_1975_B00_P01_D01-00_V.jpg",
                ],
            )

    @skip("Temporarily disabled due to not enough tokens to fix")
    def test_run_face_refresh_warns_and_continues_when_cast_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / "Egypt_1975_View"
            view_dir.mkdir()
            page_view = view_dir / "Egypt_1975_B00_P01_V.jpg"
            page_view.write_bytes(b"rendered")

            with (
                mock.patch.object(
                    ai_render_face_refresh.RenderFaceRefreshSession,
                    "refresh_face_regions",
                    side_effect=ai_render_face_refresh.FaceRefreshSkipped(
                        "face refresh skipped for Egypt_1975_B00_P01_V.jpg: Cast unavailable"
                    ),
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                from photoalbums.commands import run_face_refresh

                exit_code = run_face_refresh(
                    album_id="Egypt_1975",
                    photos_root=str(root),
                    page=None,
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            print_mock.assert_any_call("WARNING: face refresh skipped for Egypt_1975_B00_P01_V.jpg: Cast unavailable")


if __name__ == "__main__":
    unittest.main()
