"""Tests for propagate-to-crops step (Task 4.4)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_index_propagate
from photoalbums.lib.ai_index_propagate import _crop_paths_signature, run_propagate_to_crops
from photoalbums.lib import xmp_sidecar


class _FakeGeocodeResult:
    def __init__(self, query: str, latitude: str, longitude: str, city: str, country: str):
        self.query = query
        self.latitude = latitude
        self.longitude = longitude
        self.display_name = f"{city}, {country}"
        self.source = "nominatim"
        self.city = city
        self.state = ""
        self.country = country
        self.sublocation = ""


class _FakeGeocoder:
    def __init__(self, result: _FakeGeocodeResult):
        self.result = result

    def geocode(self, query: str):
        return self.result


def _write_basic_crop_xmp(path: Path, person_names: list[str] | None = None) -> None:
    xmp_sidecar.write_xmp_sidecar(
        path,
        person_names=person_names or [],
        subjects=[],
        description="Crop photo",
        source_text="",
        ocr_text="",
    )


class TestCropPathsSignature(unittest.TestCase):
    def test_empty_list_returns_consistent_hash(self):
        sig = _crop_paths_signature([])
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 16)

    def test_same_paths_same_signature(self):
        paths = [Path("/a/b.jpg"), Path("/a/c.jpg")]
        self.assertEqual(_crop_paths_signature(paths), _crop_paths_signature(paths))

    def test_order_independent(self):
        paths_a = [Path("/a/b.jpg"), Path("/a/c.jpg")]
        paths_b = [Path("/a/c.jpg"), Path("/a/b.jpg")]
        self.assertEqual(_crop_paths_signature(paths_a), _crop_paths_signature(paths_b))

    def test_different_paths_different_signature(self):
        paths_a = [Path("/a/b.jpg")]
        paths_b = [Path("/a/c.jpg")]
        self.assertNotEqual(_crop_paths_signature(paths_a), _crop_paths_signature(paths_b))


class TestRunPropagateTocrops(unittest.TestCase):
    def _setup_page_image(self, tmp_dir: Path) -> tuple[Path, Path, Path]:
        """Create a pages dir + photos dir with basic structure."""
        pages_dir = tmp_dir / "Family_2020_B01_Pages"
        photos_dir = tmp_dir / "Family_2020_B01_Photos"
        pages_dir.mkdir()
        photos_dir.mkdir()
        image = pages_dir / "Family_2020_B01_P02_V.jpg"
        image.write_bytes(b"fake-jpeg")
        return image, pages_dir, photos_dir

    def _write_page_xmp_with_regions(self, image: Path, region_names: list[str]) -> Path:
        """Write page XMP with MWG-RS face regions using real region names."""
        xmp_path = image.with_suffix(".xmp")
        from photoalbums.lib.xmp_sidecar import write_region_list
        from photoalbums.lib.ai_view_regions import RegionWithCaption, RegionResult
        regions = [
            RegionWithCaption(
                RegionResult(index=i, x=i * 100, y=0, width=100, height=100),
                "",
            )
            for i in range(len(region_names))
        ]
        # Write basic XMP then add regions
        xmp_sidecar.write_xmp_sidecar(
            xmp_path,
            person_names=region_names,
            subjects=[],
            description="Page photo",
            source_text="",
            ocr_text="",
        )
        write_region_list(xmp_path, regions, 800, 600)
        return xmp_path

    def test_no_crops_when_not_pages_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Use an Archive dir (not Pages)
            archive_dir = tmp_dir / "Family_2020_B01_Archive"
            archive_dir.mkdir()
            image = archive_dir / "Family_2020_B01_P02_S01.tif"
            image.write_bytes(b"scan")
            image.with_suffix(".xmp").write_text("<x:xmpmeta/>", encoding="utf-8")

            result = run_propagate_to_crops(image, location_payload={}, people_payload=[])
            self.assertEqual(result["crops_updated"], 0)

    def test_page_with_no_crops_records_zero_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, pages_dir, photos_dir = self._setup_page_image(tmp_dir)

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[]):
                result = run_propagate_to_crops(image, location_payload={}, people_payload=[])
            self.assertEqual(result["crops_updated"], 0)

    def test_page_scalar_location_not_written_to_crops_when_location_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, pages_dir, photos_dir = self._setup_page_image(tmp_dir)

            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop2 = photos_dir / "Family_2020_B01_P02_D02-00_V.jpg"
            crop1.write_bytes(b"crop1")
            crop2.write_bytes(b"crop2")
            _write_basic_crop_xmp(crop1.with_suffix(".xmp"))
            _write_basic_crop_xmp(crop2.with_suffix(".xmp"))

            location_payload = {
                "gps_latitude": "48.8566",
                "gps_longitude": "2.3522",
                "city": "Paris",
                "country": "France",
            }

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1, crop2]):
                result = run_propagate_to_crops(image, location_payload=location_payload, people_payload=[])

            self.assertEqual(result["crops_updated"], 2)

            state1 = xmp_sidecar.read_ai_sidecar_state(crop1.with_suffix(".xmp"))
            assert state1 is not None
            self.assertEqual(str(state1.get("gps_latitude") or ""), "")
            self.assertEqual(str(state1.get("gps_longitude") or ""), "")

            state2 = xmp_sidecar.read_ai_sidecar_state(crop2.with_suffix(".xmp"))
            assert state2 is not None
            self.assertEqual(str(state2.get("gps_latitude") or ""), "")

    def test_pipeline_record_written_to_each_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, pages_dir, photos_dir = self._setup_page_image(tmp_dir)

            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            _write_basic_crop_xmp(crop1.with_suffix(".xmp"))

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]):
                run_propagate_to_crops(image, location_payload={}, people_payload=[])

            step = xmp_sidecar.read_pipeline_step(crop1.with_suffix(".xmp"), "ai-index/propagate-to-crops")
            assert step is not None
            self.assertEqual(step["result"], "ok")
            self.assertIn("timestamp", step)

    def test_region_caption_fills_empty_crop_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, _pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            xmp_sidecar.write_xmp_sidecar(
                crop1.with_suffix(".xmp"),
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
            )

            xmp_path = image.with_suffix(".xmp")
            from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
            from photoalbums.lib.xmp_sidecar import write_region_list, write_xmp_sidecar

            write_xmp_sidecar(
                xmp_path,
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
            )
            write_region_list(
                xmp_path,
                [
                    RegionWithCaption(
                        RegionResult(index=0, x=0, y=0, width=100, height=100, caption_hint="stale hint"),
                        "OXFORD STREET, LONDON, ENGLAND - AUG. 1988",
                    )
                ],
                800,
                600,
            )

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]):
                run_propagate_to_crops(image, location_payload={}, people_payload=[])

            state = xmp_sidecar.read_ai_sidecar_state(crop1.with_suffix(".xmp"))
            assert state is not None
            self.assertEqual(state["description"], "OXFORD STREET, LONDON, ENGLAND - AUG. 1988")

    def test_region_location_and_person_filter_are_resolved_per_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, _pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            _write_basic_crop_xmp(crop1.with_suffix(".xmp"))

            xmp_path = image.with_suffix(".xmp")
            from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
            from photoalbums.lib.xmp_sidecar import write_region_list, write_xmp_sidecar

            write_xmp_sidecar(
                xmp_path,
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
                locations_shown=[{"name": "Karnten, Austria", "city": "Karnten", "country_name": "Austria"}],
            )
            write_region_list(
                xmp_path,
                [
                    RegionWithCaption(
                        RegionResult(
                            index=0,
                            x=0,
                            y=0,
                            width=100,
                            height=100,
                            caption_hint="KARNTEN, AUSTRIA",
                            person_names=["KARNTEN, AUSTRIA", "Audrey Cordell"],
                        ),
                        "",
                    )
                ],
                800,
                600,
            )

            fake_geocoder = _FakeGeocoder(
                _FakeGeocodeResult("Karnten, Austria", "46.75", "13.8333333", "Karnten", "Austria")
            )
            with (
                mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]),
                mock.patch("photoalbums.lib.ai_geocode.NominatimGeocoder", return_value=fake_geocoder),
            ):
                run_propagate_to_crops(
                    image,
                    location_payload={"city": "Vienna", "country": "Austria"},
                    people_payload=[],
                )

            state = xmp_sidecar.read_ai_sidecar_state(crop1.with_suffix(".xmp"))
            assert state is not None
            self.assertEqual(state["location_city"], "Karnten")
            xml = crop1.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertNotIn("photoshop:City", xml)
            self.assertNotIn("KARNTEN, AUSTRIA", xmp_sidecar.read_person_in_image(crop1.with_suffix(".xmp")))
            crop_locations_shown = xmp_sidecar.read_locations_shown(crop1.with_suffix(".xmp"))
            self.assertEqual(crop_locations_shown[0]["name"], "Karnten, Austria")
            self.assertEqual(crop_locations_shown[0]["city"], "Karnten")
            self.assertEqual(crop_locations_shown[0]["country_name"], "Austria")
            self.assertTrue(crop_locations_shown[0]["gps_latitude"])
            self.assertTrue(crop_locations_shown[0]["gps_longitude"])

    def test_single_location_shown_is_kept_on_crop_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, _pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")

            xmp_path = image.with_suffix(".xmp")
            locations_shown = [
                {
                    "name": "Hassan II Mosque",
                    "city": "Casablanca",
                    "country_name": "Morocco",
                    "sublocation": "Boulevard de la Corniche",
                }
            ]
            xmp_sidecar.write_xmp_sidecar(
                xmp_path,
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
                locations_shown=locations_shown,
            )
            xmp_sidecar.write_xmp_sidecar(
                crop1.with_suffix(".xmp"),
                person_names=[],
                subjects=[],
                description="Crop photo",
                source_text="",
                ocr_text="",
                locations_shown=locations_shown,
            )

            fake_geocoder = _FakeGeocoder(
                _FakeGeocodeResult(
                    "Hassan II Mosque, Casablanca, Morocco",
                    "33.6082243",
                    "-7.6324728",
                    "Casablanca",
                    "Morocco",
                )
            )
            with (
                mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]),
                mock.patch("photoalbums.lib.ai_geocode.NominatimGeocoder", return_value=fake_geocoder),
            ):
                run_propagate_to_crops(
                    image,
                    location_payload={"city": "Casablanca", "country": "Morocco"},
                    people_payload=[],
                )

            crop_locations_shown = xmp_sidecar.read_locations_shown(crop1.with_suffix(".xmp"))
            self.assertEqual(crop_locations_shown[0]["name"], "Hassan II Mosque")
            self.assertEqual(crop_locations_shown[0]["city"], "Casablanca")
            self.assertEqual(crop_locations_shown[0]["country_name"], "Morocco")
            self.assertTrue(crop_locations_shown[0]["gps_latitude"])
            self.assertTrue(crop_locations_shown[0]["gps_longitude"])

    def test_region_location_payload_writes_crop_specific_locations_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, _pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            _write_basic_crop_xmp(crop1.with_suffix(".xmp"))

            xmp_path = image.with_suffix(".xmp")
            from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
            from photoalbums.lib.xmp_sidecar import write_region_list, write_xmp_sidecar

            write_xmp_sidecar(
                xmp_path,
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
            )
            write_region_list(
                xmp_path,
                [
                    RegionWithCaption(
                        RegionResult(
                            index=0,
                            x=0,
                            y=0,
                            width=100,
                            height=100,
                            caption_hint="Temple visit",
                            location_payload={
                                "address": "Karnak Temple",
                                "city": "Luxor",
                                "state": "Luxor",
                                "country": "Egypt",
                                "sublocation": "Karnak",
                                "gps_latitude": "25.7188",
                                "gps_longitude": "32.6573",
                            },
                        ),
                        "",
                    )
                ],
                800,
                600,
            )

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]):
                run_propagate_to_crops(
                    image,
                    location_payload={"city": "Cairo", "country": "Egypt"},
                    people_payload=[],
                )

            self.assertEqual(
                xmp_sidecar.read_locations_shown(crop1.with_suffix(".xmp")),
                [
                    {
                        "name": "Karnak Temple",
                        "world_region": "",
                        "country_code": "",
                        "country_name": "Egypt",
                        "province_or_state": "Luxor",
                        "city": "Luxor",
                        "sublocation": "Karnak",
                        "gps_latitude": "25.7188",
                        "gps_longitude": "32.6573",
                    }
                ],
            )
            xml = crop1.with_suffix(".xmp").read_text(encoding="utf-8")
            self.assertNotIn("photoshop:City", xml)

    def test_page_dc_date_propagated_to_crops_without_existing_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            _write_basic_crop_xmp(crop1.with_suffix(".xmp"))

            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
                dc_date=["1988-08"],
            )

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]):
                run_propagate_to_crops(image, location_payload={}, people_payload=[])

            state = xmp_sidecar.read_ai_sidecar_state(crop1.with_suffix(".xmp"))
            assert state is not None
            self.assertEqual(state.get("dc_date_values"), ["1988-08"])

    def test_crop_own_date_takes_precedence_over_page_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image, pages_dir, photos_dir = self._setup_page_image(tmp_dir)
            crop1 = photos_dir / "Family_2020_B01_P02_D01-00_V.jpg"
            crop1.write_bytes(b"crop1")
            xmp_sidecar.write_xmp_sidecar(
                crop1.with_suffix(".xmp"),
                person_names=[],
                subjects=[],
                description="Crop photo",
                source_text="",
                ocr_text="",
                dc_date=["1988-08-15"],
            )

            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                person_names=[],
                subjects=[],
                description="Page photo",
                source_text="",
                ocr_text="",
                dc_date=["1988-08"],
            )

            with mock.patch.object(ai_index_propagate, "_find_crop_paths_for_page", return_value=[crop1]):
                run_propagate_to_crops(image, location_payload={}, people_payload=[])

            state = xmp_sidecar.read_ai_sidecar_state(crop1.with_suffix(".xmp"))
            assert state is not None
            self.assertEqual(state.get("dc_date_values"), ["1988-08-15"])

    def test_step_skipped_when_neither_upstream_reran(self):
        """StepRunner skips propagate-to-crops when hashes match for all inputs."""
        from photoalbums.lib.ai_index_steps import StepRunner, propagate_to_crops_input_hash

        settings = {
            "crop_paths_signature": "abc",
            "cast_store_signature": "sig1",
            "caption_engine": "lmstudio",
            "caption_model": "model",
            "ocr_engine": "local",
            "ocr_model": "ocr-m",
            "ocr_lang": "eng",
            "scan_group_signature": "",
            "nominatim_base_url": "",
            "model": "yolo",
            "enable_objects": True,
        }
        # Compute expected hash with empty output hashes (nothing ran upstream)
        base = StepRunner(settings=settings, existing_pipeline_state={}, existing_detections={}, forced_steps=set())
        expected_hash = propagate_to_crops_input_hash(settings, base.output_hashes)

        pipeline_state = {
            "ai-index/propagate-to-crops": {
                "timestamp": "2026-04-11T00:00:00Z",
                "result": "ok",
                "input_hash": expected_hash,
            }
        }
        runner = StepRunner(
            settings=settings,
            existing_pipeline_state=pipeline_state,
            existing_detections={},
            forced_steps=set(),
        )
        called = [False]

        def do_propagate():
            called[0] = True
            return {"crops_updated": 0}

        runner.run("propagate-to-crops", do_propagate)
        self.assertFalse(called[0], "propagate-to-crops must be skipped when hash matches")
        self.assertFalse(runner.reran.get("propagate-to-crops", True))


if __name__ == "__main__":
    unittest.main()
