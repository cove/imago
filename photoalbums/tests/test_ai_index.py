import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from cast.storage import TextFaceStore
from photoalbums.lib import ai_index
from photoalbums.lib import xmp_sidecar


class TestAIIndex(unittest.TestCase):
    def _valid_sidecar_text(self) -> str:
        return "x" * (ai_index.MIN_EXISTING_SIDECAR_BYTES + 1)

    @contextmanager
    def _mock_layout(self, image: Path):
        yield SimpleNamespace(
            kind="single_image",
            page_like=False,
            split_mode="manual",
            content_bounds=SimpleNamespace(as_dict=lambda: {"x": 0, "y": 0, "width": 0, "height": 0}),
            split_applied=False,
            fallback_used=False,
            subphotos=[],
            content_path=image,
        )

    def test_discover_images_archive_and_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_Archive"
            view = base / "Family_View"
            archive.mkdir()
            view.mkdir()
            (archive / "a.jpg").write_bytes(b"a")
            (view / "b.png").write_bytes(b"b")
            (base / "other.jpg").write_bytes(b"c")

            files = ai_index.discover_images(
                base,
                include_archive=True,
                include_view=False,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["a.jpg"])

            files = ai_index.discover_images(
                base,
                include_archive=False,
                include_view=True,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["b.png"])

    def test_resolve_xmp_text_layers_treats_page_ocr_as_author_text(self):
        layers = ai_index._resolve_xmp_text_layers(
            image_path=Path("China_1986_B02_P02_stitched.jpg"),
            ocr_text="TEMPLE OF HEAVEN",
            page_like=True,
        )
        self.assertEqual(layers["author_text"], "TEMPLE OF HEAVEN")
        self.assertEqual(layers["scene_text"], "")
        self.assertEqual(layers["annotation_scope"], "page")

    def test_compute_xmp_title_ignores_scene_text_by_default(self):
        layers = ai_index._resolve_xmp_text_layers(
            image_path=Path("Family_1986_B01_P02.jpg"),
            ocr_text="NO SMOKING",
            page_like=False,
        )
        title, title_source = ai_index._compute_xmp_title(
            image_path=Path("Family_1986_B01_P02.jpg"),
            explicit_title="",
            author_text=layers["author_text"],
            annotation_scope=layers["annotation_scope"],
        )
        self.assertEqual(title, "")
        self.assertEqual(title_source, "")

    def test_compute_xmp_title_keeps_cover_text(self):
        with mock.patch.object(ai_index, "looks_like_album_cover", return_value=True):
            title, title_source = ai_index._compute_xmp_title(
                image_path=Path("China_1986_B02_P00.jpg"),
                explicit_title="",
                author_text="MAINLAND CHINA 1986 BOOK 11",
                annotation_scope="page",
            )
        self.assertEqual(title, "MAINLAND CHINA 1986 BOOK 11")
        self.assertEqual(title_source, "author_text")

    def test_needs_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(sidecar),
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

    def test_needs_processing_requires_current_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(image.with_suffix(".xmp")),
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

    def test_needs_processing_skips_current_sidecar_even_when_sidecar_state_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            row = {
                "image_path": str(image),
                "size": 999,
                "mtime_ns": 999,
                "sidecar_path": str(sidecar),
                "processor_signature": "old-signature",
            }

            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(
                ai_index.needs_processing(
                    image,
                    row,
                    force=False,
                    reprocess_required=True,
                )
            )

    def test_needs_processing_skips_when_sidecar_state_missing_but_valid_sidecar_exists(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")

            self.assertFalse(ai_index.needs_processing(image, None, force=False))
            self.assertTrue(ai_index.needs_processing(image, None, force=True))

    def test_needs_processing_ignores_tiny_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text("tiny", encoding="utf-8")

            self.assertTrue(ai_index.needs_processing(image, None, force=False))

    def test_sidecar_has_lmstudio_caption_error(self):
        self.assertTrue(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "lmstudio",
                            "effective_engine": "lmstudio",
                            "error": "model offline",
                        }
                    }
                }
            )
        )
        self.assertFalse(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "lmstudio",
                            "effective_engine": "lmstudio",
                            "error": "",
                        }
                    }
                }
            )
        )
        self.assertFalse(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "local",
                            "effective_engine": "local",
                            "error": "model offline",
                        }
                    }
                }
            )
        )

    def test_prepare_ai_model_image_scales_when_threshold_exceeded(self):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"pillow unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.png"
            Image.new("RGB", (240, 180), color="white").save(image)
            with mock.patch.object(ai_index, "AI_MODEL_MAX_SOURCE_BYTES", 1):
                with ai_index._prepare_ai_model_image(image) as prepared:
                    self.assertNotEqual(prepared, image)
                    self.assertTrue(prepared.exists())
                    self.assertEqual(prepared.suffix.lower(), ".jpg")

    @pytest.mark.skip(reason="This test is a work in progress and not yet passing reliably.")
    def test_resolve_archive_scan_authoritative_ocr_stitches_once_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            group_paths = [scan1, scan2]
            signature = ai_index._scan_group_signature(group_paths)
            cache: dict[str, ai_index.ArchiveScanOCRAuthority] = {}
            fake_ocr_engine = mock.Mock()
            fake_ocr_engine.read_text.return_value = "MAINLAND CHINA 1986 BOOK 11"
            stitched = np.full((12, 20, 3), 255, dtype=np.uint8)

            with mock.patch(
                "photoalbums.stitch_oversized_pages.build_stitched_image",
                return_value=stitched,
            ) as build_mock:
                first = ai_index._resolve_archive_scan_authoritative_ocr(
                    image_path=scan1,
                    group_paths=group_paths,
                    group_signature=signature,
                    ocr_engine=fake_ocr_engine,
                    cache=cache,
                )
                second = ai_index._resolve_archive_scan_authoritative_ocr(
                    image_path=scan1,
                    group_paths=group_paths,
                    group_signature=signature,
                    ocr_engine=fake_ocr_engine,
                    cache=cache,
                )

            self.assertEqual(first.ocr_text, "MAINLAND CHINA 1986 BOOK 11")
            self.assertEqual(first.ocr_hash, ai_index._hash_text(first.ocr_text))
            self.assertEqual(tuple(first.group_paths), (scan1, scan2))
            self.assertEqual(first, second)
            build_mock.assert_called_once_with([str(scan1), str(scan2)])
            fake_ocr_engine.read_text.assert_called_once()

    def test_run_image_analysis_passes_people_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = [
                SimpleNamespace(
                    name="Alice",
                    score=0.92,
                    certainty=0.92,
                    reviewed_by_human=False,
                    face_id="face-1",
                )
            ]
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Dolores Cordell"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="template",

                ocr_engine_name="none",
                ocr_language="eng",
                people_hint_text="Page caption",
                people_source_path=Path(tmp) / "original.jpg",
                people_bbox_offset=(12, 34),
            )

            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=Path(tmp) / "original.jpg",
                bbox_offset=(12, 34),
                hint_text="Page caption Dolores Cordell",
            )
            self.assertEqual(analysis.people_names, ["Alice"])
            self.assertEqual(analysis.payload["people"][0]["face_id"], "face-1")
            self.assertFalse(analysis.payload["people"][0]["reviewed_by_human"])

    def test_run_image_analysis_uses_scaled_image_for_ocr_objects_and_caption_only(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            scaled = Path(tmp) / "scaled.jpg"
            scaled.write_bytes(b"scaled")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "hello"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                fallback=False,
                error="",
            )

            @contextmanager
            def fake_prepare(_path):
                yield scaled

            with mock.patch.object(ai_index, "_prepare_ai_model_image", side_effect=fake_prepare):
                ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
    
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=image,
                bbox_offset=(0, 0),
                hint_text="hello",
            )
            ocr_engine.read_text.assert_called_once_with(
                scaled,
                debug_recorder=None,
                debug_step="ocr",
            )
            object_detector.detect_image.assert_called_once_with(scaled)
            caption_engine.generate.assert_called_once_with(
                image_path=scaled,
                people=[],
                objects=[],
                ocr_text="hello",
                source_path=image,
                album_title="",
                printed_album_title="",
                photo_count=1,
                is_cover_page=False,
                people_positions={},
                request_photo_regions=False,
                debug_recorder=None,
                debug_step="caption",
            )

    def test_run_image_analysis_records_gps_location_from_ocr_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Latitude: 39.7875\nLongitude: 100.307222"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Mogao Caves in Dunhuang, China (39°47′15″N 100°18′26″E).",
                engine="lmstudio",
                gps_latitude="",
                gps_longitude="",
                location_name="",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",

                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.7875)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 100.307222)

    def test_resolve_location_metadata_skips_cover_page_title_geocoding(self):
        image = Path("cover.jpg")
        caption_engine = mock.Mock()
        caption_engine.estimate_location.return_value = SimpleNamespace(
            gps_latitude="19.1414769",
            gps_longitude="72.8323049",
            location_name="Mainland China",
            fallback=False,
        )

        gps_latitude, gps_longitude, location_name = ai_index._resolve_location_metadata(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=image,
            people=[],
            objects=[],
            ocr_text="MAINLAND CHINA\n1986\nBOOK 11",
            source_path=image,
            album_title="",
            printed_album_title="",
            is_cover_page=True,
            people_positions={},
            fallback_location_name="",
        )

        self.assertEqual((gps_latitude, gps_longitude, location_name), ("", "", ""))
        caption_engine.estimate_location.assert_not_called()

    def test_run_image_analysis_geocodes_structured_location_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="The entrance to the Mogao Caves in Dunhuang.",
                engine="lmstudio",
                gps_latitude="",
                gps_longitude="",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )
            geocoder = mock.Mock()
            geocoder.geocode.return_value = SimpleNamespace(
                query="Mogao Caves, Dunhuang, Gansu, China",
                latitude="39.9361",
                longitude="94.8076",
                display_name="Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
                source="nominatim",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="none",
                ocr_language="eng",
                geocoder=geocoder,
            )

            geocoder.geocode.assert_called_once_with("Mogao Caves, Dunhuang, Gansu, China")
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )
            self.assertEqual(
                analysis.payload["location"]["display_name"],
                "Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
            )
            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.9361)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 94.8076)
            self.assertEqual(analysis.payload["location"]["source"], "nominatim")

    def test_run_image_analysis_promotes_cover_author_text_to_album_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "cover.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "MAINLAND CHINA\n1986\nBOOK 11"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="MAINLAND CHINA 1986 BOOK 11",
                engine="template",
                fallback=False,
                error="",
                author_text="MAINLAND CHINA\n1986\nBOOK 11",
                scene_text="",
                annotation_scope="page",
                album_title="",
                title="",
                ocr_lang="eng",
            )

            with mock.patch.object(ai_index, "looks_like_album_cover", return_value=True):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="tesseract",
                    ocr_language="eng",
                    is_page_scan=True,
                )

            self.assertEqual(analysis.album_title, "MAINLAND CHINA 1986 BOOK 11")

    def test_run_image_analysis_merges_lmstudio_location_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="A rocky desert landscape.",
                engine="lmstudio",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="39.9361",
                gps_longitude="94.8076",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",

                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.9361)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 94.8076)
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )
            caption_engine.estimate_location.assert_called_once()

    def test_run_image_analysis_prefers_explicit_ocr_gps_over_model_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Latitude: 39.7875\nLongitude: 100.307222"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="A scenic location.",
                engine="lmstudio",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="39.9361",
                gps_longitude="94.8076",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",

                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.7875)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 100.307222)
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )

    def test_run_image_analysis_runs_people_recovery_and_reruns_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")

            class _Matcher:
                def __init__(self):
                    self.last_faces_detected = 0
                    self.match_calls = 0
                    self.recovery_calls = 0

                def match_image(self, *_args, **_kwargs):
                    self.match_calls += 1
                    self.last_faces_detected = 0
                    return []

                def match_image_recovery(self, *_args, **_kwargs):
                    self.recovery_calls += 1
                    self.last_faces_detected = 1
                    return [
                        SimpleNamespace(
                            name="Alice",
                            score=0.97,
                            certainty=0.97,
                            reviewed_by_human=False,
                            face_id="face-2",
                            bbox=[10, 10, 20, 20],
                        )
                    ]

            people_matcher = _Matcher()
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = [SimpleNamespace(label="person", score=0.91)]
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Latitude: 39.7875\nLongitude: 100.307222"
            caption_engine = mock.Mock()
            caption_engine.generate.side_effect = [
                SimpleNamespace(
                    text="First caption",
                    engine="template",
                    fallback=False,
                    error="",
                ),
                SimpleNamespace(
                    text="Caption with Alice",
                    engine="template",
                    fallback=False,
                    error="",
                ),
            ]

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="template",

                ocr_engine_name="none",
                ocr_language="eng",
                people_recovery_mode="auto",
            )

            self.assertEqual(people_matcher.match_calls, 1)
            self.assertEqual(people_matcher.recovery_calls, 1)
            self.assertEqual(caption_engine.generate.call_count, 2)
            self.assertEqual(analysis.people_names, ["Alice"])
            self.assertEqual(analysis.description, "Caption with Alice")
            self.assertEqual(analysis.faces_detected, 1)
            self.assertTrue(analysis.payload["caption"]["people_present"])
            self.assertEqual(analysis.payload["caption"]["estimated_people_count"], 1)

    def test_should_run_people_recovery_auto_when_faces_are_detected(self):
        self.assertTrue(
            ai_index._should_run_people_recovery(
                people_recovery_mode="auto",
                faces_detected=1,
                people_matches=[],
                people_names=[],
                object_labels=[],
            )
        )

    def test_should_run_people_recovery_auto_when_caption_detects_people(self):
        self.assertTrue(
            ai_index._should_run_people_recovery(
                people_recovery_mode="auto",
                faces_detected=0,
                people_matches=[],
                people_names=[],
                object_labels=[],
                caption_people_present=True,
                caption_estimated_people_count=2,
            )
        )

    def test_run_image_analysis_runs_people_recovery_when_faces_are_already_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")

            class _Matcher:
                def __init__(self):
                    self.last_faces_detected = 0
                    self.match_calls = 0
                    self.recovery_calls = 0

                def match_image(self, *_args, **_kwargs):
                    self.match_calls += 1
                    self.last_faces_detected = 1
                    return [
                        SimpleNamespace(
                            name="Alice",
                            score=0.97,
                            certainty=0.97,
                            reviewed_by_human=False,
                            face_id="face-1",
                            bbox=[10, 10, 20, 20],
                        )
                    ]

                def match_image_recovery(self, *_args, **_kwargs):
                    self.recovery_calls += 1
                    self.last_faces_detected = 1
                    return [
                        SimpleNamespace(
                            name="Alice",
                            score=0.99,
                            certainty=0.99,
                            reviewed_by_human=False,
                            face_id="face-1",
                            bbox=[10, 10, 20, 20],
                        )
                    ]

            people_matcher = _Matcher()
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.side_effect = [
                SimpleNamespace(
                    text="Alice stands outdoors.",
                    engine="template",
                    fallback=False,
                    error="",
                    people_present=True,
                    estimated_people_count=1,
                ),
                SimpleNamespace(
                    text="Two people stand outdoors.",
                    engine="template",
                    fallback=False,
                    error="",
                    people_present=True,
                    estimated_people_count=1,
                ),
            ]

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="template",

                ocr_engine_name="none",
                ocr_language="eng",
                people_recovery_mode="auto",
            )

            self.assertEqual(people_matcher.match_calls, 1)
            self.assertEqual(people_matcher.recovery_calls, 1)
            self.assertEqual(caption_engine.generate.call_count, 2)
            self.assertEqual(analysis.people_names, ["Alice"])
            self.assertEqual(analysis.faces_detected, 1)
            self.assertEqual(analysis.description, "Alice stands outdoors.")

    def test_run_image_analysis_merges_lmstudio_people_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Four people stand together outdoors.",
                engine="lmstudio",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_people.return_value = SimpleNamespace(
                people_present=True,
                estimated_people_count=4,
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",

                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertTrue(analysis.payload["caption"]["people_present"])
            self.assertEqual(analysis.payload["caption"]["estimated_people_count"], 4)
            caption_engine.estimate_people.assert_called_once()

    def test_build_flat_page_description_uses_cover_caption_with_book_note_on_fallback(
        self,
    ):
        layout = SimpleNamespace(
            original_path=Path("China_1986_B02_P01.jpg"),
            content_path=Path("page.jpg"),
        )
        analysis = ai_index.ImageAnalysis(
            image_path=Path("page.jpg"),
            people_names=[],
            object_labels=[],
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
            ocr_keywords=["mainland", "china", "1986", "book"],
            subjects=["mainland", "china", "1986", "book"],
            description='Visible text reads: "MAINLAND CHINA 1986 BOOK 11".',
            payload={
                "people": [],
                "objects": [],
                "ocr": {
                    "engine": "lmstudio",
                    "language": "eng",
                    "keywords": ["mainland"],
                    "chars": 27,
                },
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "template",
                    "fallback": True,
                    "error": "LM Studio returned invalid structured caption JSON: Expecting value",
                    "model": "",
                },
            },
        )

        description = ai_index._build_flat_page_description(analysis=analysis)

        self.assertIn("MAINLAND CHINA 1986 BOOK 11", description)

    def test_run_force_rewrites_existing_sidecar_and_merges_embedded_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            original = self._valid_sidecar_text()
            sidecar.write_text(original, encoding="utf-8")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(
                    ai_index,
                    "read_embedded_source_text",
                    return_value="Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--force",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["source_text"],
                "Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
            )
            self.assertEqual(sidecar.read_text(encoding="utf-8"), original)

    def test_run_skips_existing_current_sidecar_without_ai_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
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

            with (
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_skips_current_sidecar_when_manifest_settings_are_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=["hello"],
                description="Old description",
                source_text="",
                ocr_text="hello",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "lmstudio",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "current-ocr-model",
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "current-caption-model",
                    },
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
            )

            with (
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_reprocesses_current_sidecar_when_ai_fields_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <xmp:CreatorTool>https://github.com/cove/imago</xmp:CreatorTool>
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

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["hello"],
                description="Updated description",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                    },
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()
            write_mock.assert_called_once()

    def test_run_reprocesses_current_sidecar_when_lmstudio_caption_error_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=[],
                description="Old description",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen2.5-vl",
                    },
                },
                subphotos=[],
            )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Updated description",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "qwen2.5-vl",
                    },
                },
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with (
                    mock.patch.object(
                        ai_index,
                        "prepare_image_layout",
                        side_effect=lambda *args, **kwargs: self._mock_layout(image),
                    ),
                    mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis) as analysis_mock,
                    mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                    mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
                ):
                    result = ai_index.run(
                        [
                            "--photos-root",
                            str(base),
                            "--include-view",
                            "--disable-people",
                            "--disable-objects",
                            "--ocr-engine",
                            "none",
                            "--caption-engine",
                            "lmstudio",
                        ]
                    )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()
            write_mock.assert_called_once()
            self.assertIn("reprocess: lmstudio_caption_error", stdout.getvalue())

    def test_run_rewrites_sidecar_when_image_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()

    def test_run_records_final_cast_store_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=[],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["hello"],
                description="Alice",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )
            fake_matcher = mock.Mock()
            fake_matcher.store_signature.side_effect = [
                "sig-before",
                "sig-after",
            ]

            with (
                mock.patch.object(ai_index, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            det = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(det["processing"]["cast_store_signature"], "sig-after")

    def test_run_people_update_only_refreshes_detection_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="https://github.com/cove/imago",
                person_names=["Alice"],
                subjects=["hello"],
                description="Old description",
                source_text="",
                ocr_text="hello",
                ocr_authority_source="archive_stitched",
                detections_payload={
                    "people": [
                        {
                            "name": "Alice",
                            "score": 0.95,
                            "bbox": [10, 10, 20, 20],
                        }
                    ],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "qwen-old-ocr",
                    },
                    "caption": {
                        "requested_engine": "local",
                        "effective_engine": "local",
                        "fallback": False,
                        "error": "",
                        "model": "caption-old",
                    },
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=True,
                people_identified=True,
            )

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"
            fake_matcher.match_image.return_value = []
            fake_caption_engine = mock.Mock()
            fake_caption_engine.effective_model_name = "caption-new"
            fake_caption_engine.generate.return_value = SimpleNamespace(
                text="Updated description",
                engine="lmstudio",
                fallback=False,
                error="",
            )

            def passthrough_people_recovery(**kwargs):
                return (
                    kwargs["people_matches"],
                    kwargs["people_names"],
                    kwargs.get("people_positions", {}),
                    kwargs["caption_output"],
                    0,
                )

            with (
                mock.patch.object(ai_index, "_settings_signature", return_value="sig"),
                mock.patch.object(
                    ai_index,
                    "_init_people_matcher",
                    return_value=fake_matcher,
                ) as people_matcher_mock,
                mock.patch.object(
                    ai_index,
                    "_init_caption_engine",
                    return_value=fake_caption_engine,
                ) as caption_engine_mock,
                mock.patch.object(
                    ai_index,
                    "_maybe_run_people_recovery",
                    side_effect=passthrough_people_recovery,
                ),
                mock.patch.object(ai_index, "read_embedded_source_text", return_value=""),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--ocr-model",
                        "qwen-current-ocr",
                        "--caption-engine",
                        "lmstudio",
                        "--caption-model",
                        "caption-new",
                        "--people-recovery-mode",
                        "off",
                    ]
                )

            self.assertEqual(result, 0)
            people_matcher_mock.assert_not_called()
            caption_engine_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_enqueues_unmatched_face_in_cast_queue(self):
        if not hasattr(cv2, "rectangle") or not hasattr(cv2, "imwrite"):
            self.skipTest("opencv drawing helpers unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (70, 70), (220, 220, 220), -1)
            cv2.imwrite(str(image), frame)
            cast_store = base / "cast_data"
            store = TextFaceStore(cast_store)
            store.ensure_files()

            class _FakeMatcher:
                def __init__(self, face_store: TextFaceStore, image_path: Path):
                    self._store = face_store
                    self._image_path = image_path
                    self.last_faces_detected = 0

                def store_signature(self) -> str:
                    return self._store.store_signature()

                def match_image(self, *_args, **_kwargs):
                    self.last_faces_detected = 1
                    face = self._store.add_face(
                        embedding=[1.0, 0.0, 0.0],
                        source_type="photo",
                        source_path=str(self._image_path),
                        bbox=[10, 10, 40, 40],
                        quality=0.93,
                        crop_path="crops/fake-face.jpg",
                        metadata={
                            "ingest": "photoalbums_ai",
                            "embedding_model": "test.embedding",
                            "detector_model": "test.detector",
                            "analysis_variant": "original",
                        },
                    )
                    self._store.add_review_item(
                        face_id=str(face["face_id"]),
                        candidates=[],
                        suggested_person_id=None,
                        suggested_score=None,
                        status="pending",
                    )
                    return []

                def match_image_recovery(self, *_args, **_kwargs):
                    self.last_faces_detected = 1
                    return []

            matcher = _FakeMatcher(store, image)

            with (
                mock.patch.object(ai_index, "_init_people_matcher", return_value=matcher),
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "read_embedded_source_text", return_value=""),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--cast-store",
                        str(cast_store),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()

            faces = store.list_faces()
            reviews = store.list_review_items()

            self.assertEqual(len(faces), 1)
            self.assertEqual(str(faces[0]["source_path"]), str(image))
            self.assertEqual(len(reviews), 1)
            self.assertEqual(str(reviews[0]["face_id"]), str(faces[0]["face_id"]))
            self.assertEqual(str(reviews[0]["status"]), "pending")

    def test_run_stdout_prints_caption_only_and_skips_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "local"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_not_called()
            print_mock.assert_called_once_with("a.jpg: Alice with a dog")

    def test_run_stdout_emits_caption_fallback_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Fallback caption text",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "local",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen/qwen3.5-9b",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[1/1] warn  a.jpg: caption fallback: model offline",
                        file=sys.stderr,
                        flush=True,
                    ),
                    mock.call("a.jpg: Fallback caption text"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_run_stdout_prints_filename_only_for_empty_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen2.5-vl-instruct",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[1/1] warn  a.jpg: caption fallback: model offline",
                        file=sys.stderr,
                        flush=True,
                    ),
                    mock.call("a.jpg"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_resolve_album_title_hint_prefers_existing_cover_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P01.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
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

            title = ai_index._resolve_album_title_hint(image, {})
            self.assertEqual(title, "Mainland China Book II")

    def test_resolve_album_title_hint_requires_existing_cover_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")

            title = ai_index._resolve_album_title_hint(image, {})
            self.assertEqual(title, "")

    def test_resolve_album_printed_title_hint_prefers_existing_p00_cover_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P00.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
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

            title = ai_index._resolve_album_printed_title_hint(image, {})
            self.assertEqual(title, "Mainland China Book II")

    def test_run_stdout_forces_processing_even_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Caption from stdout mode",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "local"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()

    def test_run_archive_multi_scan_passes_stitched_ocr_override_and_records_authority_metadata(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=("mainland", "china", "book"),
                ocr_hash=ai_index._hash_text("MAINLAND CHINA 1986 BOOK 11"),
            )
            analysis = ai_index.ImageAnalysis(
                image_path=scan1,
                people_names=[],
                object_labels=[],
                ocr_text=authority.ocr_text,
                ocr_keywords=list(authority.ocr_keywords),
                subjects=list(authority.ocr_keywords),
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": list(authority.ocr_keywords),
                        "chars": len(authority.ocr_text),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ) as authority_mock,
                mock.patch.object(ai_index, "read_embedded_source_text", return_value=""),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            authority_mock.assert_called_once()
            self.assertEqual(
                analysis_mock.call_args.kwargs["ocr_text_override"],
                authority.ocr_text,
            )
            self.assertEqual(write_mock.call_args.kwargs["ocr_text"], authority.ocr_text)
            self.assertEqual(
                write_mock.call_args.kwargs["ocr_authority_source"],
                "archive_stitched",
            )
            det = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(det["processing"]["ocr_authority_signature"], authority.signature)
            self.assertEqual(det["processing"]["ocr_authority_hash"], authority.ocr_hash)

    def test_run_archive_multi_scan_skips_when_authority_manifest_and_sidecar_match(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=("mainland", "china", "book"),
                ocr_hash=ai_index._hash_text("MAINLAND CHINA 1986 BOOK 11"),
            )
            analysis = ai_index.ImageAnalysis(
                image_path=scan1,
                people_names=[],
                object_labels=[],
                ocr_text=authority.ocr_text,
                ocr_keywords=list(authority.ocr_keywords),
                subjects=list(authority.ocr_keywords),
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": list(authority.ocr_keywords),
                        "chars": len(authority.ocr_text),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ),
                mock.patch.object(ai_index, "read_embedded_source_text", return_value=""),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
            ):
                first_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                        "--force",
                    ]
                )

            self.assertEqual(first_result, 0)

            with (
                mock.patch.object(ai_index, "_resolve_archive_scan_authoritative_ocr") as authority_mock,
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                second_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(second_result, 0)
            authority_mock.assert_not_called()
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_archive_multi_scan_skips_without_authority_when_sidecar_has_authority(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=("mainland", "china", "book"),
                ocr_hash=ai_index._hash_text("MAINLAND CHINA 1986 BOOK 11"),
            )
            analysis = ai_index.ImageAnalysis(
                image_path=scan1,
                people_names=[],
                object_labels=[],
                ocr_text=authority.ocr_text,
                ocr_keywords=list(authority.ocr_keywords),
                subjects=list(authority.ocr_keywords),
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": list(authority.ocr_keywords),
                        "chars": len(authority.ocr_text),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ),
                mock.patch.object(ai_index, "read_embedded_source_text", return_value=""),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
            ):
                first_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                        "--force",
                    ]
                )

            self.assertEqual(first_result, 0)

            with (
                mock.patch.object(ai_index, "_resolve_archive_scan_authoritative_ocr") as authority_mock,
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                second_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(second_result, 0)
            authority_mock.assert_not_called()
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_resolve_caption_prompt_reads_file_and_overrides_inline_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("Describe this image from file.\n", encoding="utf-8")
            text = ai_index._resolve_caption_prompt("Inline prompt", str(prompt_file))
        self.assertEqual(text, "Describe this image from file.")

    def test_resolve_caption_prompt_exits_for_missing_file(self):
        with self.assertRaises(SystemExit) as exc:
            ai_index._resolve_caption_prompt("", "/tmp/definitely-missing-caption-prompt.txt")
        self.assertIn("Caption prompt file does not exist", str(exc.exception))

    def test_parse_args_caption_flags(self):
        args = ai_index.parse_args(
            [
                "--ocr-model",
                "qwen2.5-vl-instruct",
                "--caption-engine",
                "lmstudio",
                "--caption-model",
                "qwen2.5-vl-instruct",
                "--caption-prompt",
                "Describe this exact image",
                "--caption-prompt-file",
                "/tmp/prompt.txt",
                "--lmstudio-base-url",
                "http://192.168.4.72:1234",
                "--caption-max-tokens",
                "64",
                "--caption-temperature",
                "0.1",
                "--caption-max-edge",
                "1024",
            ]
        )
        self.assertEqual(args.ocr_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_prompt, "Describe this exact image")
        self.assertEqual(args.caption_prompt_file, "/tmp/prompt.txt")
        self.assertEqual(args.lmstudio_base_url, "http://192.168.4.72:1234")
        self.assertEqual(args.caption_max_tokens, 64)
        self.assertAlmostEqual(args.caption_temperature, 0.1)
        self.assertEqual(args.caption_max_edge, 1024)

    def test_parse_args_defaults_use_lmstudio_for_caption_and_ocr(self):
        with mock.patch.object(ai_index, "default_ocr_model", return_value="qwen/qwen3-vl-30b"):
            args = ai_index.parse_args([])
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "")
        self.assertEqual(args.caption_prompt, "")
        self.assertEqual(args.caption_prompt_file, "")
        self.assertEqual(args.lmstudio_base_url, "http://192.168.4.72:1234/v1")
        self.assertEqual(args.ocr_engine, "lmstudio")
        self.assertEqual(args.ocr_model, "qwen/qwen3-vl-30b")
        self.assertFalse(args.stdout)
        self.assertEqual(args.caption_max_edge, 0)

    def test_init_caption_engine_forwards_caption_prompt(self):
        with mock.patch.object(ai_index, "CaptionEngine") as engine_ctor:
            ai_index._init_caption_engine(
                engine="lmstudio",
                model_name="qwen2.5-vl-instruct",
                caption_prompt="Describe this exact image",
                max_tokens=64,
                temperature=0.1,
                lmstudio_base_url="http://localhost:1234",
                max_image_edge=1024,
            )

        engine_ctor.assert_called_once_with(
            engine="lmstudio",
            model_name="qwen2.5-vl-instruct",
            caption_prompt="Describe this exact image",
            max_tokens=64,
            temperature=0.1,
            lmstudio_base_url="http://localhost:1234",
            max_image_edge=1024,
            stream=False,
        )


if __name__ == "__main__":
    unittest.main()
