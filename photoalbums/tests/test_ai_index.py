import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from cast.storage import TextFaceStore
from photoalbums.lib import ai_index
from photoalbums.lib.ai_people import CastPeopleMatcher
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
            content_bounds=SimpleNamespace(
                as_dict=lambda: {"x": 0, "y": 0, "width": 0, "height": 0}
            ),
            footer_trimmed=False,
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

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = {
                "/a.jpg": {"image_path": "/a.jpg", "size": 1, "mtime_ns": 2},
                "/b.jpg": {"image_path": "/b.jpg", "size": 3, "mtime_ns": 4},
            }
            ai_index.save_manifest(path, rows)
            loaded = ai_index.load_manifest(path)
            self.assertEqual(loaded, rows)

            raw = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw), 2)
            self.assertTrue(all(isinstance(json.loads(line), dict) for line in raw))

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

            next_ns = (
                max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            )
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

            image.with_suffix(".xmp").write_text(
                self._valid_sidecar_text(), encoding="utf-8"
            )
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

    def test_needs_processing_skips_when_manifest_missing_but_valid_sidecar_exists(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(
                self._valid_sidecar_text(), encoding="utf-8"
            )

            self.assertFalse(ai_index.needs_processing(image, None, force=False))
            self.assertTrue(ai_index.needs_processing(image, None, force=True))

    def test_needs_processing_ignores_tiny_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text("tiny", encoding="utf-8")

            self.assertTrue(ai_index.needs_processing(image, None, force=False))

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
                requested_caption_model="",
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

            with mock.patch.object(
                ai_index, "_prepare_ai_model_image", side_effect=fake_prepare
            ):
                ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    requested_caption_model="",
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=image,
                bbox_offset=(0, 0),
                hint_text="hello",
            )
            ocr_engine.read_text.assert_called_once_with(scaled)
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
            )

    def test_run_image_analysis_records_gps_location_from_caption_output(self):
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
                text="Mogao Caves in Dunhuang, China (39°47′15″N 100°18′26″E).",
                engine="lmstudio",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
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
                requested_caption_model="",
                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.7875)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 100.307222)

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
                requested_caption_model="",
                ocr_engine_name="none",
                ocr_language="eng",
                geocoder=geocoder,
            )

            geocoder.geocode.assert_called_once_with(
                "Mogao Caves, Dunhuang, Gansu, China"
            )
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

    def test_build_page_payload_uses_cover_caption_for_fallback_text_page(self):
        content_bounds = SimpleNamespace(
            as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100}
        )
        subphoto_bounds = SimpleNamespace(
            as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100}
        )
        layout = SimpleNamespace(
            kind="page_view",
            page_like=True,
            split_mode="auto",
            content_bounds=content_bounds,
            content_path=Path("page.jpg"),
            original_path=Path("China_1986_B02_P01.jpg"),
            footer_trimmed=False,
            split_applied=False,
            fallback_used=True,
            subphotos=[
                SimpleNamespace(index=1, bounds=subphoto_bounds, path=Path("page.jpg"))
            ],
        )
        sub_result = ai_index.ImageAnalysis(
            image_path=Path("page.jpg"),
            people_names=[],
            object_labels=[],
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
            ocr_keywords=["mainland", "china", "1986", "book"],
            subjects=["mainland", "china", "1986", "book"],
            description="Subphoto caption",
            payload={
                "people": [],
                "objects": [],
                "ocr": {
                    "engine": "qwen",
                    "language": "eng",
                    "keywords": [],
                    "chars": 27,
                },
                "caption": {"engine": "template"},
            },
        )

        _people, _objects, _subjects, description, _payload, _subphotos = (
            ai_index._build_page_payload(
                layout=layout,
                sub_results=[sub_result],
                page_ocr_text="MAINLAND CHINA 1986 BOOK 11",
                page_ocr_keywords=["mainland", "china", "1986", "book"],
                requested_caption_engine="template",
            )
        )

        self.assertIn("Subphoto caption", description)
        self.assertNotIn("contains 1 photo(s)", description)

    def test_build_page_payload_marks_family_album_pages(self):
        content_bounds = SimpleNamespace(
            as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100}
        )
        subphoto_bounds = SimpleNamespace(
            as_dict=lambda: {"x": 0, "y": 0, "width": 50, "height": 50}
        )
        layout = SimpleNamespace(
            kind="page_view",
            page_like=True,
            split_mode="auto",
            content_bounds=content_bounds,
            content_path=Path("family_page.jpg"),
            original_path=Path("Family_View") / "Family_1980-1985_B08_P12.jpg",
            footer_trimmed=False,
            split_applied=True,
            fallback_used=False,
            subphotos=[
                SimpleNamespace(
                    index=1, bounds=subphoto_bounds, path=Path("photo1.jpg")
                ),
                SimpleNamespace(
                    index=2, bounds=subphoto_bounds, path=Path("photo2.jpg")
                ),
            ],
        )
        sub_result = ai_index.ImageAnalysis(
            image_path=Path("photo1.jpg"),
            people_names=[],
            object_labels=[],
            ocr_text="",
            ocr_keywords=[],
            subjects=[],
            description="Subphoto caption",
            payload={
                "people": [],
                "objects": [],
                "ocr": {
                    "engine": "qwen",
                    "language": "eng",
                    "keywords": [],
                    "chars": 0,
                },
                "caption": {"engine": "template"},
            },
        )

        _people, _objects, _subjects, description, _payload, _subphotos = (
            ai_index._build_page_payload(
                layout=layout,
                sub_results=[sub_result, sub_result],
                page_ocr_text="",
                page_ocr_keywords=[],
                requested_caption_engine="template",
            )
        )

        self.assertIn(
            "This page from Family Book VIII, a Family Photo Album, contains 2 photo(s).",
            description,
        )

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

        with mock.patch.object(ai_index, "looks_like_album_cover", return_value=True):
            description = ai_index._build_flat_page_description(
                layout=layout,
                analysis=analysis,
                requested_caption_engine="lmstudio",
            )

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
            manifest = base / "manifest.jsonl"

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
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
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
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--force",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["album_title"], "Family Book I"
            )
            self.assertEqual(
                write_mock.call_args.kwargs["source_text"],
                "Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
            )
            self.assertEqual(sidecar.read_text(encoding="utf-8"), original)

    def test_run_skips_processing_when_valid_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="imago-photoalbums-ai-index",
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
            manifest = base / "manifest.jsonl"

            with (
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
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
      <xmp:CreatorTool>imago-photoalbums-ai-index</xmp:CreatorTool>
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
            manifest = base / "manifest.jsonl"

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
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ) as analysis_mock,
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()
            write_mock.assert_called_once()

    def test_run_rewrites_sidecar_when_image_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            manifest = base / "manifest.jsonl"

            stat = image.stat()
            ai_index.save_manifest(
                manifest,
                {
                    str(image): {
                        "image_path": str(image),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "sidecar_path": str(sidecar),
                        "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                        "settings_signature": "",
                    }
                },
            )

            next_ns = (
                max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            )
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
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
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
            manifest = base / "manifest.jsonl"

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
                "sig-final",
            ]

            with (
                mock.patch.object(
                    ai_index, "_init_people_matcher", return_value=fake_matcher
                ),
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            rows = ai_index.load_manifest(manifest)
            self.assertEqual(rows[str(image)]["cast_store_signature"], "sig-final")

    def test_run_enqueues_unmatched_face_in_cast_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (70, 70), (220, 220, 220), -1)
            cv2.imwrite(str(image), frame)
            manifest = base / "manifest.jsonl"
            cast_store = base / "cast_data"
            store = TextFaceStore(cast_store)
            store.ensure_files()
            matcher = CastPeopleMatcher(cast_store_dir=cast_store, max_faces=1)

            with (
                mock.patch.object(
                    ai_index, "_init_people_matcher", return_value=matcher
                ),
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(
                    ai_index, "read_embedded_source_text", return_value=""
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
                mock.patch.object(
                    matcher, "_detect_faces", return_value=[(10, 10, 40, 40)]
                ),
                mock.patch.object(
                    matcher._ingestor, "is_valid_face_crop", return_value=True
                ),
                mock.patch.object(
                    matcher, "_arcface_embed", return_value=[1.0, 0.0, 0.0]
                ),
                mock.patch.object(matcher, "_estimate_quality", return_value=0.93),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--cast-store",
                        str(cast_store),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
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
            manifest = base / "manifest.jsonl"

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
                    "caption": {"engine": "qwen"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
                mock.patch.object(ai_index, "save_manifest") as save_mock,
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_not_called()
            save_mock.assert_not_called()
            print_mock.assert_called_once_with("a.jpg: Alice with a dog")

    def test_run_stdout_emits_caption_fallback_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

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
                        "requested_engine": "qwen",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
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
            manifest = base / "manifest.jsonl"

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
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
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

    def test_run_stdout_uses_built_in_qwen_prompt_for_page_like_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "Family_1986_B02_P01.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            content_bounds = SimpleNamespace(
                as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100}
            )
            subphoto_bounds = SimpleNamespace(
                as_dict=lambda: {"x": 0, "y": 0, "width": 80, "height": 80}
            )

            @contextmanager
            def mock_page_layout(*args, **kwargs):
                yield SimpleNamespace(
                    kind="page_view",
                    split_mode="auto",
                    content_bounds=content_bounds,
                    content_path=image,
                    original_path=image,
                    page_like=True,
                    footer_trimmed=False,
                    split_applied=False,
                    fallback_used=True,
                    subphotos=[
                        SimpleNamespace(index=1, bounds=subphoto_bounds, path=image)
                    ],
                )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Subphoto caption",
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
                        "requested_engine": "qwen",
                        "effective_engine": "qwen",
                        "fallback": False,
                        "error": "",
                        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    },
                },
            )
            fake_caption_engine = mock.Mock()
            fake_caption_engine.generate.return_value = SimpleNamespace(
                text="Describe this page exactly",
                engine="qwen",
                fallback=False,
                error="",
            )
            fake_ocr_engine = mock.Mock()
            fake_ocr_engine.read_text.return_value = ""

            with (
                mock.patch.object(
                    ai_index, "prepare_image_layout", side_effect=mock_page_layout
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_init_caption_engine", return_value=fake_caption_engine
                ),
                mock.patch.object(ai_index, "OCREngine", return_value=fake_ocr_engine),
                mock.patch.object(ai_index, "extract_keywords", return_value=[]),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            fake_caption_engine.generate.assert_called_once_with(
                image_path=image,
                people=[],
                objects=[],
                ocr_text="",
                source_path=image,
                album_title="Family Book II",
                printed_album_title="",
                photo_count=1,
            )
            print_mock.assert_called_once_with(
                "Family_1986_B02_P01.jpg: Describe this page exactly"
            )

    def test_run_stdout_keeps_page_summary_when_page_caption_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "China_1986_B02_View"
            photos.mkdir()
            image = photos / "China_1986_B02_P01.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            content_bounds = SimpleNamespace(
                as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100}
            )
            subphoto_bounds = SimpleNamespace(
                as_dict=lambda: {"x": 0, "y": 0, "width": 80, "height": 80}
            )

            @contextmanager
            def mock_page_layout(*args, **kwargs):
                yield SimpleNamespace(
                    kind="page_view",
                    split_mode="auto",
                    content_bounds=content_bounds,
                    content_path=image,
                    original_path=image,
                    page_like=True,
                    footer_trimmed=False,
                    split_applied=False,
                    fallback_used=True,
                    subphotos=[
                        SimpleNamespace(index=1, bounds=subphoto_bounds, path=image)
                    ],
                )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "1986", "book"],
                subjects=["mainland", "china", "1986", "book"],
                description="Subphoto caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "lmstudio",
                        "language": "eng",
                        "keywords": [],
                        "chars": 27,
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "LMSTUDIO returned empty output.",
                        "model": "",
                    },
                },
            )
            fake_caption_engine = mock.Mock()
            fake_caption_engine.generate.return_value = SimpleNamespace(
                text='Visible text reads: "MAINLAND CHINA 1986 BOOK 11".',
                engine="template",
                fallback=True,
                error="LMSTUDIO returned empty output.",
            )
            fake_ocr_engine = mock.Mock()
            fake_ocr_engine.read_text.return_value = "MAINLAND CHINA 1986 BOOK 11"

            with (
                mock.patch.object(
                    ai_index, "prepare_image_layout", side_effect=mock_page_layout
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ),
                mock.patch.object(
                    ai_index, "_init_caption_engine", return_value=fake_caption_engine
                ),
                mock.patch.object(ai_index, "OCREngine", return_value=fake_ocr_engine),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "lmstudio",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[1/1] warn  China_1986_B02_P01.jpg: caption fallback: LMSTUDIO returned empty output.",
                        file=sys.stderr,
                        flush=True,
                    ),
                    mock.call("China_1986_B02_P01.jpg: Subphoto caption"),
                ]
            )

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
            self.assertEqual(title, "Mainland China Book 11")

    def test_run_stdout_forces_processing_even_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(
                self._valid_sidecar_text(), encoding="utf-8"
            )
            manifest = base / "manifest.jsonl"

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
                    "caption": {"engine": "qwen"},
                },
            )

            with (
                mock.patch.object(
                    ai_index,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(
                    ai_index, "_run_image_analysis", return_value=analysis
                ) as analysis_mock,
                mock.patch.object(
                    ai_index, "_build_flat_payload", return_value=analysis.payload
                ),
                mock.patch("builtins.print"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()

    def test_build_description(self):
        text = ai_index.build_description(
            people=["Alice", "Bob"],
            objects=["dog", "car"],
            ocr_text="Hello world from a sign",
        )
        self.assertIn("Alice", text)
        self.assertIn("dog", text)
        self.assertIn("Visible text reads:", text)

    def test_resolve_caption_prompt_reads_file_and_overrides_inline_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("Describe this image from file.\n", encoding="utf-8")
            text = ai_index._resolve_caption_prompt("Inline prompt", str(prompt_file))
        self.assertEqual(text, "Describe this image from file.")

    def test_resolve_caption_prompt_exits_for_missing_file(self):
        with self.assertRaises(SystemExit) as exc:
            ai_index._resolve_caption_prompt(
                "", "/tmp/definitely-missing-caption-prompt.txt"
            )
        self.assertIn("Caption prompt file does not exist", str(exc.exception))

    def test_parse_args_caption_flags(self):
        args = ai_index.parse_args(
            [
                "--caption-engine",
                "lmstudio",
                "--caption-model",
                "qwen2.5-vl-instruct",
                "--caption-prompt",
                "Describe this exact image",
                "--caption-prompt-file",
                "/tmp/prompt.txt",
                "--lmstudio-base-url",
                "http://localhost:1234",
                "--caption-max-tokens",
                "64",
                "--caption-temperature",
                "0.1",
                "--caption-max-edge",
                "1024",
                "--qwen-attn-implementation",
                "sdpa",
                "--qwen-min-pixels",
                "131072",
                "--qwen-max-pixels",
                "524288",
            ]
        )
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_prompt, "Describe this exact image")
        self.assertEqual(args.caption_prompt_file, "/tmp/prompt.txt")
        self.assertEqual(args.lmstudio_base_url, "http://localhost:1234")
        self.assertEqual(args.caption_max_tokens, 64)
        self.assertAlmostEqual(args.caption_temperature, 0.1)
        self.assertEqual(args.caption_max_edge, 1024)
        self.assertEqual(args.qwen_attn_implementation, "sdpa")
        self.assertEqual(args.qwen_min_pixels, 131072)
        self.assertEqual(args.qwen_max_pixels, 524288)

    def test_parse_args_defaults_use_qwen_and_qwen_ocr(self):
        args = ai_index.parse_args([])
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "")
        self.assertEqual(args.caption_prompt, "")
        self.assertEqual(args.caption_prompt_file, "")
        self.assertEqual(args.lmstudio_base_url, "http://192.168.4.72:1234/v1")
        self.assertEqual(args.ocr_engine, "lmstudio")
        self.assertFalse(args.stdout)
        self.assertEqual(args.qwen_attn_implementation, "auto")
        self.assertEqual(args.qwen_min_pixels, 0)
        self.assertEqual(args.qwen_max_pixels, 0)
        self.assertEqual(args.caption_max_edge, 0)

    def test_init_caption_engine_forwards_caption_prompt(self):
        with mock.patch.object(ai_index, "CaptionEngine") as engine_ctor:
            ai_index._init_caption_engine(
                engine="lmstudio",
                model_name="qwen2.5-vl-instruct",
                caption_prompt="Describe this exact image",
                max_tokens=64,
                temperature=0.1,
                qwen_attn_implementation="sdpa",
                qwen_min_pixels=131072,
                qwen_max_pixels=524288,
                lmstudio_base_url="http://localhost:1234",
                max_image_edge=1024,
            )

        engine_ctor.assert_called_once_with(
            engine="lmstudio",
            model_name="qwen2.5-vl-instruct",
            caption_prompt="Describe this exact image",
            max_tokens=64,
            temperature=0.1,
            qwen_attn_implementation="sdpa",
            qwen_min_pixels=131072,
            qwen_max_pixels=524288,
            lmstudio_base_url="http://localhost:1234",
            max_image_edge=1024,
            fallback_to_template=True,
            stream=False,
        )


if __name__ == "__main__":
    unittest.main()
